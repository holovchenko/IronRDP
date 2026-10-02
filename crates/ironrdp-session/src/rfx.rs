use core::cmp::min;

use ironrdp_graphics::color_conversion::{self, YCbCrBuffer};
use ironrdp_graphics::image_processing::PixelFormat;
use ironrdp_graphics::rectangle_processing::Region;
use ironrdp_graphics::{dwt, quantization, rlgr, subband_reconstruction};
use ironrdp_pdu::codecs::rfx::{self, EntropyAlgorithm, Quant, RfxRectangle, Tile};
use ironrdp_pdu::geometry::{InclusiveRectangle, Rectangle as _};
use ironrdp_pdu::{Decode as _, ReadCursor, decode_cursor};
use tracing::{instrument, trace};

use crate::image::{DecodedImage, exclusive_to_inclusive};
use crate::{SessionResult, custom_err, general_err, reason_err};

const TILE_SIZE: u16 = 64;

pub type FrameId = u32;

pub struct DecodingContext {
    context: rfx::ContextPdu,
    channels: rfx::ChannelsPdu,
    decoding_tiles: DecodingTileContext,
}

impl Default for DecodingContext {
    fn default() -> Self {
        Self {
            context: rfx::ContextPdu {
                flags: rfx::OperatingMode::empty(),
                entropy_algorithm: EntropyAlgorithm::Rlgr1,
            },
            channels: rfx::ChannelsPdu(Vec::new()),
            decoding_tiles: DecodingTileContext::new(),
        }
    }
}

impl DecodingContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn decode(
        &mut self,
        image: &mut DecodedImage,
        destination: &InclusiveRectangle,
        input: &mut ReadCursor<'_>,
    ) -> SessionResult<(FrameId, InclusiveRectangle)> {
        loop {
            let block = rfx::Block::decode(input).map_err(|e| custom_err!("decode block", e))?;

            match block {
                rfx::Block::Sync(_) => {
                    self.process_sync(input)?;
                }
                rfx::Block::CodecChannel(rfx::CodecChannel::FrameBegin(f)) => {
                    return self.process_frame(f, input, image, destination);
                }
                _ => {
                    return Err(reason_err!(
                        "rfx::DecodingContext",
                        "unexpected RFX block type: {:?}",
                        block.block_type()
                    ));
                }
            }
        }
    }

    fn process_sync(&mut self, input: &mut ReadCursor<'_>) -> SessionResult<()> {
        self.process_headers(input)
    }

    fn process_headers(&mut self, input: &mut ReadCursor<'_>) -> SessionResult<()> {
        let mut context = None;
        let mut channels = None;

        // headers can appear in any order: CodecVersions, Channels, Context
        for _ in 0..3 {
            match decode_cursor(input).map_err(|e| custom_err!("decode headers", e))? {
                rfx::Block::CodecChannel(rfx::CodecChannel::Context(c)) => context = Some(c),
                rfx::Block::Channels(c) => channels = Some(c),
                rfx::Block::CodecVersions(_) => (),
                _ => {
                    return Err(general_err!("unexpected RFX block type"));
                }
            }
        }

        let context = context.ok_or_else(|| general_err!("context header is missing"))?;
        let channels = channels.ok_or_else(|| general_err!("channels header is missing"))?;

        if channels.0.is_empty() {
            return Err(general_err!("no RFX channel announced"));
        }

        self.context = context;
        self.channels = channels;

        Ok(())
    }

    #[instrument(skip_all)]
    fn process_frame(
        &mut self,
        frame_begin: rfx::FrameBeginPdu,
        input: &mut ReadCursor<'_>,
        image: &mut DecodedImage,
        destination: &InclusiveRectangle,
    ) -> SessionResult<(FrameId, InclusiveRectangle)> {
        let channel = self
            .channels
            .0
            .first()
            .ok_or_else(|| general_err!("no RFX channel found"))?;
        let width = channel.width.try_into().map_err(|_| general_err!("invalid width"))?;
        let height = channel.height.try_into().map_err(|_| general_err!("invalid height"))?;
        let entropy_algorithm = self.context.entropy_algorithm;

        let region: rfx::Block<'_> = decode_cursor(input).map_err(|e| custom_err!("decode region", e))?;
        let mut region = match region {
            rfx::Block::CodecChannel(rfx::CodecChannel::Region(region)) => region,
            _ => return Err(general_err!("unexpected block type")),
        };
        let tile_set: rfx::Block<'_> = decode_cursor(input).map_err(|e| custom_err!("decode tile_set", e))?;
        let tile_set = match tile_set {
            rfx::Block::CodecChannel(rfx::CodecChannel::TileSet(t)) => t,
            _ => return Err(general_err!("unexpected block type")),
        };
        let frame_end: rfx::Block<'_> = decode_cursor(input).map_err(|e| custom_err!("decode frame_end", e))?;
        if !matches!(frame_end, rfx::Block::CodecChannel(rfx::CodecChannel::FrameEnd(_))) {
            return Err(general_err!("unexpected block type"));
        }

        if region.rectangles.is_empty() {
            region.rectangles = vec![RfxRectangle {
                x: 0,
                y: 0,
                width,
                height,
            }];
        }
        let region = region;

        trace!(frame_index = frame_begin.index);
        trace!(destination_rectangle = ?destination);
        trace!(context = ?self.context);
        trace!(channels = ?self.channels);
        trace!(?region);

        let clipping_rectangles = clipping_rectangles(region.rectangles.as_slice(), destination, width, height);
        trace!("Clipping rectangles: {:?}", clipping_rectangles);

        let mut final_update_rectangle = if clipping_rectangles.rectangles.is_empty() {
            InclusiveRectangle::empty()
        } else {
            exclusive_to_inclusive(&clipping_rectangles.extents)
        };

        for (update_rectangle, tile_data) in tiles_to_rectangles(tile_set.tiles.as_slice(), destination)
            .zip(map_tiles_data(tile_set.tiles.as_slice(), tile_set.quants.as_slice()))
        {
            decode_tile(
                &tile_data,
                entropy_algorithm,
                self.decoding_tiles.tile_output.as_mut(),
                self.decoding_tiles.ycbcr_buffer.as_mut(),
                self.decoding_tiles.ycbcr_temp_buffer.as_mut(),
            )?;

            let current_update_rectangle = image.apply_tile(
                &self.decoding_tiles.tile_output,
                PixelFormat::RgbA32,
                &clipping_rectangles,
                &update_rectangle,
            )?;

            final_update_rectangle = final_update_rectangle.union(&current_update_rectangle);
        }

        Ok((frame_begin.index, final_update_rectangle))
    }
}

#[derive(Debug, Clone)]
struct DecodingTileContext {
    tile_output: Vec<u8>,
    ycbcr_buffer: Vec<Vec<i16>>,
    ycbcr_temp_buffer: Vec<i16>,
}

impl DecodingTileContext {
    fn new() -> Self {
        let tile_size = usize::from(TILE_SIZE);
        Self {
            tile_output: vec![0; tile_size * tile_size * 4],
            ycbcr_buffer: vec![vec![0; tile_size * tile_size]; 3],
            ycbcr_temp_buffer: vec![0; tile_size * tile_size],
        }
    }
}

fn decode_tile(
    tile: &TileData<'_>,
    entropy_algorithm: EntropyAlgorithm,
    output: &mut [u8],
    ycbcr_temp: &mut [Vec<i16>],
    temp: &mut [i16],
) -> SessionResult<()> {
    for ((quant, data), ycbcr_buffer) in tile.quants.iter().zip(tile.data.iter()).zip(ycbcr_temp.iter_mut()) {
        decode_component(quant, entropy_algorithm, data, ycbcr_buffer.as_mut_slice(), temp)?;
    }

    let ycbcr_buffer = YCbCrBuffer {
        y: ycbcr_temp[0].as_slice(),
        cb: ycbcr_temp[1].as_slice(),
        cr: ycbcr_temp[2].as_slice(),
    };

    color_conversion::ycbcr_to_rgba(ycbcr_buffer, output).map_err(|e| custom_err!("decode_tile", e))?;

    Ok(())
}

fn decode_component(
    quant: &Quant,
    entropy_algorithm: EntropyAlgorithm,
    data: &[u8],
    output: &mut [i16],
    temp: &mut [i16],
) -> SessionResult<()> {
    rlgr::decode(entropy_algorithm, data, output).map_err(|e| custom_err!("decode_component", e))?;
    subband_reconstruction::decode(&mut output[4032..]);
    quantization::decode(output, quant);
    dwt::decode(output, temp);

    Ok(())
}

/// Builds the clipping region with exclusive right/bottom bounds, as FreeRDP does
/// (`region16_union_rect` with `right = x + width`). `Region` ports region16 and works on
/// exclusive bounds despite the `InclusiveRectangle` type; inclusive bounds drop one-row
/// rectangles that touch a band.
fn clipping_rectangles(
    rectangles: &[RfxRectangle],
    destination: &InclusiveRectangle,
    width: u16,
    height: u16,
) -> Region {
    let mut clipping_rectangles = Region::new();

    for r in rectangles {
        let left = destination.left.saturating_add(r.x);
        let top = destination.top.saturating_add(r.y);
        let clipped = InclusiveRectangle {
            left: min(left, width),
            top: min(top, height),
            right: min(left.saturating_add(r.width), width),
            bottom: min(top.saturating_add(r.height), height),
        };
        if clipped.left < clipped.right && clipped.top < clipped.bottom {
            clipping_rectangles.union_rectangle(clipped);
        }
    }

    clipping_rectangles
}

/// Tile rectangles with exclusive right/bottom bounds (`x + 64`), matching the clipping region.
/// Server-controlled tile indices saturate instead of overflowing.
fn tiles_to_rectangles<'a>(
    tiles: &'a [Tile<'_>],
    destination: &'a InclusiveRectangle,
) -> impl Iterator<Item = InclusiveRectangle> + 'a {
    tiles.iter().map(|t| {
        let left = destination.left.saturating_add(t.x.saturating_mul(TILE_SIZE));
        let top = destination.top.saturating_add(t.y.saturating_mul(TILE_SIZE));
        InclusiveRectangle {
            left,
            top,
            right: left.saturating_add(TILE_SIZE),
            bottom: top.saturating_add(TILE_SIZE),
        }
    })
}

fn map_tiles_data<'a>(tiles: &[Tile<'a>], quants: &[Quant]) -> Vec<TileData<'a>> {
    tiles
        .iter()
        .map(|t| TileData {
            quants: [
                quants[usize::from(t.y_quant_index)].clone(),
                quants[usize::from(t.cb_quant_index)].clone(),
                quants[usize::from(t.cr_quant_index)].clone(),
            ],
            data: [t.y_data, t.cb_data, t.cr_data],
        })
        .collect()
}

struct TileData<'a> {
    quants: [Quant; 3],
    data: [&'a [u8]; 3],
}

#[cfg(test)]
mod tests {
    use super::*;

    const IMAGE_SIZE: u16 = 128;

    fn rfx_rect(x: u16, y: u16, width: u16, height: u16) -> RfxRectangle {
        RfxRectangle { x, y, width, height }
    }

    /// Applies one fully opaque tile at tile coordinates `(tile_x, tile_y)` through the region clipping and
    /// returns the image, whose touched pixels are the non-zero ones.
    fn apply_tile_in_region(rectangles: &[RfxRectangle], tile_x: u16, tile_y: u16) -> DecodedImage {
        let destination = InclusiveRectangle {
            left: 0,
            top: 0,
            right: IMAGE_SIZE - 1,
            bottom: IMAGE_SIZE - 1,
        };
        let clipping = clipping_rectangles(rectangles, &destination, IMAGE_SIZE, IMAGE_SIZE);
        let tile = Tile {
            y_quant_index: 0,
            cb_quant_index: 0,
            cr_quant_index: 0,
            x: tile_x,
            y: tile_y,
            y_data: &[],
            cb_data: &[],
            cr_data: &[],
        };
        let update_rectangle = tiles_to_rectangles(&[tile], &destination)
            .next()
            .expect("one tile yields one rectangle");

        let mut image = DecodedImage::new(PixelFormat::RgbA32, IMAGE_SIZE, IMAGE_SIZE);
        let tile_output = vec![0xAB; usize::from(TILE_SIZE) * usize::from(TILE_SIZE) * 4];
        image
            .apply_tile(&tile_output, PixelFormat::RgbA32, &clipping, &update_rectangle)
            .expect("tile applies");
        image
    }

    fn is_painted(image: &DecodedImage, x: usize, y: usize) -> bool {
        image.data()[(y * usize::from(image.width()) + x) * 4] != 0
    }

    #[test]
    fn region_keeps_one_row_rect_adjacent_to_a_band() {
        let image = apply_tile_in_region(
            &[
                rfx_rect(32, 40, 32, 18),
                rfx_rect(0, 58, 16, 1),
                rfx_rect(32, 58, 32, 1),
            ],
            0,
            0,
        );

        let painted_rows: Vec<usize> = (0..usize::from(IMAGE_SIZE))
            .filter(|&y| is_painted(&image, 32, y) && is_painted(&image, 63, y))
            .collect();
        assert_eq!(painted_rows, (40..59).collect::<Vec<usize>>());
        assert!(is_painted(&image, 0, 58));
        assert!(is_painted(&image, 15, 58));
        assert!(!is_painted(&image, 16, 58));
        assert!(!is_painted(&image, 31, 58));
        assert!(!is_painted(&image, 32, 59));
        assert!(!is_painted(&image, 64, 58));
    }

    #[test]
    fn region_clips_rect_extending_past_the_surface() {
        let image = apply_tile_in_region(&[rfx_rect(120, 120, 64, 64)], 1, 1);

        assert!(is_painted(&image, 120, 120));
        assert!(is_painted(&image, 127, 127));
        assert!(!is_painted(&image, 119, 120));
        assert!(!is_painted(&image, 120, 119));
    }

    fn any_painted(image: &DecodedImage) -> bool {
        image.data().iter().any(|&b| b != 0)
    }

    #[test]
    fn apply_tile_skips_degenerate_intersections() {
        // The tile at (0, 1) touches the first and third rectangles only along an edge; their
        // intersection with the tile is a zero-sized rectangle that must be skipped.
        let image = apply_tile_in_region(
            &[
                rfx_rect(0, 0, 32, 64),
                rfx_rect(0, 64, 64, 10),
                rfx_rect(64, 0, 64, 128),
            ],
            0,
            1,
        );

        assert!(!is_painted(&image, 0, 63));
        for y in 64..=73 {
            assert!(is_painted(&image, 0, y), "row {y} at x=0");
            assert!(is_painted(&image, 63, y), "row {y} at x=63");
        }
        assert!(!is_painted(&image, 0, 74));
        assert!(!is_painted(&image, 63, 74));
    }

    #[test]
    fn region_outside_the_surface_paints_nothing_and_reports_empty() {
        let image_size = IMAGE_SIZE;
        let destination = InclusiveRectangle {
            left: 0,
            top: 0,
            right: image_size - 1,
            bottom: image_size - 1,
        };
        let clipping = clipping_rectangles(&[rfx_rect(130, 0, 8, 8)], &destination, image_size, image_size);
        let tile = Tile {
            y_quant_index: 0,
            cb_quant_index: 0,
            cr_quant_index: 0,
            x: 1,
            y: 0,
            y_data: &[],
            cb_data: &[],
            cr_data: &[],
        };
        let update_rectangle = tiles_to_rectangles(&[tile], &destination)
            .next()
            .expect("one tile yields one rectangle");
        let mut image = DecodedImage::new(PixelFormat::RgbA32, image_size, image_size);
        let tile_output = vec![0xAB; usize::from(TILE_SIZE) * usize::from(TILE_SIZE) * 4];

        let reported = image
            .apply_tile(&tile_output, PixelFormat::RgbA32, &clipping, &update_rectangle)
            .expect("tile applies");

        assert!(!any_painted(&image));
        assert_eq!(reported, InclusiveRectangle::empty());
    }

    fn decode_empty_tile_frame(rectangles: Vec<RfxRectangle>) -> InclusiveRectangle {
        let blocks = [
            rfx::Block::CodecChannel(rfx::CodecChannel::Region(rfx::RegionPdu { rectangles })),
            rfx::Block::CodecChannel(rfx::CodecChannel::TileSet(rfx::TileSetPdu {
                entropy_algorithm: EntropyAlgorithm::Rlgr1,
                quants: Vec::new(),
                tiles: Vec::new(),
            })),
            rfx::Block::CodecChannel(rfx::CodecChannel::FrameEnd(rfx::FrameEndPdu)),
        ];
        let mut bytes = Vec::new();
        for block in &blocks {
            bytes.extend(ironrdp_pdu::encode_vec(block).expect("block encodes"));
        }

        let mut context = DecodingContext::new();
        context.channels = rfx::ChannelsPdu(vec![rfx::RfxChannel {
            width: i16::try_from(IMAGE_SIZE).expect("fits"),
            height: i16::try_from(IMAGE_SIZE).expect("fits"),
        }]);
        let mut image = DecodedImage::new(PixelFormat::RgbA32, IMAGE_SIZE, IMAGE_SIZE);
        let destination = InclusiveRectangle {
            left: 0,
            top: 0,
            right: IMAGE_SIZE - 1,
            bottom: IMAGE_SIZE - 1,
        };
        let frame_begin = rfx::FrameBeginPdu {
            index: 7,
            number_of_regions: 1,
        };

        let (frame_id, reported) = context
            .process_frame(frame_begin, &mut ReadCursor::new(&bytes), &mut image, &destination)
            .expect("frame decodes");
        assert_eq!(frame_id, 7);
        reported
    }

    #[test]
    fn frame_reports_inclusive_extents_at_the_surface_edge() {
        let reported = decode_empty_tile_frame(vec![rfx_rect(100, 90, 28, 38)]);

        assert_eq!(
            reported,
            InclusiveRectangle {
                left: 100,
                top: 90,
                right: IMAGE_SIZE - 1,
                bottom: IMAGE_SIZE - 1,
            }
        );
    }

    #[test]
    fn frame_with_region_outside_the_surface_reports_empty() {
        assert_eq!(
            decode_empty_tile_frame(vec![rfx_rect(130, 0, 8, 8)]),
            InclusiveRectangle::empty()
        );
    }

    fn tile_at(x: u16, y: u16) -> Tile<'static> {
        Tile {
            y_quant_index: 0,
            cb_quant_index: 0,
            cr_quant_index: 0,
            x,
            y,
            y_data: &[],
            cb_data: &[],
            cr_data: &[],
        }
    }

    fn destination_at(left: u16, top: u16) -> InclusiveRectangle {
        InclusiveRectangle {
            left,
            top,
            right: IMAGE_SIZE - 1,
            bottom: IMAGE_SIZE - 1,
        }
    }

    fn first_tile_rectangle(tile: Tile<'static>, destination: &InclusiveRectangle) -> InclusiveRectangle {
        tiles_to_rectangles(&[tile], destination)
            .next()
            .expect("one tile yields one rectangle")
    }

    #[test]
    fn tiles_to_rectangles_saturates_server_controlled_tile_index() {
        let rectangle = first_tile_rectangle(tile_at(1023, 1023), &destination_at(0, 0));

        assert_eq!(rectangle.left, 1023 * TILE_SIZE);
        assert_eq!(rectangle.right, u16::MAX);
        assert_eq!(rectangle.bottom, u16::MAX);
    }

    #[test]
    fn tiles_to_rectangles_saturates_tile_index_multiplication() {
        for index in [1024, u16::MAX] {
            let rectangle = first_tile_rectangle(tile_at(index, index), &destination_at(0, 0));

            assert_eq!(rectangle.left, u16::MAX, "tile index {index}");
            assert_eq!(rectangle.top, u16::MAX, "tile index {index}");
            assert_eq!(rectangle.right, u16::MAX, "tile index {index}");
            assert_eq!(rectangle.bottom, u16::MAX, "tile index {index}");
        }
    }

    #[test]
    fn tiles_to_rectangles_saturates_destination_offset() {
        let rectangle = first_tile_rectangle(tile_at(1, 0), &destination_at(65500, 0));
        assert_eq!(rectangle.left, u16::MAX);
        assert_eq!(rectangle.top, 0);

        let rectangle = first_tile_rectangle(tile_at(0, 1), &destination_at(0, 65500));
        assert_eq!(rectangle.left, 0);
        assert_eq!(rectangle.top, u16::MAX);
    }

    #[test]
    fn clipping_rectangles_clamps_overflowing_rectangle_instead_of_dropping_it() {
        let destination = InclusiveRectangle {
            left: 100,
            top: 100,
            right: IMAGE_SIZE - 1,
            bottom: IMAGE_SIZE - 1,
        };

        let region = clipping_rectangles(
            &[rfx_rect(0, 0, u16::MAX, u16::MAX)],
            &destination,
            IMAGE_SIZE,
            IMAGE_SIZE,
        );

        assert_eq!(region.extents.left, 100);
        assert_eq!(region.extents.top, 100);
        assert_eq!(region.extents.right, IMAGE_SIZE);
        assert_eq!(region.extents.bottom, IMAGE_SIZE);
    }

    #[test]
    fn clipping_rectangles_drops_rectangle_whose_offset_overflows() {
        let region = clipping_rectangles(
            &[rfx_rect(100, 0, 64, 64)],
            &destination_at(65500, 0),
            IMAGE_SIZE,
            IMAGE_SIZE,
        );
        assert_eq!(region.extents, InclusiveRectangle::empty());

        let region = clipping_rectangles(
            &[rfx_rect(0, 100, 64, 64)],
            &destination_at(0, 65500),
            IMAGE_SIZE,
            IMAGE_SIZE,
        );
        assert_eq!(region.extents, InclusiveRectangle::empty());
    }
}
