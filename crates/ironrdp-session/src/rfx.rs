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
        let clipped = InclusiveRectangle {
            left: min(destination.left + r.x, width),
            top: min(destination.top + r.y, height),
            right: min(destination.left + r.x + r.width, width),
            bottom: min(destination.top + r.y + r.height, height),
        };
        if clipped.left < clipped.right && clipped.top < clipped.bottom {
            clipping_rectangles.union_rectangle(clipped);
        }
    }

    clipping_rectangles
}

/// Tile rectangles with exclusive right/bottom bounds (`x + 64`), matching the clipping region.
fn tiles_to_rectangles<'a>(
    tiles: &'a [Tile<'_>],
    destination: &'a InclusiveRectangle,
) -> impl Iterator<Item = InclusiveRectangle> + 'a {
    tiles.iter().map(|t| InclusiveRectangle {
        left: destination.left + t.x * TILE_SIZE,
        top: destination.top + t.y * TILE_SIZE,
        right: destination.left + t.x * TILE_SIZE + TILE_SIZE,
        bottom: destination.top + t.y * TILE_SIZE + TILE_SIZE,
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
}
