//! Regression tests for `composite_graphics_updates`, the accumulator that turns a
//! drain of EGFX compositor deltas into the single region `ActiveStage::process`
//! reports to its caller, and for `apply_reset_graphics`, which resizes the session's
//! output image to the dimensions a `ResetGraphics` PDU declared.
//!
//! `ironrdp-session` builds with `[lib] test = false`, so inline `#[cfg(test)]`
//! modules there never run under `cargo test --workspace --locked`. These tests
//! live here instead so they actually execute in CI.

use std::any::TypeId;

use ironrdp_core::encode_vec;
use ironrdp_dvc::DrdynvcClient;
use ironrdp_dvc::pdu::{CreateRequestPdu, DataPdu, DrdynvcDataPdu, DrdynvcServerPdu};
use ironrdp_egfx::CHANNEL_NAME;
use ironrdp_egfx::client::{BitmapUpdate, GraphicsPipelineClient, GraphicsPipelineHandler, Surface};
use ironrdp_egfx::pdu::{
    CapabilitySet, Color, CreateSurfacePdu, EndFramePdu, GfxPdu, MapSurfaceToOutputPdu, PixelFormat as GfxPixelFormat,
    ResetGraphicsPdu, SolidFillPdu, StartFramePdu, Timestamp,
};
use ironrdp_graphics::image_processing::PixelFormat;
use ironrdp_graphics::zgfx::wrap_uncompressed;
use ironrdp_pdu::Action;
use ironrdp_pdu::geometry::ExclusiveRectangle;
use ironrdp_pdu::mcs::SendDataIndication;
use ironrdp_pdu::rdp::vc::{ChannelControlFlags, ChannelPduHeader};
use ironrdp_pdu::x224::X224;
use ironrdp_session::image::DecodedImage;
use ironrdp_session::{
    ActiveStageBuilder, ActiveStageOutput, MAX_GRAPHICS_OUTPUT_DIMENSION, apply_reset_graphics,
    composite_graphics_updates,
};
use ironrdp_svc::StaticChannelSet;

fn update(left: u16, top: u16, right: u16, bottom: u16) -> (ExclusiveRectangle, Vec<u8>) {
    let w = usize::from(right - left);
    let h = usize::from(bottom - top);
    (
        ExclusiveRectangle {
            left,
            top,
            right,
            bottom,
        },
        vec![0xFF; w * h * 4],
    )
}

/// Two disjoint deltas collapse to the rectangle spanning both, so a consumer that
/// redraws the named region does one copy instead of one per delta.
#[test]
fn disjoint_deltas_collapse_to_their_union() {
    let mut image = DecodedImage::new(PixelFormat::RgbA32, 200, 200);

    let region = composite_graphics_updates(&mut image, [update(10, 10, 20, 20), update(100, 100, 150, 150)])
        .expect("both deltas are inside the image")
        .expect("two deltas produce a region");

    // Exclusive right/bottom of 20 and 150 become inclusive 19 and 149.
    assert_eq!(region.left, 10);
    assert_eq!(region.top, 10);
    assert_eq!(region.right, 149);
    assert_eq!(region.bottom, 149);
}

/// The count is what matters: any number of deltas yields exactly one region, which
/// is the property that keeps `ironrdp-client` from rebuilding the framebuffer once
/// per rectangle.
#[test]
fn many_deltas_yield_one_region() {
    let mut image = DecodedImage::new(PixelFormat::RgbA32, 512, 512);
    let updates: Vec<_> = (0..64).map(|i| update(i, i, i + 8, i + 8)).collect();

    let region = composite_graphics_updates(&mut image, updates)
        .expect("all deltas are inside the image")
        .expect("64 deltas produce a region");

    assert_eq!(region.left, 0);
    assert_eq!(region.top, 0);
    assert_eq!(region.right, 70);
    assert_eq!(region.bottom, 70);
}

/// A drain that produced nothing must not surface an update at all, so a non-EGFX
/// session sees no change in behavior.
#[test]
fn no_deltas_yield_no_region() {
    let mut image = DecodedImage::new(PixelFormat::RgbA32, 64, 64);
    assert!(
        composite_graphics_updates(&mut image, [])
            .expect("an empty drain cannot fail")
            .is_none()
    );
}

/// One delta passes through as itself rather than being widened by the accumulator.
#[test]
fn a_single_delta_is_its_own_region() {
    let mut image = DecodedImage::new(PixelFormat::RgbA32, 64, 64);

    let region = composite_graphics_updates(&mut image, [update(4, 8, 12, 16)])
        .expect("the delta is inside the image")
        .expect("one delta produces a region");

    assert_eq!(region.left, 4);
    assert_eq!(region.top, 8);
    assert_eq!(region.right, 11);
    assert_eq!(region.bottom, 15);
}

/// A delta outside the image bounds (the compositor's output can be larger than
/// `image`, since `image` is sized from the desktop and never resized on
/// ResetGraphics) must not be folded into the accumulator at all. Before this fix,
/// `apply_rgba32`'s `InclusiveRectangle::empty()` rejection sentinel, which is
/// `(0, 0, 0, 0)`, got unioned in like a real update, corrupting the reported
/// region to include the unpainted origin.
#[test]
fn an_out_of_bounds_delta_is_dropped_not_unioned() {
    let mut image = DecodedImage::new(PixelFormat::RgbA32, 64, 64);

    let region = composite_graphics_updates(&mut image, [update(20, 20, 30, 30), update(100, 100, 149, 149)])
        .expect("the in-bounds delta succeeds")
        .expect("the in-bounds delta produces a region");

    assert_eq!(
        (region.left, region.top, region.right, region.bottom),
        (20, 20, 29, 29),
        "the out-of-bounds delta must not widen the region to include the origin"
    );
}

/// When every delta is out of bounds, the drain must report no region at all, not a
/// phantom 1x1 rectangle at the origin. Before this fix, `dirty` ended up
/// `Some((0, 0, 0, 0))` in this case, contradicting the invariant
/// `no_deltas_yield_no_region` asserts for an empty drain.
#[test]
fn every_delta_out_of_bounds_yields_no_region() {
    let mut image = DecodedImage::new(PixelFormat::RgbA32, 64, 64);

    let region = composite_graphics_updates(&mut image, [update(100, 100, 149, 149)])
        .expect("an out-of-bounds delta does not error");

    assert!(
        region.is_none(),
        "a frame where nothing was painted must not report a region, got {region:?}"
    );
}

/// The bounds check is `>=`, not `>`: a delta whose exclusive right/bottom equals the
/// image width/height is exactly at the edge the exclusive-to-inclusive conversion
/// turns on (an exclusive bound of `width` becomes inclusive `width - 1`, which fits).
/// This pins that edge is accepted, not dropped as if it were one pixel out of bounds.
#[test]
fn a_delta_touching_the_image_edge_is_accepted() {
    let mut image = DecodedImage::new(PixelFormat::RgbA32, 64, 64);

    let region = composite_graphics_updates(&mut image, [update(60, 60, 64, 64)])
        .expect("a delta flush with the image edge is inside the image")
        .expect("the delta produces a region");

    assert_eq!(
        (region.left, region.top, region.right, region.bottom),
        (60, 60, 63, 63),
        "an exclusive bound equal to the image dimension must convert to the last valid pixel, not be dropped"
    );
}

#[test]
fn apply_reset_graphics_resizes_the_image_to_the_declared_dimensions() {
    let mut image = DecodedImage::new(PixelFormat::RgbA32, 1, 1);

    apply_reset_graphics(&mut image, 800, 600, MAX_GRAPHICS_OUTPUT_DIMENSION).unwrap();

    assert_eq!(image.width(), 800);
    assert_eq!(image.height(), 600);
    assert_eq!(image.pixel_format(), PixelFormat::RgbA32);
    assert!(image.data().iter().all(|&byte| byte == 0));
}

#[test]
fn apply_reset_graphics_rejects_zero_dimensions() {
    let mut image = DecodedImage::new(PixelFormat::RgbA32, 1, 1);

    assert!(apply_reset_graphics(&mut image, 0, 600, MAX_GRAPHICS_OUTPUT_DIMENSION).is_err());
    assert!(apply_reset_graphics(&mut image, 800, 0, MAX_GRAPHICS_OUTPUT_DIMENSION).is_err());
}

#[test]
fn apply_reset_graphics_rejects_dimensions_past_the_spec_maximum() {
    let mut image = DecodedImage::new(PixelFormat::RgbA32, 1, 1);

    assert!(apply_reset_graphics(&mut image, 32767, 600, MAX_GRAPHICS_OUTPUT_DIMENSION).is_err());
    assert!(apply_reset_graphics(&mut image, 800, 32767, MAX_GRAPHICS_OUTPUT_DIMENSION).is_err());
    assert!(apply_reset_graphics(&mut image, u32::MAX, 600, MAX_GRAPHICS_OUTPUT_DIMENSION).is_err());
    assert!(apply_reset_graphics(&mut image, 800, u32::MAX, MAX_GRAPHICS_OUTPUT_DIMENSION).is_err());
}

#[test]
fn apply_reset_graphics_accepts_the_exact_spec_maximum() {
    // 32766 on both axes is ~4.1 GiB of RGBA8888 and too heavy to allocate in a unit
    // test, so this pins the boundary on one axis at a time instead, with the other
    // held small: it exercises the same `> max` comparison a careless `>=` refactor
    // would break, without the large allocation.
    let mut image = DecodedImage::new(PixelFormat::RgbA32, 1, 1);
    apply_reset_graphics(&mut image, 32766, 1, MAX_GRAPHICS_OUTPUT_DIMENSION).unwrap();
    assert_eq!(image.width(), 32766);
    assert_eq!(image.height(), 1);

    let mut image = DecodedImage::new(PixelFormat::RgbA32, 1, 1);
    apply_reset_graphics(&mut image, 1, 32766, MAX_GRAPHICS_OUTPUT_DIMENSION).unwrap();
    assert_eq!(image.width(), 1);
    assert_eq!(image.height(), 32766);
}

/// A client with a tighter framebuffer limit than the spec maximum (e.g. wgpu's
/// `max_texture_dimension_2d` under `Limits::default()`) rejects a `ResetGraphics`
/// past its own limit instead of allocating an image it cannot back with a texture.
#[test]
fn apply_reset_graphics_rejects_dimensions_past_a_client_supplied_maximum() {
    let mut image = DecodedImage::new(PixelFormat::RgbA32, 1, 1);

    assert!(apply_reset_graphics(&mut image, 8193, 600, 8192).is_err());
    assert!(apply_reset_graphics(&mut image, 800, 8193, 8192).is_err());
}

#[test]
fn apply_reset_graphics_accepts_the_exact_client_supplied_maximum() {
    let mut image = DecodedImage::new(PixelFormat::RgbA32, 1, 1);
    apply_reset_graphics(&mut image, 8192, 600, 8192).unwrap();
    assert_eq!(image.width(), 8192);
    assert_eq!(image.height(), 600);

    let mut image = DecodedImage::new(PixelFormat::RgbA32, 1, 1);
    apply_reset_graphics(&mut image, 800, 8192, 8192).unwrap();
    assert_eq!(image.width(), 800);
    assert_eq!(image.height(), 8192);
}

/// A `ResetGraphics` declaring the image's current dimensions must leave the pixels
/// untouched: the server can send `ResetGraphics` to rebuild its own surfaces without
/// the client's framebuffer changing size, and rebuilding the image in that case would
/// paint a visible blank flash and drop the software-pointer state for no reason.
#[test]
fn apply_reset_graphics_preserves_pixels_on_a_same_size_reset() {
    let mut image = DecodedImage::new(PixelFormat::RgbA32, 4, 4);
    composite_graphics_updates(&mut image, [update(0, 0, 4, 4)])
        .expect("the update covers the whole image")
        .expect("a full-image update produces a region");
    let painted = image.data().to_vec();
    assert!(
        painted.iter().any(|&byte| byte != 0),
        "the update must have painted something"
    );

    apply_reset_graphics(&mut image, 4, 4, MAX_GRAPHICS_OUTPUT_DIMENSION).unwrap();

    assert_eq!(image.width(), 4);
    assert_eq!(image.height(), 4);
    assert_eq!(
        image.data(),
        painted.as_slice(),
        "a same-size reset must not touch the pixels"
    );
}

/// Bounds validation runs before the same-size guard: a reset whose declared size
/// matches the image but is itself out of range must still be rejected.
#[test]
fn apply_reset_graphics_still_validates_bounds_on_a_same_size_reset() {
    let mut image = DecodedImage::new(PixelFormat::RgbA32, 0, 0);
    // width/height of 0 match the image's current (default) dimensions, but 0 is
    // rejected by the range check regardless.
    assert!(apply_reset_graphics(&mut image, 0, 0, MAX_GRAPHICS_OUTPUT_DIMENSION).is_err());
}

struct NoopGfxHandler;

impl GraphicsPipelineHandler for NoopGfxHandler {
    fn on_capabilities_confirmed(&mut self, _caps: &CapabilitySet) {}
    fn on_reset_graphics(&mut self, _width: u32, _height: u32) {}
    fn on_surface_created(&mut self, _surface: &Surface) {}
    fn on_surface_deleted(&mut self, _surface_id: u16) {}
    fn on_surface_mapped(&mut self, _surface_id: u16, _x: u32, _y: u32) {}
    fn on_bitmap_updated(&mut self, _update: &BitmapUpdate) {}
    fn on_frame_complete(&mut self, _frame_id: u32) {}
    fn on_close(&mut self) {}
    fn on_unhandled_pdu(&mut self, _pdu: &GfxPdu) {}
}

const USER_CHANNEL_ID: u16 = 1002;
const IO_CHANNEL_ID: u16 = 1003;
const DRDYNVC_CHANNEL_ID: u16 = 1004;
const GFX_DVC_ID: u32 = 3;

/// Wraps a drdynvc-encoded payload as a Virtual Channel PDU (MS-RDPBCGR 2.2.6.1) — a
/// `ChannelPduHeader` naming the total length, followed by the data, with
/// FLAG_FIRST | FLAG_LAST since this test never spans multiple chunks — then as an MCS
/// Send Data Indication addressing the drdynvc static channel, the way a server frame
/// addresses it.
fn encode_drdynvc_packet(dvc_data: Vec<u8>) -> Vec<u8> {
    let mut channel_pdu = encode_vec(&ChannelPduHeader {
        length: u32::try_from(dvc_data.len()).expect("length fits in u32"),
        flags: ChannelControlFlags::FLAG_FIRST | ChannelControlFlags::FLAG_LAST,
    })
    .expect("encode channel PDU header");
    channel_pdu.extend_from_slice(&dvc_data);

    let indication = SendDataIndication {
        initiator_id: USER_CHANNEL_ID,
        channel_id: DRDYNVC_CHANNEL_ID,
        user_data: channel_pdu.into(),
    };

    encode_vec(&X224(indication)).expect("encode Send Data Indication")
}

/// Frames the DVC Create Request that opens the EGFX channel, the way a server does
/// before sending any EGFX PDU.
fn encode_egfx_create_packet() -> Vec<u8> {
    let dvc_data = encode_vec(&DrdynvcServerPdu::Create(CreateRequestPdu::new(
        GFX_DVC_ID,
        CHANNEL_NAME.to_owned(),
    )))
    .expect("encode drdynvc create PDU");

    encode_drdynvc_packet(dvc_data)
}

/// Frames a sequence of EGFX PDUs as a single drdynvc data message on the (already
/// created) EGFX dynamic channel.
fn encode_egfx_packet(pdus: &[GfxPdu]) -> Vec<u8> {
    let mut raw = Vec::new();
    for pdu in pdus {
        raw.extend(encode_vec(pdu).expect("encode GFX PDU"));
    }
    let zgfx_payload = wrap_uncompressed(&raw);

    let dvc_data = encode_vec(&DrdynvcServerPdu::Data(DrdynvcDataPdu::Data(DataPdu::new(
        GFX_DVC_ID,
        zgfx_payload,
    ))))
    .expect("encode drdynvc data PDU");

    encode_drdynvc_packet(dvc_data)
}

/// This is the property `one_packet_yields_both_the_reset_size_and_a_full_surface_delta`
/// (in `ironrdp-egfx`) cannot pin: that test reads `take_reset_graphics()` and
/// `drain_output()` as two independent fields of `GraphicsPipelineClient`, so swapping
/// the order `ActiveStage::process` consumes them in would not fail it. The actual
/// invariant lives in `ActiveStage::process`: it must apply a drained `ResetGraphics`
/// to `image` *before* compositing the drained graphics deltas, because a delta against
/// the new (larger) surface is out of bounds for the still-old-sized `image` and gets
/// silently dropped by `composite_graphics_updates`'s bounds check. This test drives a
/// real `ActiveStage` end to end through a real `GraphicsPipelineClient` DVC to pin that
/// ordering at the only level that can observe it.
#[test]
fn a_reset_and_its_repaint_in_one_packet_resize_the_image_before_compositing() {
    const OLD_SIZE: u16 = 200;
    const NEW_SIZE: u16 = 400;
    const SURFACE_ID: u16 = 1;

    let drdynvc =
        DrdynvcClient::new().with_dynamic_channel(GraphicsPipelineClient::new(Box::new(NoopGfxHandler), None));

    let mut static_channels = StaticChannelSet::new();
    static_channels.insert(drdynvc);
    static_channels.attach_channel_id(TypeId::of::<DrdynvcClient>(), DRDYNVC_CHANNEL_ID);

    let mut active = ActiveStageBuilder {
        static_channels,
        user_channel_id: USER_CHANNEL_ID,
        io_channel_id: IO_CHANNEL_ID,
        message_channel_id: None,
        share_id: 0,
        compression_type: None,
        enable_server_pointer: false,
        pointer_software_rendering: false,
    }
    .build();

    let mut image = DecodedImage::new(PixelFormat::RgbA32, OLD_SIZE, OLD_SIZE);

    // Open the EGFX dynamic channel first, as a server does before sending any EGFX PDU.
    active
        .process(&mut image, Action::X224, &encode_egfx_create_packet())
        .expect("the EGFX channel should open");

    // A single server packet, as sent after a display resize: ResetGraphics to the new
    // (larger) size, followed by a surface sized to match and a full-surface repaint —
    // all decoded from one `process()` call.
    let frame = encode_egfx_packet(&[
        GfxPdu::ResetGraphics(ResetGraphicsPdu {
            width: u32::from(NEW_SIZE),
            height: u32::from(NEW_SIZE),
            monitors: vec![],
        }),
        GfxPdu::CreateSurface(CreateSurfacePdu {
            surface_id: SURFACE_ID,
            width: NEW_SIZE,
            height: NEW_SIZE,
            pixel_format: GfxPixelFormat::XRgb,
        }),
        GfxPdu::MapSurfaceToOutput(MapSurfaceToOutputPdu {
            surface_id: SURFACE_ID,
            output_origin_x: 0,
            output_origin_y: 0,
        }),
        GfxPdu::StartFrame(StartFramePdu {
            timestamp: Timestamp {
                milliseconds: 0,
                seconds: 0,
                minutes: 0,
                hours: 0,
            },
            frame_id: 0,
        }),
        GfxPdu::SolidFill(SolidFillPdu {
            surface_id: SURFACE_ID,
            fill_pixel: Color {
                b: 0,
                g: 0,
                r: 0,
                xa: 0xFF,
            },
            rectangles: vec![ExclusiveRectangle {
                left: 0,
                top: 0,
                right: NEW_SIZE,
                bottom: NEW_SIZE,
            }],
        }),
        GfxPdu::EndFrame(EndFramePdu { frame_id: 0 }),
    ]);

    let outputs = active
        .process(&mut image, Action::X224, &frame)
        .expect("processing the combined ResetGraphics + repaint packet should succeed");

    assert_eq!(
        image.width(),
        NEW_SIZE,
        "the image must be resized to the ResetGraphics dimensions"
    );
    assert_eq!(
        image.height(),
        NEW_SIZE,
        "the image must be resized to the ResetGraphics dimensions"
    );

    let region = outputs
        .iter()
        .find_map(|output| match output {
            ActiveStageOutput::GraphicsUpdate(region) => Some(region),
            _ => None,
        })
        .expect(
            "the full-surface repaint must reach the caller as a graphics update; if the reset were applied \
             after compositing, the repaint would be out of bounds for the still-old-sized image and \
             `composite_graphics_updates` would silently drop it, producing no `GraphicsUpdate` at all",
        );

    // Exclusive right/bottom of NEW_SIZE become inclusive NEW_SIZE - 1: this is the whole
    // resized surface, not clipped to the old, smaller image.
    assert_eq!(
        (region.left, region.top, region.right, region.bottom),
        (0, 0, NEW_SIZE - 1, NEW_SIZE - 1),
        "the repaint region must cover the entire resized surface"
    );
}
