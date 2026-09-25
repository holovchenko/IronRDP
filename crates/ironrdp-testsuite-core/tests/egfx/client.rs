use ironrdp_core::{Decode as _, Encode, ReadCursor, WriteCursor, encode_vec};
use ironrdp_dvc::DvcProcessor as _;
use ironrdp_egfx::client::{
    BitmapUpdate, CacheImportOutcome, CacheImportTile, CodecProcessed, DecodedCodec, GraphicsPipelineClient,
    GraphicsPipelineHandler, Surface,
};
use ironrdp_egfx::decode::{DecodedFrame, DecoderResult, H264Decoder};
use ironrdp_egfx::pdu::{
    CacheImportReplyPdu, CacheToSurfacePdu, CapabilitiesAdvertisePdu, CapabilitiesConfirmPdu, CapabilitiesV8Flags,
    CapabilitySet, CapabilityVersion, Codec1Type, Codec2Type, Color, CreateSurfacePdu, DeleteSurfacePdu, EndFramePdu,
    FrameAcknowledgePdu, GfxPdu, MapSurfaceToOutputPdu, PixelFormat, Point, ResetGraphicsPdu, SolidFillPdu,
    StartFramePdu, SurfaceToCachePdu, Timestamp, WireToSurface1Pdu, WireToSurface2Pdu,
};
use ironrdp_graphics::clearcodec::ClearCodecEncoder;
use ironrdp_graphics::zgfx::wrap_uncompressed;
use ironrdp_pdu::geometry::ExclusiveRectangle;

// ============================================================================
// Test Handler
// ============================================================================

struct TestHandler {
    caps_confirmed: bool,
    bitmaps_received: Vec<(u16, Codec1Type)>,
    frames_completed: Vec<u32>,
    reset_count: u32,
}

impl TestHandler {
    fn new() -> Self {
        Self {
            caps_confirmed: false,
            bitmaps_received: Vec::new(),
            frames_completed: Vec::new(),
            reset_count: 0,
        }
    }
}

impl GraphicsPipelineHandler for TestHandler {
    fn on_capabilities_confirmed(&mut self, _caps: &CapabilitySet) {
        self.caps_confirmed = true;
    }

    fn on_reset_graphics(&mut self, _width: u32, _height: u32) {
        self.reset_count += 1;
    }

    fn on_surface_created(&mut self, _surface: &Surface) {}
    fn on_surface_deleted(&mut self, _surface_id: u16) {}
    fn on_surface_mapped(&mut self, _surface_id: u16, _x: u32, _y: u32) {}

    fn on_bitmap_updated(&mut self, update: &BitmapUpdate) {
        self.bitmaps_received.push((update.surface_id, update.codec_id));
    }

    fn on_frame_complete(&mut self, frame_id: u32) {
        self.frames_completed.push(frame_id);
    }

    fn on_close(&mut self) {}
    fn on_unhandled_pdu(&mut self, _pdu: &GfxPdu) {}
}

// ============================================================================
// Mock H.264 Decoder
// ============================================================================

struct MockH264Decoder;

impl H264Decoder for MockH264Decoder {
    fn decode(&mut self, _data: &[u8]) -> DecoderResult<DecodedFrame> {
        // Return a 16x16 solid red frame (macroblock-aligned minimum)
        let mut data = vec![0u8; 16 * 16 * 4];
        for pixel in data.chunks_exact_mut(4) {
            pixel[0] = 255; // R
            pixel[3] = 255; // A
        }
        Ok(DecodedFrame::new(data, 16, 16))
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn encode_pdu<T: Encode>(pdu: &T) -> Vec<u8> {
    let mut buf = vec![0u8; pdu.size()];
    let mut cursor = WriteCursor::new(&mut buf);
    pdu.encode(&mut cursor).expect("encode failed");
    buf
}

/// Encode a GfxPdu and wrap in a ZGFX uncompressed segment descriptor.
/// The client's process() expects ZGFX-segmented input (it runs decompression first).
fn encode_for_process(pdu: &GfxPdu) -> Vec<u8> {
    let raw = encode_pdu(pdu);
    wrap_uncompressed(&raw)
}

fn decode_caps_from_message(msg: &ironrdp_dvc::DvcMessage) -> CapabilitiesAdvertisePdu {
    let encoded = encode_vec(msg.as_ref()).expect("encode should succeed");
    let mut cursor = ReadCursor::new(&encoded);
    let pdu = GfxPdu::decode(&mut cursor).expect("decode should succeed");
    match pdu {
        GfxPdu::CapabilitiesAdvertise(caps) => caps,
        other => panic!("expected CapabilitiesAdvertise, got {other:?}"),
    }
}

/// Create a client, send CapabilitiesConfirm V8 through process(), and create a surface.
fn setup_active_client_with_surface(
    decoder: Option<Box<dyn H264Decoder>>,
    surface_id: u16,
    width: u16,
    height: u16,
) -> GraphicsPipelineClient {
    let handler = TestHandler::new();
    let mut client = GraphicsPipelineClient::new(Box::new(handler), decoder);

    // Activate via CapabilitiesConfirm
    let confirm = GfxPdu::CapabilitiesConfirm(CapabilitiesConfirmPdu::from_typed(&CapabilitySet::V8 {
        flags: CapabilitiesV8Flags::empty(),
    }));
    client
        .process(0, &encode_for_process(&confirm))
        .expect("confirm should succeed");

    // Create surface
    let create = GfxPdu::CreateSurface(CreateSurfacePdu {
        surface_id,
        width,
        height,
        pixel_format: PixelFormat::XRgb,
    });
    client
        .process(0, &encode_for_process(&create))
        .expect("create surface should succeed");

    client
}

// ============================================================================
// Tests: Capability Advertisement
// ============================================================================

#[test]
fn client_sends_capabilities_on_start() {
    let handler = TestHandler::new();
    let mut client = GraphicsPipelineClient::new(Box::new(handler), None);
    let messages = client.start(0).expect("start should succeed");
    assert_eq!(messages.len(), 1);
}

#[test]
fn client_filters_avc_caps_without_decoder() {
    let handler = TestHandler::new();
    let mut client = GraphicsPipelineClient::new(Box::new(handler), None);
    let messages = client.start(0).expect("start should succeed");
    assert_eq!(messages.len(), 1);

    let caps_pdu = decode_caps_from_message(&messages[0]);
    assert_eq!(
        caps_pdu.0.len(),
        1,
        "expected exactly one capability set when no decoder is present"
    );
    assert!(
        caps_pdu.0[0].version == CapabilityVersion::V8,
        "expected only V8 capability set without decoder, got {:?}",
        caps_pdu.0[0]
    );
}

#[test]
fn client_keeps_avc_caps_with_decoder() {
    let handler = TestHandler::new();
    let mut client = GraphicsPipelineClient::new(Box::new(handler), Some(Box::new(MockH264Decoder)));
    let messages = client.start(0).expect("start should succeed");
    assert_eq!(messages.len(), 1);

    let caps_pdu = decode_caps_from_message(&messages[0]);
    assert_eq!(
        caps_pdu.0.len(),
        2,
        "expected both capability sets with decoder present"
    );
    // V10.x is absent by design: those versions imply AVC444, which the client
    // cannot decode, and the server would then send only frames it discards.
    assert_eq!(caps_pdu.0[0].version, CapabilityVersion::V8_1);
    assert_eq!(caps_pdu.0[1].version, CapabilityVersion::V8);
}

// ============================================================================
// Tests: Frame Flow (via process() with encoded PDUs)
// ============================================================================

#[test]
fn client_sends_frame_ack_on_end_frame() {
    let mut client = setup_active_client_with_surface(None, 1, 4, 4);

    let end = GfxPdu::EndFrame(EndFramePdu { frame_id: 42 });
    let responses = client
        .process(0, &encode_for_process(&end))
        .expect("end frame should succeed");

    assert_eq!(responses.len(), 1, "should produce exactly one FrameAcknowledge");
    assert_eq!(client.total_frames_decoded(), 1);
}

#[test]
fn client_handles_uncompressed_via_process() {
    let mut client = setup_active_client_with_surface(None, 1, 4, 4);

    let pdu = GfxPdu::WireToSurface1(WireToSurface1Pdu {
        surface_id: 1,
        codec_id: Codec1Type::Uncompressed,
        pixel_format: PixelFormat::XRgb,
        destination_rectangle: ExclusiveRectangle {
            left: 0,
            top: 0,
            right: 4,
            bottom: 4,
        },
        bitmap_data: vec![0u8; 4 * 4 * 4],
    });
    client
        .process(0, &encode_for_process(&pdu))
        .expect("uncompressed should succeed");
}

#[test]
fn client_dispatches_avc420_via_process() {
    let mut client = setup_active_client_with_surface(Some(Box::new(MockH264Decoder)), 1, 16, 16);

    // Build minimal AVC420 bitmap stream
    let mut bitmap_data = Vec::new();
    bitmap_data.extend_from_slice(&1u32.to_le_bytes()); // nRect = 1
    bitmap_data.extend_from_slice(&0u16.to_le_bytes()); // left
    bitmap_data.extend_from_slice(&0u16.to_le_bytes()); // top
    bitmap_data.extend_from_slice(&15u16.to_le_bytes()); // right
    bitmap_data.extend_from_slice(&15u16.to_le_bytes()); // bottom
    bitmap_data.push(22); // qp
    bitmap_data.push(100); // quality
    bitmap_data.extend_from_slice(&[0x00, 0x00, 0x00, 0x01, 0x67]); // fake H.264

    let pdu = GfxPdu::WireToSurface1(WireToSurface1Pdu {
        surface_id: 1,
        codec_id: Codec1Type::Avc420,
        pixel_format: PixelFormat::XRgb,
        destination_rectangle: ExclusiveRectangle {
            left: 0,
            top: 0,
            right: 16,
            bottom: 16,
        },
        bitmap_data,
    });
    client
        .process(0, &encode_for_process(&pdu))
        .expect("AVC420 should succeed");
}

#[test]
fn client_skips_avc420_without_decoder() {
    let mut client = setup_active_client_with_surface(None, 1, 16, 16);

    let mut bitmap_data = Vec::new();
    bitmap_data.extend_from_slice(&1u32.to_le_bytes());
    bitmap_data.extend_from_slice(&0u16.to_le_bytes());
    bitmap_data.extend_from_slice(&0u16.to_le_bytes());
    bitmap_data.extend_from_slice(&15u16.to_le_bytes());
    bitmap_data.extend_from_slice(&15u16.to_le_bytes());
    bitmap_data.push(22);
    bitmap_data.push(100);
    bitmap_data.extend_from_slice(&[0x00, 0x00, 0x00, 0x01, 0x67]);

    let pdu = GfxPdu::WireToSurface1(WireToSurface1Pdu {
        surface_id: 1,
        codec_id: Codec1Type::Avc420,
        pixel_format: PixelFormat::XRgb,
        destination_rectangle: ExclusiveRectangle {
            left: 0,
            top: 0,
            right: 16,
            bottom: 16,
        },
        bitmap_data,
    });
    client
        .process(0, &encode_for_process(&pdu))
        .expect("should succeed without decoder");
}

#[test]
fn client_frame_ordering_via_process() {
    let mut client = setup_active_client_with_surface(None, 1, 4, 4);

    // StartFrame
    let start = GfxPdu::StartFrame(StartFramePdu {
        timestamp: Timestamp {
            milliseconds: 0,
            seconds: 0,
            minutes: 0,
            hours: 0,
        },
        frame_id: 1,
    });
    client.process(0, &encode_for_process(&start)).expect("start frame");

    // WireToSurface1 (uncompressed)
    let wire = GfxPdu::WireToSurface1(WireToSurface1Pdu {
        surface_id: 1,
        codec_id: Codec1Type::Uncompressed,
        pixel_format: PixelFormat::XRgb,
        destination_rectangle: ExclusiveRectangle {
            left: 0,
            top: 0,
            right: 4,
            bottom: 4,
        },
        bitmap_data: vec![0u8; 4 * 4 * 4],
    });
    client.process(0, &encode_for_process(&wire)).expect("wire to surface");

    // EndFrame should produce FrameAcknowledge
    let end = GfxPdu::EndFrame(EndFramePdu { frame_id: 1 });
    let responses = client.process(0, &encode_for_process(&end)).expect("end frame");

    assert_eq!(responses.len(), 1);
    assert_eq!(client.total_frames_decoded(), 1);
}

// ============================================================================
// Tests: Surface Lifecycle (via process())
// ============================================================================

#[test]
fn client_creates_and_queries_surface() {
    let client = setup_active_client_with_surface(None, 7, 1920, 1080);

    let surface = client.get_surface(7);
    assert!(surface.is_some(), "surface 7 should exist after creation");
    assert_eq!(surface.unwrap().width, 1920);
    assert_eq!(surface.unwrap().height, 1080);

    // Nonexistent surface
    assert!(client.get_surface(99).is_none());
}

#[test]
fn client_deletes_surface_via_process() {
    let mut client = setup_active_client_with_surface(None, 5, 100, 100);
    assert!(client.get_surface(5).is_some());

    let delete = GfxPdu::DeleteSurface(DeleteSurfacePdu { surface_id: 5 });
    client
        .process(0, &encode_for_process(&delete))
        .expect("delete should succeed");

    assert!(client.get_surface(5).is_none(), "surface should be gone after delete");
}

#[test]
fn client_keeps_surfaces_across_reset_via_process() {
    let mut client = setup_active_client_with_surface(None, 1, 100, 100);

    // Create a second surface
    let create2 = GfxPdu::CreateSurface(CreateSurfacePdu {
        surface_id: 2,
        width: 200,
        height: 200,
        pixel_format: PixelFormat::XRgb,
    });
    client.process(0, &encode_for_process(&create2)).expect("create 2");
    assert!(client.get_surface(1).is_some());
    assert!(client.get_surface(2).is_some());

    // Per MS-RDPEGFX 3.3.5.14, ResetGraphics only resizes the Graphics Output Buffer;
    // it does not destroy surfaces.
    let reset = GfxPdu::ResetGraphics(ResetGraphicsPdu {
        width: 1920,
        height: 1080,
        monitors: vec![],
    });
    client.process(0, &encode_for_process(&reset)).expect("reset");

    let surface1 = client.get_surface(1);
    assert!(surface1.is_some(), "surface 1 should survive reset");
    assert_eq!(surface1.unwrap().width, 100);
    assert_eq!(surface1.unwrap().height, 100);

    let surface2 = client.get_surface(2);
    assert!(surface2.is_some(), "surface 2 should survive reset");
    assert_eq!(surface2.unwrap().width, 200);
    assert_eq!(surface2.unwrap().height, 200);

    assert_eq!(
        client.take_reset_graphics(),
        Some((1920, 1080)),
        "the reset dimensions must be available to the session exactly once"
    );
    assert_eq!(
        client.take_reset_graphics(),
        None,
        "take_reset_graphics should not return the same reset twice"
    );
}

/// A same-size `ResetGraphics` does not invalidate the frame committed just before
/// it. Per MS-RDPEGFX 3.3.5.14, `ResetGraphics` only resizes the Graphics Output
/// Buffer; a reset that does not change the output's dimensions resizes nothing,
/// so the delta produced by the frame preceding it is still valid against the
/// (unchanged) output and must stay drainable.
#[test]
fn client_keeps_the_delta_across_a_same_size_reset_via_process() {
    const WIDTH: u16 = 100;
    const HEIGHT: u16 = 100;

    let mut client = setup_active_client_with_surface(None, 1, WIDTH, HEIGHT);

    // Establish the output's initial dimensions, matching the surface.
    let initial_reset = GfxPdu::ResetGraphics(ResetGraphicsPdu {
        width: u32::from(WIDTH),
        height: u32::from(HEIGHT),
        monitors: vec![],
    });
    client
        .process(0, &encode_for_process(&initial_reset))
        .expect("initial reset");

    let map = GfxPdu::MapSurfaceToOutput(MapSurfaceToOutputPdu {
        surface_id: 1,
        output_origin_x: 0,
        output_origin_y: 0,
    });
    client.process(0, &encode_for_process(&map)).expect("map surface");
    let end_map = GfxPdu::EndFrame(EndFramePdu { frame_id: 0 });
    client.process(0, &encode_for_process(&end_map)).expect("end map frame");
    let _ = client.drain_output(); // discard the mapping delta

    let start = GfxPdu::StartFrame(StartFramePdu {
        timestamp: Timestamp {
            milliseconds: 0,
            seconds: 0,
            minutes: 0,
            hours: 0,
        },
        frame_id: 1,
    });
    client.process(0, &encode_for_process(&start)).expect("start frame");

    let fill = GfxPdu::SolidFill(SolidFillPdu {
        surface_id: 1,
        fill_pixel: Color {
            b: 0x33,
            g: 0x22,
            r: 0x11,
            xa: 0xFF,
        },
        rectangles: vec![ExclusiveRectangle {
            left: 0,
            top: 0,
            right: 4,
            bottom: 2,
        }],
    });
    client.process(0, &encode_for_process(&fill)).expect("solid fill");

    let end = GfxPdu::EndFrame(EndFramePdu { frame_id: 1 });
    client.process(0, &encode_for_process(&end)).expect("end frame");

    // Same size as the output already has: nothing resizes.
    let reset = GfxPdu::ResetGraphics(ResetGraphicsPdu {
        width: u32::from(WIDTH),
        height: u32::from(HEIGHT),
        monitors: vec![],
    });
    client.process(0, &encode_for_process(&reset)).expect("same-size reset");

    let updates = client.drain_output();
    assert_eq!(
        updates.len(),
        1,
        "the delta committed just before a same-size reset must still be drainable"
    );
    assert_eq!(
        (
            updates[0].region.left,
            updates[0].region.top,
            updates[0].region.right,
            updates[0].region.bottom
        ),
        (0, 0, 4, 2),
        "the drained region must match what the SolidFill dirtied"
    );
}

/// The other half of the pair: a `ResetGraphics` that DOES change the output's
/// dimensions discards the preceding frame's delta, because it was clipped against
/// the output size that no longer applies. Per MS-RDPEGFX 3.3.5.14, a dimension
/// change resizes the Graphics Output Buffer, invalidating what was clipped to the
/// old one.
#[test]
fn client_discards_the_delta_across_a_different_size_reset_via_process() {
    const WIDTH: u16 = 100;
    const HEIGHT: u16 = 100;

    let mut client = setup_active_client_with_surface(None, 1, WIDTH, HEIGHT);

    // Establish the output's initial dimensions, matching the surface.
    let initial_reset = GfxPdu::ResetGraphics(ResetGraphicsPdu {
        width: u32::from(WIDTH),
        height: u32::from(HEIGHT),
        monitors: vec![],
    });
    client
        .process(0, &encode_for_process(&initial_reset))
        .expect("initial reset");

    let map = GfxPdu::MapSurfaceToOutput(MapSurfaceToOutputPdu {
        surface_id: 1,
        output_origin_x: 0,
        output_origin_y: 0,
    });
    client.process(0, &encode_for_process(&map)).expect("map surface");
    let end_map = GfxPdu::EndFrame(EndFramePdu { frame_id: 0 });
    client.process(0, &encode_for_process(&end_map)).expect("end map frame");
    let _ = client.drain_output(); // discard the mapping delta

    let start = GfxPdu::StartFrame(StartFramePdu {
        timestamp: Timestamp {
            milliseconds: 0,
            seconds: 0,
            minutes: 0,
            hours: 0,
        },
        frame_id: 1,
    });
    client.process(0, &encode_for_process(&start)).expect("start frame");

    let fill = GfxPdu::SolidFill(SolidFillPdu {
        surface_id: 1,
        fill_pixel: Color {
            b: 0x33,
            g: 0x22,
            r: 0x11,
            xa: 0xFF,
        },
        rectangles: vec![ExclusiveRectangle {
            left: 0,
            top: 0,
            right: 4,
            bottom: 2,
        }],
    });
    client.process(0, &encode_for_process(&fill)).expect("solid fill");

    let end = GfxPdu::EndFrame(EndFramePdu { frame_id: 1 });
    client.process(0, &encode_for_process(&end)).expect("end frame");

    // Different size than the output already has.
    let reset = GfxPdu::ResetGraphics(ResetGraphicsPdu {
        width: u32::from(WIDTH) * 2,
        height: u32::from(HEIGHT) * 2,
        monitors: vec![],
    });
    client
        .process(0, &encode_for_process(&reset))
        .expect("different-size reset");

    let updates = client.drain_output();
    assert!(
        updates.is_empty(),
        "the delta committed before a resizing reset must be discarded, \
         since it was clipped against the output size that no longer applies"
    );
}

// ============================================================================
// Tests: Error Handling
// ============================================================================

#[test]
fn client_rejects_wire_to_unknown_surface() {
    let mut client = setup_active_client_with_surface(None, 1, 4, 4);

    let pdu = GfxPdu::WireToSurface1(WireToSurface1Pdu {
        surface_id: 99, // does not exist
        codec_id: Codec1Type::Uncompressed,
        pixel_format: PixelFormat::XRgb,
        destination_rectangle: ExclusiveRectangle {
            left: 0,
            top: 0,
            right: 4,
            bottom: 4,
        },
        bitmap_data: vec![0u8; 4 * 4 * 4],
    });
    let result = client.process(0, &encode_for_process(&pdu));
    assert!(result.is_err(), "should reject write to nonexistent surface");
}

#[test]
fn client_rejects_invalid_rectangle_ordering() {
    let mut client = setup_active_client_with_surface(None, 1, 100, 100);

    // left > right
    let pdu = GfxPdu::WireToSurface1(WireToSurface1Pdu {
        surface_id: 1,
        codec_id: Codec1Type::Uncompressed,
        pixel_format: PixelFormat::XRgb,
        destination_rectangle: ExclusiveRectangle {
            left: 50,
            top: 0,
            right: 10,
            bottom: 10,
        },
        bitmap_data: vec![0u8; 4],
    });
    let result = client.process(0, &encode_for_process(&pdu));
    assert!(result.is_err(), "left > right should be rejected");
}

#[test]
fn client_tolerates_out_of_bounds_rectangle() {
    let mut client = setup_active_client_with_surface(None, 1, 100, 100);

    // Rectangle exceeds surface dimensions. The client logs a warning
    // but continues processing (defensive: avoid disconnecting for a
    // recoverable server-side error).
    let pdu = GfxPdu::WireToSurface1(WireToSurface1Pdu {
        surface_id: 1,
        codec_id: Codec1Type::Uncompressed,
        pixel_format: PixelFormat::XRgb,
        destination_rectangle: ExclusiveRectangle {
            left: 0,
            top: 0,
            right: 200, // exceeds surface width of 100
            bottom: 50,
        },
        bitmap_data: vec![0u8; 200 * 50 * 4],
    });
    let result = client.process(0, &encode_for_process(&pdu));
    assert!(
        result.is_ok(),
        "out-of-bounds rectangle should be tolerated (warn, not error)"
    );
}

// ============================================================================
// Tests: ClearCodec Decode
// ============================================================================

#[test]
fn client_dispatches_clearcodec_via_process() {
    let mut client = setup_active_client_with_surface(None, 1, 4, 4);

    // Encode a valid ClearCodec frame: 4x4 solid red (BGRA: B=0, G=0, R=255, A=255)
    let mut cc_enc = ClearCodecEncoder::new();
    let bgra: Vec<u8> = (0..16).flat_map(|_| [0x00u8, 0x00, 0xFF, 0xFF]).collect();
    let cc_data = cc_enc.encode(&bgra, 4, 4);

    let pdu = GfxPdu::WireToSurface1(WireToSurface1Pdu {
        surface_id: 1,
        codec_id: Codec1Type::ClearCodec,
        pixel_format: PixelFormat::XRgb,
        destination_rectangle: ExclusiveRectangle {
            left: 0,
            top: 0,
            right: 4,
            bottom: 4,
        },
        bitmap_data: cc_data,
    });
    client
        .process(0, &encode_for_process(&pdu))
        .expect("ClearCodec decode should succeed");
}

#[test]
fn client_clearcodec_produces_rgba_output() {
    use std::sync::{Arc, Mutex};

    #[derive(Default)]
    struct CapturedBitmap {
        data: Option<Vec<u8>>,
        codec: Option<Codec1Type>,
    }

    struct CapturingHandler {
        captured: Arc<Mutex<CapturedBitmap>>,
    }

    impl GraphicsPipelineHandler for CapturingHandler {
        fn on_bitmap_updated(&mut self, update: &BitmapUpdate) {
            let mut cap = self.captured.lock().unwrap();
            cap.data = Some(update.data.clone());
            cap.codec = Some(update.codec_id);
        }
    }

    let captured = Arc::new(Mutex::new(CapturedBitmap::default()));
    let handler = CapturingHandler {
        captured: Arc::clone(&captured),
    };
    let mut client = GraphicsPipelineClient::new(Box::new(handler), None);

    // Activate and create surface
    let confirm = GfxPdu::CapabilitiesConfirm(CapabilitiesConfirmPdu::from_typed(&CapabilitySet::V8 {
        flags: CapabilitiesV8Flags::empty(),
    }));
    client.process(0, &encode_for_process(&confirm)).expect("confirm");

    let create = GfxPdu::CreateSurface(CreateSurfacePdu {
        surface_id: 1,
        width: 2,
        height: 1,
        pixel_format: PixelFormat::XRgb,
    });
    client.process(0, &encode_for_process(&create)).expect("create surface");

    // Encode a 2x1 frame: pixel 0 = blue (BGRA: FF,00,00,FF), pixel 1 = green (BGRA: 00,FF,00,FF)
    let mut cc_enc = ClearCodecEncoder::new();
    let bgra = vec![0xFF, 0x00, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF];
    let cc_data = cc_enc.encode(&bgra, 2, 1);

    let pdu = GfxPdu::WireToSurface1(WireToSurface1Pdu {
        surface_id: 1,
        codec_id: Codec1Type::ClearCodec,
        pixel_format: PixelFormat::XRgb,
        destination_rectangle: ExclusiveRectangle {
            left: 0,
            top: 0,
            right: 2,
            bottom: 1,
        },
        bitmap_data: cc_data,
    });
    client
        .process(0, &encode_for_process(&pdu))
        .expect("ClearCodec should succeed");

    // Verify BGRA-to-RGBA conversion: blue (BGRA FF,00,00,FF) -> RGBA (00,00,FF,FF)
    let cap = captured.lock().unwrap();
    assert_eq!(cap.codec, Some(Codec1Type::ClearCodec));
    let bitmap = cap.data.as_ref().expect("handler should have received bitmap");
    assert_eq!(bitmap.len(), 8, "2 pixels * 4 bytes");
    assert_eq!(&bitmap[0..4], &[0x00, 0x00, 0xFF, 0xFF], "pixel 0: blue in RGBA");
    assert_eq!(&bitmap[4..8], &[0x00, 0xFF, 0x00, 0xFF], "pixel 1: green in RGBA");
}

/// A `ResetGraphics` must not drop the ClearCodec glyph cache. Per MS-RDPEGFX 3.3.5.14 it only
/// resizes the Graphics Output Buffer; cache lifetime is driven by the stream through
/// CLEARCODEC_FLAG_CACHE_RESET (2.2.4.1). A server may therefore send a GLYPH_HIT after a reset
/// for a glyph it cached before it, and the client has to still have it.
///
/// The encoder is reused across the reset on purpose: `ClearCodecEncoder::encode` emits a glyph
/// hit when it finds the same pixels in its own cache, so re-encoding the identical tile is what
/// produces the post-reset GLYPH_HIT. A fresh encoder would start with an empty cache and send a
/// full frame instead, which exercises nothing.
#[test]
fn client_clearcodec_glyph_cache_survives_reset() {
    let mut client = setup_active_client_with_surface(None, 1, 4, 4);

    // Encode and decode a ClearCodec frame
    let mut cc_enc = ClearCodecEncoder::new();
    let bgra: Vec<u8> = (0..16).flat_map(|_| [0x00u8, 0x00, 0xFF, 0xFF]).collect();
    let cc_data = cc_enc.encode(&bgra, 4, 4);

    let pdu = GfxPdu::WireToSurface1(WireToSurface1Pdu {
        surface_id: 1,
        codec_id: Codec1Type::ClearCodec,
        pixel_format: PixelFormat::XRgb,
        destination_rectangle: ExclusiveRectangle {
            left: 0,
            top: 0,
            right: 4,
            bottom: 4,
        },
        bitmap_data: cc_data,
    });
    client
        .process(0, &encode_for_process(&pdu))
        .expect("pre-reset ClearCodec should succeed");

    // Reset graphics. Surfaces are cleared; the ClearCodec caches are not.
    let reset = GfxPdu::ResetGraphics(ResetGraphicsPdu {
        width: 1920,
        height: 1080,
        monitors: vec![],
    });
    client
        .process(0, &encode_for_process(&reset))
        .expect("reset should succeed");

    // Re-create surface and decode another frame
    let create = GfxPdu::CreateSurface(CreateSurfacePdu {
        surface_id: 2,
        width: 4,
        height: 4,
        pixel_format: PixelFormat::XRgb,
    });
    client
        .process(0, &encode_for_process(&create))
        .expect("create surface after reset");

    // Same encoder, same pixels: this hits its glyph cache and emits a GLYPH_HIT referencing the
    // index the decoder stored before the reset.
    let cc_data2 = cc_enc.encode(&bgra, 4, 4);

    let pdu2 = GfxPdu::WireToSurface1(WireToSurface1Pdu {
        surface_id: 2,
        codec_id: Codec1Type::ClearCodec,
        pixel_format: PixelFormat::XRgb,
        destination_rectangle: ExclusiveRectangle {
            left: 0,
            top: 0,
            right: 4,
            bottom: 4,
        },
        bitmap_data: cc_data2,
    });
    client
        .process(0, &encode_for_process(&pdu2))
        .expect("post-reset GLYPH_HIT should resolve against the surviving glyph cache");
}

// ============================================================================
// Tests: Multiple Frames
// ============================================================================

#[test]
fn client_tracks_frame_count_across_multiple_frames() {
    let mut client = setup_active_client_with_surface(None, 1, 4, 4);

    for frame_id in 1..=5 {
        let end = GfxPdu::EndFrame(EndFramePdu { frame_id });
        client.process(0, &encode_for_process(&end)).expect("end frame");
    }

    assert_eq!(client.total_frames_decoded(), 5);
}

// ============================================================================
// Tests: Close
// ============================================================================

#[test]
fn client_close_transitions_to_inactive() {
    let mut client = setup_active_client_with_surface(None, 1, 4, 4);
    assert!(client.is_active());

    client.close(0);
    assert!(!client.is_active(), "client should not be active after close");
}

// ============================================================================
// Tests: Codec/Frame-Ack Telemetry
// ============================================================================

use std::sync::{Arc, Mutex};

/// Shared, lock-guarded log of telemetry callback invocations.
type Recorded<T> = Arc<Mutex<Vec<T>>>;

/// Handler recording `on_codec_processed` and `on_frame_acknowledged` calls,
/// used to pin the telemetry callbacks introduced for surface-command cost
/// reporting and frame acknowledgement reporting.
#[derive(Default)]
struct RecordingHandler {
    codec_processed: Recorded<CodecProcessed>,
    frame_acks: Recorded<FrameAcknowledgePdu>,
}

impl RecordingHandler {
    fn new() -> (Self, Recorded<CodecProcessed>, Recorded<FrameAcknowledgePdu>) {
        let codec_processed = Arc::new(Mutex::new(Vec::new()));
        let frame_acks = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                codec_processed: Arc::clone(&codec_processed),
                frame_acks: Arc::clone(&frame_acks),
            },
            codec_processed,
            frame_acks,
        )
    }
}

impl GraphicsPipelineHandler for RecordingHandler {
    fn on_codec_processed(&mut self, info: &CodecProcessed) {
        self.codec_processed.lock().unwrap().push(*info);
    }

    fn on_frame_acknowledged(&mut self, ack: &FrameAcknowledgePdu) {
        self.frame_acks.lock().unwrap().push(ack.clone());
    }
}

fn setup_recording_client_with_surface(
    surface_id: u16,
    width: u16,
    height: u16,
) -> (
    GraphicsPipelineClient,
    Recorded<CodecProcessed>,
    Recorded<FrameAcknowledgePdu>,
) {
    let (handler, codec_processed, frame_acks) = RecordingHandler::new();
    let mut client = GraphicsPipelineClient::new(Box::new(handler), None);

    let confirm = GfxPdu::CapabilitiesConfirm(CapabilitiesConfirmPdu::from_typed(&CapabilitySet::V8 {
        flags: CapabilitiesV8Flags::empty(),
    }));
    client
        .process(0, &encode_for_process(&confirm))
        .expect("confirm should succeed");

    let create = GfxPdu::CreateSurface(CreateSurfacePdu {
        surface_id,
        width,
        height,
        pixel_format: PixelFormat::XRgb,
    });
    client
        .process(0, &encode_for_process(&create))
        .expect("create surface should succeed");

    (client, codec_processed, frame_acks)
}

#[test]
fn codec_processed_reports_surface_codec_payload_size() {
    let (mut client, codec_processed, _frame_acks) = setup_recording_client_with_surface(1, 4, 4);

    let bitmap_data = vec![0u8; 4 * 4 * 4];
    let expected_len = bitmap_data.len();

    let pdu = GfxPdu::WireToSurface1(WireToSurface1Pdu {
        surface_id: 1,
        codec_id: Codec1Type::Uncompressed,
        pixel_format: PixelFormat::XRgb,
        destination_rectangle: ExclusiveRectangle {
            left: 0,
            top: 0,
            right: 4,
            bottom: 4,
        },
        bitmap_data,
    });
    client
        .process(0, &encode_for_process(&pdu))
        .expect("uncompressed should succeed");

    let recorded = codec_processed.lock().unwrap();
    assert_eq!(recorded.len(), 1, "expected exactly one on_codec_processed call");
    assert_eq!(recorded[0].surface_id, 1);
    assert_eq!(recorded[0].codec, DecodedCodec::Wire1(Codec1Type::Uncompressed));
    assert_eq!(recorded[0].payload_bytes, expected_len);
}

#[test]
fn codec_processed_fires_for_a_forwarded_avc444_pdu() {
    let (mut client, codec_processed, _frame_acks) = setup_recording_client_with_surface(1, 4, 4);

    let bitmap_data = vec![0u8; 12];
    let expected_len = bitmap_data.len();

    let pdu = GfxPdu::WireToSurface1(WireToSurface1Pdu {
        surface_id: 1,
        codec_id: Codec1Type::Avc444,
        pixel_format: PixelFormat::XRgb,
        destination_rectangle: ExclusiveRectangle {
            left: 0,
            top: 0,
            right: 4,
            bottom: 4,
        },
        bitmap_data,
    });
    client
        .process(0, &encode_for_process(&pdu))
        .expect("AVC444 forward should succeed");

    let recorded = codec_processed.lock().unwrap();
    assert_eq!(
        recorded.len(),
        1,
        "a codec forwarded to on_unhandled_pdu must still report once"
    );
    assert_eq!(recorded[0].surface_id, 1);
    assert_eq!(recorded[0].codec, DecodedCodec::Wire1(Codec1Type::Avc444));
    assert_eq!(recorded[0].payload_bytes, expected_len);
}

#[test]
fn codec_processed_is_not_reported_when_handling_fails() {
    let (mut client, codec_processed, _frame_acks) = setup_recording_client_with_surface(1, 4, 4);

    let pdu = GfxPdu::WireToSurface1(WireToSurface1Pdu {
        surface_id: 99, // unknown surface
        codec_id: Codec1Type::Uncompressed,
        pixel_format: PixelFormat::XRgb,
        destination_rectangle: ExclusiveRectangle {
            left: 0,
            top: 0,
            right: 4,
            bottom: 4,
        },
        bitmap_data: vec![0u8; 4 * 4 * 4],
    });
    let result = client.process(0, &encode_for_process(&pdu));
    assert!(result.is_err(), "unknown surface should still be rejected");

    let recorded = codec_processed.lock().unwrap();
    assert!(
        recorded.is_empty(),
        "on_codec_processed must not fire when handling returns Err"
    );
}

#[test]
fn codec_processed_fires_for_a_skipped_progressive_pdu() {
    let (mut client, codec_processed, _frame_acks) = setup_recording_client_with_surface(1, 4, 4);

    // Empty bitmap_data is not a valid progressive stream: decode_bitmap fails and
    // handle_wire_to_surface2 hits the skip path, returning Ok(()).
    let pdu = WireToSurface2Pdu {
        surface_id: 1,
        codec_id: Codec2Type::RemoteFxProgressive,
        codec_context_id: 0,
        pixel_format: PixelFormat::XRgb,
        bitmap_data: Vec::new(),
    };
    let expected_len = pdu.bitmap_data.len();
    let gfx_pdu = GfxPdu::WireToSurface2(pdu);
    client
        .process(0, &encode_for_process(&gfx_pdu))
        .expect("skipped progressive PDU should still return Ok");

    let recorded = codec_processed.lock().unwrap();
    assert_eq!(
        recorded.len(),
        1,
        "a progressive PDU skipped after a decode failure must still report once"
    );
    assert_eq!(recorded[0].surface_id, 1);
    assert_eq!(recorded[0].codec, DecodedCodec::Progressive);
    assert_eq!(recorded[0].payload_bytes, expected_len);
}

#[test]
fn frame_acknowledged_reports_the_ack_that_is_sent() {
    let (mut client, _codec_processed, frame_acks) = setup_recording_client_with_surface(1, 4, 4);

    let start = GfxPdu::StartFrame(StartFramePdu {
        timestamp: Timestamp {
            milliseconds: 0,
            seconds: 0,
            minutes: 0,
            hours: 0,
        },
        frame_id: 7,
    });
    client.process(0, &encode_for_process(&start)).expect("start frame");

    let end = GfxPdu::EndFrame(EndFramePdu { frame_id: 7 });
    let responses = client
        .process(0, &encode_for_process(&end))
        .expect("end frame should succeed");
    assert_eq!(responses.len(), 1, "should produce exactly one FrameAcknowledge");

    let encoded = encode_vec(responses[0].as_ref()).expect("encode should succeed");
    let mut cursor = ReadCursor::new(&encoded);
    let sent_ack = match GfxPdu::decode(&mut cursor).expect("decode should succeed") {
        GfxPdu::FrameAcknowledge(ack) => ack,
        other => panic!("expected FrameAcknowledge, got {other:?}"),
    };

    let recorded = frame_acks.lock().unwrap();
    assert_eq!(recorded.len(), 1, "expected exactly one on_frame_acknowledged call");
    assert_eq!(recorded[0], sent_ack, "reported ack must match the one that is sent");
}

// ============================================================================
// Tests: Persistent cache import (Tessera)
// ============================================================================

#[derive(Default)]
struct CacheLog {
    import_calls: u32,
    stored: Vec<(u64, u16, u16, u16, Vec<u8>)>,
    outcomes: Vec<CacheImportOutcome>,
}

struct CacheHandler {
    tiles: Vec<CacheImportTile>,
    log: Arc<Mutex<CacheLog>>,
}

impl GraphicsPipelineHandler for CacheHandler {
    fn on_cache_tile_stored(&mut self, cache_key: u64, cache_slot: u16, width: u16, height: u16, rgba: &[u8]) {
        self.log
            .lock()
            .unwrap()
            .stored
            .push((cache_key, cache_slot, width, height, rgba.to_vec()));
    }
    fn cache_import_tiles(&mut self) -> Vec<CacheImportTile> {
        self.log.lock().unwrap().import_calls += 1;
        core::mem::take(&mut self.tiles)
    }
    fn on_cache_import_outcome(&mut self, outcome: &CacheImportOutcome) {
        self.log.lock().unwrap().outcomes.push(outcome.clone());
    }
}

fn solid_tile(cache_key: u64, width: u16, height: u16, fill: u8) -> CacheImportTile {
    CacheImportTile {
        cache_key,
        width,
        height,
        data: vec![fill; usize::from(width) * usize::from(height) * 4],
    }
}

fn cache_client(tiles: Vec<CacheImportTile>) -> (GraphicsPipelineClient, Arc<Mutex<CacheLog>>) {
    let log = Arc::new(Mutex::new(CacheLog::default()));
    let handler = CacheHandler {
        tiles,
        log: Arc::clone(&log),
    };
    (GraphicsPipelineClient::new(Box::new(handler), None), log)
}

fn confirm_v8(flags: CapabilitiesV8Flags) -> Vec<u8> {
    encode_for_process(&GfxPdu::CapabilitiesConfirm(CapabilitiesConfirmPdu::from_typed(
        &CapabilitySet::V8 { flags },
    )))
}

fn decode_gfx(msg: &ironrdp_dvc::DvcMessage) -> GfxPdu {
    let encoded = encode_vec(msg.as_ref()).expect("encode");
    GfxPdu::decode(&mut ReadCursor::new(&encoded)).expect("decode")
}

fn frame(id: u32, pdus: Vec<GfxPdu>) -> Vec<GfxPdu> {
    let ts = Timestamp {
        milliseconds: 0,
        seconds: 0,
        minutes: 0,
        hours: 0,
    };
    let mut all = vec![GfxPdu::StartFrame(StartFramePdu {
        timestamp: ts,
        frame_id: id,
    })];
    all.extend(pdus);
    all.push(GfxPdu::EndFrame(EndFramePdu { frame_id: id }));
    all
}

fn process_all(client: &mut GraphicsPipelineClient, pdus: Vec<GfxPdu>) {
    for pdu in pdus {
        client.process(0, &encode_for_process(&pdu)).expect("process");
    }
}

/// Paint drained updates, in order, into a `width x height` RGBA canvas. Checking the
/// canvas instead of individual updates keeps the tests independent of how the
/// compositor splits or merges dirty regions.
fn compose(updates: &[ironrdp_egfx::compositor::OutputUpdate], width: usize, height: usize) -> Vec<u8> {
    let mut canvas = vec![0u8; width * height * 4];
    for u in updates {
        let (l, t) = (usize::from(u.region.left), usize::from(u.region.top));
        let w = usize::from(u.region.right - u.region.left);
        for row in 0..usize::from(u.region.bottom - u.region.top) {
            let dst = ((t + row) * width + l) * 4;
            canvas[dst..dst + w * 4].copy_from_slice(&u.data[row * w * 4..(row + 1) * w * 4]);
        }
    }
    canvas
}

/// The `w x h` block at `(x, y)` of a canvas `width` pixels wide, row-major RGBA.
fn block(canvas: &[u8], width: usize, x: usize, y: usize, w: usize, h: usize) -> Vec<u8> {
    (0..h)
        .flat_map(|r| canvas[((y + r) * width + x) * 4..((y + r) * width + x + w) * 4].to_vec())
        .collect()
}

#[test]
fn client_offers_cache_import_after_caps_confirm() {
    let (mut client, log) = cache_client(vec![solid_tile(0xAA, 2, 2, 1), solid_tile(0xBB, 4, 1, 2)]);
    let out = client
        .process(0, &confirm_v8(CapabilitiesV8Flags::empty()))
        .expect("confirm");
    assert_eq!(out.len(), 1, "the offer is the one response to the confirm");
    let GfxPdu::CacheImportOffer(offer) = decode_gfx(&out[0]) else {
        panic!("expected CacheImportOffer")
    };
    let entries: Vec<(u64, u32)> = offer
        .cache_entries
        .iter()
        .map(|e| (e.cache_key, e.bitmap_len))
        .collect();
    assert_eq!(entries, vec![(0xAA, 16), (0xBB, 16)]);
    assert_eq!(log.lock().unwrap().import_calls, 1);
}

#[test]
fn client_does_not_offer_under_small_cache() {
    let (mut client, log) = cache_client(vec![solid_tile(0xAA, 2, 2, 1)]);
    let out = client
        .process(0, &confirm_v8(CapabilitiesV8Flags::SMALL_CACHE))
        .expect("confirm");
    assert!(out.is_empty());
    assert_eq!(log.lock().unwrap().import_calls, 0, "tiles must not even be requested");
}

#[test]
fn client_offers_at_most_once() {
    let (mut client, log) = cache_client(vec![solid_tile(0xAA, 2, 2, 1)]);
    assert_eq!(
        client
            .process(0, &confirm_v8(CapabilitiesV8Flags::empty()))
            .expect("confirm")
            .len(),
        1
    );
    assert!(
        client
            .process(0, &confirm_v8(CapabilitiesV8Flags::empty()))
            .expect("confirm 2")
            .is_empty()
    );
    assert_eq!(log.lock().unwrap().import_calls, 1);
}

#[test]
fn client_sends_no_offer_without_tiles() {
    let (mut client, _log) = cache_client(Vec::new());
    assert!(
        client
            .process(0, &confirm_v8(CapabilitiesV8Flags::empty()))
            .expect("confirm")
            .is_empty()
    );
}

/// The invariant this feature must never break: every slot the reply names paints
/// the offered tile at the same index, and the outcome lists exactly those slots.
#[test]
fn client_fills_exactly_the_accepted_slots() {
    let (mut client, log) = cache_client(vec![
        solid_tile(1, 2, 2, 0x11),
        solid_tile(2, 2, 2, 0x22),
        solid_tile(3, 2, 2, 0x33),
    ]);
    client
        .process(0, &confirm_v8(CapabilitiesV8Flags::empty()))
        .expect("confirm");
    process_all(
        &mut client,
        vec![
            GfxPdu::CacheImportReply(CacheImportReplyPdu {
                cache_slots: vec![5, 0, 9],
            }),
            GfxPdu::ResetGraphics(ResetGraphicsPdu {
                width: 16,
                height: 16,
                monitors: vec![],
            }),
            GfxPdu::CreateSurface(CreateSurfacePdu {
                surface_id: 1,
                width: 16,
                height: 16,
                pixel_format: PixelFormat::XRgb,
            }),
            GfxPdu::MapSurfaceToOutput(MapSurfaceToOutputPdu {
                surface_id: 1,
                output_origin_x: 0,
                output_origin_y: 0,
            }),
        ],
    );
    let _ = client.drain_output();
    process_all(
        &mut client,
        frame(
            1,
            vec![
                GfxPdu::CacheToSurface(CacheToSurfacePdu {
                    cache_slot: 5,
                    surface_id: 1,
                    destination_points: vec![Point { x: 0, y: 0 }],
                }),
                GfxPdu::CacheToSurface(CacheToSurfacePdu {
                    cache_slot: 9,
                    surface_id: 1,
                    destination_points: vec![Point { x: 8, y: 8 }],
                }),
            ],
        ),
    );
    let canvas = compose(&client.drain_output(), 16, 16);
    assert_eq!(
        block(&canvas, 16, 0, 0, 2, 2),
        vec![0x11; 16],
        "slot 5 must hold offer entry 0"
    );
    assert_eq!(
        block(&canvas, 16, 8, 8, 2, 2),
        vec![0x33; 16],
        "slot 9 must hold offer entry 2"
    );
    let log = log.lock().unwrap();
    assert_eq!(log.outcomes.len(), 1);
    assert_eq!(log.outcomes[0].imported, vec![(1, 5), (3, 9)]);
    assert!(log.outcomes[0].unfilled_slots.is_empty());
}

#[test]
fn client_reports_reply_slots_it_cannot_fill() {
    let (mut client, log) = cache_client(vec![solid_tile(1, 1, 1, 0x11)]);
    client
        .process(0, &confirm_v8(CapabilitiesV8Flags::empty()))
        .expect("confirm");
    process_all(
        &mut client,
        vec![GfxPdu::CacheImportReply(CacheImportReplyPdu {
            cache_slots: vec![4, 6, 8],
        })],
    );
    let log = log.lock().unwrap();
    assert_eq!(log.outcomes[0].imported, vec![(1, 4)]);
    assert_eq!(log.outcomes[0].unfilled_slots, vec![6, 8]);
}

#[test]
fn client_hands_stored_tiles_to_the_handler() {
    let (mut client, log) = cache_client(Vec::new());
    client
        .process(0, &confirm_v8(CapabilitiesV8Flags::empty()))
        .expect("confirm");
    process_all(
        &mut client,
        vec![GfxPdu::CreateSurface(CreateSurfacePdu {
            surface_id: 1,
            width: 8,
            height: 8,
            pixel_format: PixelFormat::XRgb,
        })],
    );
    process_all(
        &mut client,
        frame(
            1,
            vec![
                GfxPdu::SolidFill(SolidFillPdu {
                    surface_id: 1,
                    fill_pixel: Color {
                        b: 0x30,
                        g: 0x20,
                        r: 0x10,
                        xa: 0,
                    },
                    rectangles: vec![ExclusiveRectangle {
                        left: 0,
                        top: 0,
                        right: 2,
                        bottom: 1,
                    }],
                }),
                GfxPdu::SurfaceToCache(SurfaceToCachePdu {
                    surface_id: 1,
                    cache_key: 0xDEAD_BEEF,
                    cache_slot: 3,
                    source_rectangle: ExclusiveRectangle {
                        left: 0,
                        top: 0,
                        right: 2,
                        bottom: 1,
                    },
                }),
                // A tile from a surface that does not exist is not stored, so it is not handed over.
                GfxPdu::SurfaceToCache(SurfaceToCachePdu {
                    surface_id: 42,
                    cache_key: 0xFEED,
                    cache_slot: 4,
                    source_rectangle: ExclusiveRectangle {
                        left: 0,
                        top: 0,
                        right: 2,
                        bottom: 1,
                    },
                }),
            ],
        ),
    );
    let log = log.lock().unwrap();
    assert_eq!(
        log.stored,
        vec![(
            0xDEAD_BEEF,
            3,
            2,
            1,
            vec![0x10, 0x20, 0x30, 0xFF, 0x10, 0x20, 0x30, 0xFF]
        )]
    );
}

#[test]
fn client_close_releases_staged_import() {
    let (mut client, log) = cache_client(vec![solid_tile(1, 1, 1, 0x11)]);
    client
        .process(0, &confirm_v8(CapabilitiesV8Flags::empty()))
        .expect("confirm");
    client.close(0);
    // Nothing staged survives the close: a late reply fills nothing and reports the slot.
    client
        .process(
            0,
            &encode_for_process(&GfxPdu::CacheImportReply(CacheImportReplyPdu { cache_slots: vec![4] })),
        )
        .expect("process after close");
    let log = log.lock().unwrap();
    assert_eq!(log.outcomes.len(), 1);
    assert!(log.outcomes[0].imported.is_empty());
    assert_eq!(log.outcomes[0].unfilled_slots, vec![4]);
}
