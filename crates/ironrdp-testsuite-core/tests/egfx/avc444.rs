//! Client-side AVC444 / AVC444v2 decode tests ([MS-RDPEGFX] 3.3.8.3.2 / 3.3.8.3.3).
//!
//! These exercise `GraphicsPipelineClient::decode_avc444` (private) through the
//! public `process()` API, using a scripted `H264YuvDecoder` mock instead of a
//! real H.264 decoder: the mock hands back pre-built `Yuv420View`s (produced by
//! `yuv444::split::main_view` / `aux_view` from a synthetic 4:4:4 source), so the
//! test checks the reconstruction + colour conversion math end to end without any
//! actual video decoding.

use std::collections::{HashSet, VecDeque};
use std::sync::{Arc, Mutex};

use ironrdp_core::{Decode as _, Encode, ReadCursor, WriteCursor, encode_vec};
use ironrdp_dvc::DvcProcessor as _;
use ironrdp_egfx::client::{BitmapUpdate, GraphicsPipelineClient, GraphicsPipelineHandler, Surface};
use ironrdp_egfx::decode::{DecodedFrame, H264Decoder};
use ironrdp_egfx::decode::{DecoderError, DecoderResult, H264YuvDecoder, Yuv420View, YuvDecoderFactory};
use ironrdp_egfx::pdu::{
    Avc420BitmapStream, Avc444BitmapStream, CapabilitiesAdvertisePdu, CapabilitiesConfirmPdu, CapabilitiesV8Flags,
    CapabilitiesV81Flags, CapabilitiesV107Flags, CapabilitySet, CapabilityVersion, Codec1Type, CreateSurfacePdu,
    DeleteSurfacePdu, Encoding, GfxPdu, PixelFormat, QuantQuality, ResetGraphicsPdu, WireToSurface1Pdu,
};
use ironrdp_egfx::yuv444::{ChromaLayout, Yuv444Planes, color, split};
use ironrdp_graphics::zgfx::wrap_uncompressed;
use ironrdp_pdu::geometry::ExclusiveRectangle;

// ============================================================================
// Recorder handler
// ============================================================================

/// Records every `BitmapUpdate` (with its pixel data, unlike `client.rs`'s
/// `TestHandler`, which only keeps `(surface_id, codec_id)`).
struct Recorder {
    painted: Vec<PaintedRect>,
    caps_confirmed: bool,
}

struct PaintedRect {
    surface_id: u16,
    codec_id: Codec1Type,
    rect: ExclusiveRectangle,
    data: Vec<u8>,
}

impl Recorder {
    fn new() -> Self {
        Self {
            painted: Vec::new(),
            caps_confirmed: false,
        }
    }
}

/// A handler whose `capabilities()` is injected (for the `start()` filtering
/// tests) and whose recorded state is shared back to the test through `Arc<Mutex<_>>`,
/// since `GraphicsPipelineClient` owns the handler as an opaque `Box<dyn ...>`.
struct RecorderHandler {
    recorder: Arc<Mutex<Recorder>>,
    caps: Vec<CapabilitySet>,
}

impl GraphicsPipelineHandler for RecorderHandler {
    fn capabilities(&self) -> Vec<CapabilitySet> {
        self.caps.clone()
    }

    fn on_capabilities_confirmed(&mut self, _caps: &CapabilitySet) {
        self.recorder.lock().expect("lock poisoned").caps_confirmed = true;
    }

    fn on_surface_created(&mut self, _surface: &Surface) {}
    fn on_surface_deleted(&mut self, _surface_id: u16) {}

    fn on_bitmap_updated(&mut self, update: &BitmapUpdate) {
        self.recorder.lock().expect("lock poisoned").painted.push(PaintedRect {
            surface_id: update.surface_id,
            codec_id: update.codec_id,
            rect: update.destination_rectangle.clone(),
            data: update.data.clone(),
        });
    }
}

fn default_caps() -> Vec<CapabilitySet> {
    vec![
        CapabilitySet::V8_1 {
            flags: CapabilitiesV81Flags::empty(),
        },
        CapabilitySet::V8 {
            flags: CapabilitiesV8Flags::empty(),
        },
    ]
}

// ============================================================================
// Mock H264YuvDecoder
// ============================================================================

#[derive(Default)]
struct DecoderCounters {
    decode_calls: u32,
    reset_calls: u32,
}

/// A scripted `H264YuvDecoder`: returns pre-built frames in order, optionally
/// fails on specific (1-indexed) `decode_yuv` calls, and shares its call
/// counters with the test through `counters`.
struct MockYuvDecoder {
    frames: VecDeque<split::OwnedYuv420>,
    fail_on_calls: HashSet<u32>,
    counters: Arc<Mutex<DecoderCounters>>,
    current: Option<split::OwnedYuv420>,
}

impl MockYuvDecoder {
    fn new(frames: Vec<split::OwnedYuv420>, fail_on_calls: &[u32], counters: Arc<Mutex<DecoderCounters>>) -> Self {
        Self {
            frames: frames.into(),
            fail_on_calls: fail_on_calls.iter().copied().collect(),
            counters,
            current: None,
        }
    }
}

impl H264YuvDecoder for MockYuvDecoder {
    fn decode_yuv(&mut self, _data: &[u8]) -> DecoderResult<Yuv420View<'_>> {
        let call_n = {
            let mut counters = self.counters.lock().expect("lock poisoned");
            counters.decode_calls += 1;
            counters.decode_calls
        };
        if self.fail_on_calls.contains(&call_n) {
            return Err(DecoderError::msg("mock decode failure"));
        }
        let frame = self.frames.pop_front().expect("mock ran out of scripted frames");
        self.current = Some(frame);
        Ok(self.current.as_ref().expect("just set").view())
    }

    fn reset(&mut self) {
        self.counters.lock().expect("lock poisoned").reset_calls += 1;
    }
}

/// A factory that hands out pre-built decoders from a queue, one per call, and
/// returns `None` once the queue is empty (models "factory returned None" as
/// well as counting total factory calls when combined with `factory_calls`).
fn queued_factory(decoders: Vec<Box<dyn H264YuvDecoder>>, factory_calls: Arc<Mutex<u32>>) -> YuvDecoderFactory {
    let queue = Arc::new(Mutex::new(VecDeque::from(decoders)));
    Box::new(move || {
        *factory_calls.lock().expect("lock poisoned") += 1;
        queue.lock().expect("lock poisoned").pop_front()
    })
}

// ============================================================================
// Synthetic source images
// ============================================================================

fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

/// Two-level (40/200) pseudo-random chroma over an `((x+y)*2) & 0xFF` luma
/// gradient — round-trips exactly through the main+aux split (see
/// `yuv444.rs`'s own tests), so the expected RGBA is `color::yuv_to_rgb`
/// applied directly to this source. Returns the source planes alongside the
/// raw `(y, u, v)` vectors used to build them, for `direct_rgba`.
fn two_level_source(width: usize, height: usize, seed: u64) -> (Yuv444Planes, Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; width * height];
    let mut u = vec![0u8; width * height];
    let mut v = vec![0u8; width * height];
    let mut state = seed;
    for py in 0..height {
        for px in 0..width {
            let idx = py * width + px;
            y[idx] = u8::try_from(((px + py) * 2) & 0xFF).unwrap_or(0);
            // Bit 0 of this LCG strictly alternates every step (odd multiplier, odd
            // increment), which would make an `& 1` extraction constant across all
            // pixels here (two calls per pixel, so both parities are call-count
            // invariants) instead of pseudo-random. Use a high bit instead.
            u[idx] = if (lcg_next(&mut state) >> 32) & 1 == 0 { 40 } else { 200 };
            v[idx] = if (lcg_next(&mut state) >> 32) & 1 == 0 { 40 } else { 200 };
        }
    }
    let planes = Yuv444Planes::from_planes(width, height, y.clone(), u.clone(), v.clone());
    (planes, y, u, v)
}

fn direct_rgba(width: usize, height: usize, y: &[u8], u: &[u8], v: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(width * height * 4);
    for idx in 0..width * height {
        let rgb = color::yuv_to_rgb(y[idx], u[idx], v[idx]);
        out.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 0xFF]);
    }
    out
}

fn as_u16(v: usize) -> u16 {
    u16::try_from(v).expect("test dimensions fit in u16")
}

fn full_rect(w: usize, h: usize) -> ExclusiveRectangle {
    ExclusiveRectangle {
        left: 0,
        top: 0,
        right: u16::try_from(w).unwrap(),
        bottom: u16::try_from(h).unwrap(),
    }
}

// ============================================================================
// PDU encoding helpers
// ============================================================================

fn encode_pdu<T: Encode>(pdu: &T) -> Vec<u8> {
    let mut buf = vec![0u8; pdu.size()];
    let mut cursor = WriteCursor::new(&mut buf);
    pdu.encode(&mut cursor).expect("encode failed");
    buf
}

fn encode_for_process(pdu: &GfxPdu) -> Vec<u8> {
    wrap_uncompressed(&encode_pdu(pdu))
}

fn decode_caps_from_message(msg: &ironrdp_dvc::DvcMessage) -> CapabilitiesAdvertisePdu {
    let encoded = encode_vec(msg.as_ref()).expect("encode should succeed");
    let mut cursor = ReadCursor::new(&encoded);
    match GfxPdu::decode(&mut cursor).expect("decode should succeed") {
        GfxPdu::CapabilitiesAdvertise(caps) => caps,
        other => panic!("expected CapabilitiesAdvertise, got {other:?}"),
    }
}

fn avc420_stream(rectangles: Vec<ExclusiveRectangle>, data: &'static [u8]) -> Avc420BitmapStream<'static> {
    let quant_qual_vals = rectangles
        .iter()
        .map(|_| QuantQuality {
            quantization_parameter: 22,
            progressive: false,
            quality: 100,
        })
        .collect();
    Avc420BitmapStream {
        rectangles,
        quant_qual_vals,
        data,
    }
}

fn wire_to_surface1(
    surface_id: u16,
    codec_id: Codec1Type,
    dest_rect: ExclusiveRectangle,
    bitmap_data: Vec<u8>,
) -> GfxPdu {
    GfxPdu::WireToSurface1(WireToSurface1Pdu {
        surface_id,
        codec_id,
        pixel_format: PixelFormat::XRgb,
        destination_rectangle: dest_rect,
        bitmap_data,
    })
}

/// Encode an `Avc444BitmapStream` as the `bitmap_data` payload of a
/// `WireToSurface1` PDU and run it through `process()`.
fn send_avc444(
    client: &mut GraphicsPipelineClient,
    surface_id: u16,
    codec_id: Codec1Type,
    dest_rect: ExclusiveRectangle,
    encoding: Encoding,
    stream1: Avc420BitmapStream<'static>,
    stream2: Option<Avc420BitmapStream<'static>>,
) -> ironrdp_pdu::PduResult<Vec<ironrdp_dvc::DvcMessage>> {
    let stream = Avc444BitmapStream {
        encoding,
        stream1,
        stream2,
    };
    let bitmap_data = encode_pdu(&stream);
    let pdu = wire_to_surface1(surface_id, codec_id, dest_rect, bitmap_data);
    client.process(0, &encode_for_process(&pdu))
}

fn setup_active_client(
    factory: Option<YuvDecoderFactory>,
    surface_id: u16,
    width: u16,
    height: u16,
) -> (GraphicsPipelineClient, Arc<Mutex<Recorder>>) {
    let recorder = Arc::new(Mutex::new(Recorder::new()));
    let handler = RecorderHandler {
        recorder: Arc::clone(&recorder),
        caps: default_caps(),
    };
    let mut client = GraphicsPipelineClient::new(Box::new(handler), None);
    if let Some(factory) = factory {
        client = client.with_avc444_decoders(factory);
    }

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

    (client, recorder)
}

/// Build the main+aux `MockYuvDecoder`s (each scripted with `frames`) and a
/// factory yielding them (main first, then aux), plus the counters the test
/// can inspect afterward. Tests that also need the factory's own call count
/// (test 8) build the queue directly instead of going through this helper.
struct MockDecoders {
    factory: YuvDecoderFactory,
    main_counters: Arc<Mutex<DecoderCounters>>,
    aux_counters: Arc<Mutex<DecoderCounters>>,
}

fn mock_decoders(
    main_frames: Vec<split::OwnedYuv420>,
    aux_frames: Vec<split::OwnedYuv420>,
    main_fail_on: &[u32],
    aux_fail_on: &[u32],
) -> MockDecoders {
    let main_counters = Arc::new(Mutex::new(DecoderCounters::default()));
    let aux_counters = Arc::new(Mutex::new(DecoderCounters::default()));
    let factory_calls = Arc::new(Mutex::new(0u32));

    let main: Box<dyn H264YuvDecoder> = Box::new(MockYuvDecoder::new(
        main_frames,
        main_fail_on,
        Arc::clone(&main_counters),
    ));
    let aux: Box<dyn H264YuvDecoder> =
        Box::new(MockYuvDecoder::new(aux_frames, aux_fail_on, Arc::clone(&aux_counters)));

    MockDecoders {
        factory: queued_factory(vec![main, aux], factory_calls),
        main_counters,
        aux_counters,
    }
}

fn last_painted(recorder: &Arc<Mutex<Recorder>>, surface_id: u16) -> (Codec1Type, ExclusiveRectangle, Vec<u8>) {
    let recorder = recorder.lock().expect("lock poisoned");
    let entry = recorder
        .painted
        .iter()
        .rev()
        .find(|p| p.surface_id == surface_id)
        .expect("expected a painted update for this surface");
    (entry.codec_id, entry.rect.clone(), entry.data.clone())
}

fn painted_count(recorder: &Arc<Mutex<Recorder>>, surface_id: u16) -> usize {
    recorder
        .lock()
        .expect("lock poisoned")
        .painted
        .iter()
        .filter(|p| p.surface_id == surface_id)
        .count()
}

// ============================================================================
// 1. luma+chroma paints exactly, codec_id == Avc444
// ============================================================================

#[test]
fn luma_and_chroma_paints_exact_rgba_via_process() {
    let (width, height) = (48, 32);
    let (src, y, u, v) = two_level_source(width, height, 0x1234_5678_9abc_def0);
    let main_frame = split::main_view(&src);
    let aux_frame = split::aux_view(&src, ChromaLayout::V1);

    let decoders = mock_decoders(vec![main_frame], vec![aux_frame], &[], &[]);
    let (mut client, recorder) = setup_active_client(Some(decoders.factory), 1, as_u16(width), as_u16(height));

    let rect = full_rect(width, height);
    let result = send_avc444(
        &mut client,
        1,
        Codec1Type::Avc444,
        rect.clone(),
        Encoding::LUMA_AND_CHROMA,
        avc420_stream(vec![rect.clone()], &[0x00, 0x00, 0x00, 0x01, 0x67]),
        Some(avc420_stream(vec![rect], &[0x00, 0x00, 0x00, 0x01, 0x68])),
    );
    result.expect("LC=0 should succeed");

    let (codec_id, painted_rect, data) = last_painted(&recorder, 1);
    assert_eq!(codec_id, Codec1Type::Avc444);
    assert_eq!(painted_rect, full_rect(width, height));
    assert_eq!(data, direct_rgba(width, height, &y, &u, &v));
}

// ============================================================================
// 2. AVC444v2 uses the V2 layout
// ============================================================================

#[test]
fn avc444v2_uses_the_v2_layout() {
    let (width, height) = (48, 32);
    let (src, y, u, v) = two_level_source(width, height, 0xdead_beef_1234_5678);

    // Correct pairing: v2 codec + v2 aux layout reconstructs exactly.
    let main_frame = split::main_view(&src);
    let aux_frame_v2 = split::aux_view(&src, ChromaLayout::V2);
    let decoders = mock_decoders(vec![main_frame], vec![aux_frame_v2], &[], &[]);
    let (mut client, recorder) = setup_active_client(Some(decoders.factory), 1, as_u16(width), as_u16(height));
    let rect = full_rect(width, height);
    send_avc444(
        &mut client,
        1,
        Codec1Type::Avc444v2,
        rect.clone(),
        Encoding::LUMA_AND_CHROMA,
        avc420_stream(vec![rect.clone()], &[0x00, 0x00, 0x00, 0x01, 0x67]),
        Some(avc420_stream(vec![rect], &[0x00, 0x00, 0x00, 0x01, 0x68])),
    )
    .expect("LC=0 with v2 layout should succeed");
    let (_, _, data) = last_painted(&recorder, 1);
    assert_eq!(
        data,
        direct_rgba(width, height, &y, &u, &v),
        "v2 codec + v2 aux must round-trip exactly"
    );

    // Mismatched pairing: v2 codec + v1-shaped aux data must NOT reconstruct exactly,
    // proving the client picks the layout from `codec_id` rather than hardcoding one.
    let main_frame2 = split::main_view(&src);
    let aux_frame_v1 = split::aux_view(&src, ChromaLayout::V1);
    let decoders2 = mock_decoders(vec![main_frame2], vec![aux_frame_v1], &[], &[]);
    let (mut client2, recorder2) = setup_active_client(Some(decoders2.factory), 2, as_u16(width), as_u16(height));
    send_avc444(
        &mut client2,
        2,
        Codec1Type::Avc444v2,
        full_rect(width, height),
        Encoding::LUMA_AND_CHROMA,
        avc420_stream(vec![full_rect(width, height)], &[0x00, 0x00, 0x00, 0x01, 0x67]),
        Some(avc420_stream(
            vec![full_rect(width, height)],
            &[0x00, 0x00, 0x00, 0x01, 0x68],
        )),
    )
    .expect("mismatched layout still decodes without error");
    let (_, _, mismatched_data) = last_painted(&recorder2, 2);
    assert_ne!(
        mismatched_data,
        direct_rgba(width, height, &y, &u, &v),
        "v1-shaped aux data read as v2 must not accidentally round-trip"
    );
}

// ============================================================================
// 3. LC=1 then LC=2 sequencing
// ============================================================================

#[test]
fn lc1_then_lc2_composes_to_the_exact_4_4_4_picture() {
    let (width, height) = (48, 32);
    let (src, y, u, v) = two_level_source(width, height, 0x0011_2233_4455_6677);
    let main_frame = split::main_view(&src);
    let aux_frame = split::aux_view(&src, ChromaLayout::V1);

    let decoders = mock_decoders(vec![main_frame], vec![aux_frame], &[], &[]);
    let (mut client, recorder) = setup_active_client(Some(decoders.factory), 1, as_u16(width), as_u16(height));
    let rect = full_rect(width, height);

    // LC=1: main only.
    send_avc444(
        &mut client,
        1,
        Codec1Type::Avc444,
        rect.clone(),
        Encoding::LUMA,
        avc420_stream(vec![rect.clone()], &[0x00, 0x00, 0x00, 0x01, 0x67]),
        None,
    )
    .expect("LC=1 should succeed");
    let (_, _, luma_only_data) = last_painted(&recorder, 1);
    assert_ne!(
        luma_only_data,
        direct_rgba(width, height, &y, &u, &v),
        "LC=1 alone only has the replicated 4:2:0 chroma, not the exact 4:4:4 picture"
    );

    // LC=2: aux only, combined with the luma already painted.
    send_avc444(
        &mut client,
        1,
        Codec1Type::Avc444,
        rect.clone(),
        Encoding::CHROMA,
        avc420_stream(vec![rect], &[0x00, 0x00, 0x00, 0x01, 0x68]),
        None,
    )
    .expect("LC=2 should succeed");
    let (_, _, combined_data) = last_painted(&recorder, 1);
    assert_eq!(
        combined_data,
        direct_rgba(width, height, &y, &u, &v),
        "LC=2 after LC=1 must produce the exact 4:4:4 picture"
    );
}

// ============================================================================
// 4. chroma before any luma
// ============================================================================

#[test]
fn chroma_before_any_luma_is_skipped_and_resets_both_decoders() {
    let (width, height) = (16, 16);
    let (src, _y, _u, _v) = two_level_source(width, height, 1);
    let aux_frame = split::aux_view(&src, ChromaLayout::V1);
    let decoders = mock_decoders(vec![], vec![aux_frame], &[], &[]);
    let (mut client, recorder) = setup_active_client(Some(decoders.factory), 1, as_u16(width), as_u16(height));

    let rect = full_rect(width, height);
    send_avc444(
        &mut client,
        1,
        Codec1Type::Avc444,
        rect.clone(),
        Encoding::CHROMA,
        avc420_stream(vec![rect], &[0x00, 0x00, 0x00, 0x01, 0x68]),
        None,
    )
    .expect("chroma-before-luma must not error");

    assert_eq!(painted_count(&recorder, 1), 0, "nothing should be painted");
    assert_eq!(decoders.main_counters.lock().unwrap().reset_calls, 1);
    assert_eq!(decoders.aux_counters.lock().unwrap().reset_calls, 1);
    assert_eq!(
        decoders.aux_counters.lock().unwrap().decode_calls,
        0,
        "aux.decode_yuv must not run"
    );
}

// ============================================================================
// 5. decode failures
// ============================================================================

#[test]
fn a_failing_main_or_aux_decoder_paints_nothing_and_resets_both() {
    let (width, height) = (16, 16);
    let (src, _y, _u, _v) = two_level_source(width, height, 2);
    let rect = full_rect(width, height);

    // Main decoder fails.
    let main_frame = split::main_view(&src);
    let aux_frame = split::aux_view(&src, ChromaLayout::V1);
    let decoders = mock_decoders(vec![main_frame], vec![aux_frame], &[1], &[]);
    let (mut client, recorder) = setup_active_client(Some(decoders.factory), 1, as_u16(width), as_u16(height));
    send_avc444(
        &mut client,
        1,
        Codec1Type::Avc444,
        rect.clone(),
        Encoding::LUMA_AND_CHROMA,
        avc420_stream(vec![rect.clone()], &[0x00, 0x00, 0x00, 0x01, 0x67]),
        Some(avc420_stream(vec![rect.clone()], &[0x00, 0x00, 0x00, 0x01, 0x68])),
    )
    .expect("a decode failure must not error the PDU");
    assert_eq!(painted_count(&recorder, 1), 0);
    assert_eq!(decoders.main_counters.lock().unwrap().reset_calls, 1);
    assert_eq!(decoders.aux_counters.lock().unwrap().reset_calls, 1);
    // Both streams decode before any paint, so aux.decode_yuv still ran once.
    assert_eq!(decoders.aux_counters.lock().unwrap().decode_calls, 1);

    // Aux decoder fails: luma must not be painted either.
    let main_frame2 = split::main_view(&src);
    let aux_frame2 = split::aux_view(&src, ChromaLayout::V1);
    let decoders2 = mock_decoders(vec![main_frame2], vec![aux_frame2], &[], &[1]);
    let (mut client2, recorder2) = setup_active_client(Some(decoders2.factory), 2, as_u16(width), as_u16(height));
    send_avc444(
        &mut client2,
        2,
        Codec1Type::Avc444,
        rect.clone(),
        Encoding::LUMA_AND_CHROMA,
        avc420_stream(vec![rect.clone()], &[0x00, 0x00, 0x00, 0x01, 0x67]),
        Some(avc420_stream(vec![rect], &[0x00, 0x00, 0x00, 0x01, 0x68])),
    )
    .expect("a decode failure must not error the PDU");
    assert_eq!(
        painted_count(&recorder2, 2),
        0,
        "luma must not be painted when aux fails"
    );
    assert_eq!(decoders2.main_counters.lock().unwrap().reset_calls, 1);
    assert_eq!(decoders2.aux_counters.lock().unwrap().reset_calls, 1);
}

// ============================================================================
// 6. malformed stream info
// ============================================================================

#[test]
fn malformed_avc444_stream_info_is_skipped_without_a_panic() {
    let (mut client, recorder) = setup_active_client(None, 1, 16, 16);
    let rect = full_rect(16, 16);

    // LC=3 (reserved encoding value): `stream_info` bits 30..32 = 3, len = 0.
    let stream_info: u32 = 3u32 << 30;
    let pdu = wire_to_surface1(1, Codec1Type::Avc444, rect.clone(), stream_info.to_le_bytes().to_vec());
    client
        .process(0, &encode_for_process(&pdu))
        .expect("LC=3 must be skipped, not errored");
    assert_eq!(painted_count(&recorder, 1), 0);

    // Declared stream1 length past the end of the buffer.
    let bogus_len: u32 = 0xFF_FF;
    let stream_info2: u32 = bogus_len; // encoding bits 30..32 = 0 (LUMA_AND_CHROMA), len = 0xFFFF
    let mut bitmap_data = stream_info2.to_le_bytes().to_vec();
    bitmap_data.extend_from_slice(&[0u8; 4]); // a few trailing bytes, far short of bogus_len
    let pdu2 = wire_to_surface1(1, Codec1Type::Avc444, rect, bitmap_data);
    client
        .process(0, &encode_for_process(&pdu2))
        .expect("an over-length stream must be skipped, not errored");
    assert_eq!(painted_count(&recorder, 1), 0);
}

// ============================================================================
// 7. unknown surface stays fatal
// ============================================================================

#[test]
fn avc444_to_an_unknown_surface_is_an_error() {
    let (mut client, _recorder) = setup_active_client(None, 1, 16, 16);
    let pdu = wire_to_surface1(999, Codec1Type::Avc444, full_rect(16, 16), vec![0u8; 4]);
    let result = client.process(0, &encode_for_process(&pdu));
    assert!(
        result.is_err(),
        "unknown surface must stay fatal, matching decode_avc420"
    );
}

// ============================================================================
// 8. delete + recreate re-invokes the factory
// ============================================================================

#[test]
fn delete_then_recreate_surface_invokes_the_factory_again() {
    let (width, height) = (16, 16);
    let (src, y, u, v) = two_level_source(width, height, 3);
    let rect = full_rect(width, height);

    let main1 = split::main_view(&src);
    let aux1 = split::aux_view(&src, ChromaLayout::V1);
    let main2 = split::main_view(&src);
    let aux2 = split::aux_view(&src, ChromaLayout::V1);

    let main_counters = Arc::new(Mutex::new(DecoderCounters::default()));
    let aux_counters = Arc::new(Mutex::new(DecoderCounters::default()));
    let factory_calls = Arc::new(Mutex::new(0u32));
    let decoders: Vec<Box<dyn H264YuvDecoder>> = vec![
        Box::new(MockYuvDecoder::new(vec![main1], &[], Arc::clone(&main_counters))),
        Box::new(MockYuvDecoder::new(vec![aux1], &[], Arc::clone(&aux_counters))),
        Box::new(MockYuvDecoder::new(vec![main2], &[], Arc::clone(&main_counters))),
        Box::new(MockYuvDecoder::new(vec![aux2], &[], Arc::clone(&aux_counters))),
    ];
    let factory = queued_factory(decoders, Arc::clone(&factory_calls));
    let (mut client, recorder) = setup_active_client(Some(factory), 1, as_u16(width), as_u16(height));

    send_avc444(
        &mut client,
        1,
        Codec1Type::Avc444,
        rect.clone(),
        Encoding::LUMA_AND_CHROMA,
        avc420_stream(vec![rect.clone()], &[0x00, 0x00, 0x00, 0x01, 0x67]),
        Some(avc420_stream(vec![rect.clone()], &[0x00, 0x00, 0x00, 0x01, 0x68])),
    )
    .expect("first surface decode should succeed");
    assert_eq!(*factory_calls.lock().unwrap(), 2);

    let delete = GfxPdu::DeleteSurface(DeleteSurfacePdu { surface_id: 1 });
    client
        .process(0, &encode_for_process(&delete))
        .expect("delete should succeed");

    let create = GfxPdu::CreateSurface(CreateSurfacePdu {
        surface_id: 1,
        width: as_u16(width),
        height: as_u16(height),
        pixel_format: PixelFormat::XRgb,
    });
    client
        .process(0, &encode_for_process(&create))
        .expect("recreate should succeed");

    send_avc444(
        &mut client,
        1,
        Codec1Type::Avc444,
        rect.clone(),
        Encoding::LUMA_AND_CHROMA,
        avc420_stream(vec![rect.clone()], &[0x00, 0x00, 0x00, 0x01, 0x67]),
        Some(avc420_stream(vec![rect], &[0x00, 0x00, 0x00, 0x01, 0x68])),
    )
    .expect("second surface decode should succeed");

    assert_eq!(
        *factory_calls.lock().unwrap(),
        4,
        "the factory must be re-invoked for the recreated surface"
    );
    let (_, _, data) = last_painted(&recorder, 1);
    assert_eq!(data, direct_rgba(width, height, &y, &u, &v));
}

// ============================================================================
// 9. ResetGraphics resets both decoders, planes survive
// ============================================================================

#[test]
fn reset_graphics_resets_both_decoders_but_keeps_planes() {
    let (width, height) = (16, 16);
    let (src, y, u, v) = two_level_source(width, height, 4);
    let rect = full_rect(width, height);

    let main_frame = split::main_view(&src);
    let aux_frame1 = split::aux_view(&src, ChromaLayout::V1);
    let aux_frame2 = split::aux_view(&src, ChromaLayout::V1);
    let decoders = mock_decoders(vec![main_frame], vec![aux_frame1, aux_frame2], &[], &[]);
    let main_counters = Arc::clone(&decoders.main_counters);
    let aux_counters = Arc::clone(&decoders.aux_counters);
    let (mut client, recorder) = setup_active_client(Some(decoders.factory), 1, as_u16(width), as_u16(height));

    // LC=1: establish `has_luma` and the planes' size.
    send_avc444(
        &mut client,
        1,
        Codec1Type::Avc444,
        rect.clone(),
        Encoding::LUMA,
        avc420_stream(vec![rect.clone()], &[0x00, 0x00, 0x00, 0x01, 0x67]),
        None,
    )
    .expect("LC=1 should succeed");

    let reset = GfxPdu::ResetGraphics(ResetGraphicsPdu {
        width: u32::from(as_u16(width)),
        height: u32::from(as_u16(height)),
        monitors: vec![],
    });
    client
        .process(0, &encode_for_process(&reset))
        .expect("reset should succeed");
    assert_eq!(main_counters.lock().unwrap().reset_calls, 1);
    assert_eq!(aux_counters.lock().unwrap().reset_calls, 1);

    // LC=2 after the reset: `has_luma` (part of the surface, not the decoder) must
    // have survived, so this still paints.
    send_avc444(
        &mut client,
        1,
        Codec1Type::Avc444,
        rect.clone(),
        Encoding::CHROMA,
        avc420_stream(vec![rect], &[0x00, 0x00, 0x00, 0x01, 0x68]),
        None,
    )
    .expect("LC=2 after reset should succeed");

    let (_, _, data) = last_painted(&recorder, 1);
    assert_eq!(data, direct_rgba(width, height, &y, &u, &v));
}

// ============================================================================
// 10. start() capability filtering
// ============================================================================

struct MockH264Decoder;
impl H264Decoder for MockH264Decoder {
    fn decode(&mut self, _data: &[u8]) -> DecoderResult<DecodedFrame> {
        Err(DecoderError::msg("not used in this test"))
    }
}

fn caps_with_v10() -> Vec<CapabilitySet> {
    vec![
        CapabilitySet::V10_7 {
            flags: CapabilitiesV107Flags::empty(),
        },
        CapabilitySet::V8_1 {
            flags: CapabilitiesV81Flags::AVC420_ENABLED,
        },
        CapabilitySet::V8 {
            flags: CapabilitiesV8Flags::empty(),
        },
    ]
}

fn advertised_versions(client: &mut GraphicsPipelineClient) -> Vec<CapabilityVersion> {
    let messages = client.start(0).expect("start should succeed");
    assert_eq!(messages.len(), 1);
    decode_caps_from_message(&messages[0])
        .0
        .iter()
        .map(|c| c.version)
        .collect()
}

#[test]
fn start_filters_v10_sets_independently_of_v8_1() {
    // With an H264 decoder but no AVC444 factory: V10.x dropped, V8_1 kept.
    let recorder = Arc::new(Mutex::new(Recorder::new()));
    let handler = RecorderHandler {
        recorder: Arc::clone(&recorder),
        caps: caps_with_v10(),
    };
    let mut client = GraphicsPipelineClient::new(Box::new(handler), Some(Box::new(MockH264Decoder)));
    let versions = advertised_versions(&mut client);
    assert!(!versions.contains(&CapabilityVersion::V10_7));
    assert!(versions.contains(&CapabilityVersion::V8_1));

    // With both an H264 decoder and an AVC444 factory: V10.x kept.
    let recorder2 = Arc::new(Mutex::new(Recorder::new()));
    let handler2 = RecorderHandler {
        recorder: Arc::clone(&recorder2),
        caps: caps_with_v10(),
    };
    let factory_calls = Arc::new(Mutex::new(0u32));
    let factory = queued_factory(vec![], Arc::clone(&factory_calls));
    let mut client2 =
        GraphicsPipelineClient::new(Box::new(handler2), Some(Box::new(MockH264Decoder))).with_avc444_decoders(factory);
    let versions2 = advertised_versions(&mut client2);
    assert!(versions2.contains(&CapabilityVersion::V10_7));

    // With an AVC444 factory but no H264 decoder: V10.x still dropped (implies AVC420).
    let recorder3 = Arc::new(Mutex::new(Recorder::new()));
    let handler3 = RecorderHandler {
        recorder: Arc::clone(&recorder3),
        caps: caps_with_v10(),
    };
    let factory_calls3 = Arc::new(Mutex::new(0u32));
    let factory3 = queued_factory(vec![], Arc::clone(&factory_calls3));
    let mut client3 = GraphicsPipelineClient::new(Box::new(handler3), None).with_avc444_decoders(factory3);
    let versions3 = advertised_versions(&mut client3);
    assert!(!versions3.contains(&CapabilityVersion::V10_7));
}

// ============================================================================
// 11. region rects beyond the surface / frame are clipped, no panic
// ============================================================================

#[test]
fn region_rects_beyond_surface_and_frame_are_clipped_without_panicking() {
    let (surface_w, surface_h) = (16u16, 16u16);
    let (frame_w, frame_h) = (32, 32);
    let (src, _y, _u, _v) = two_level_source(frame_w, frame_h, 5);
    let main_frame = split::main_view(&src);
    let aux_frame = split::aux_view(&src, ChromaLayout::V1);
    let decoders = mock_decoders(vec![main_frame], vec![aux_frame], &[], &[]);
    let (mut client, recorder) = setup_active_client(Some(decoders.factory), 1, surface_w, surface_h);

    // Region rect covers the whole (macroblock-aligned) frame, well beyond the
    // 16x16 surface; dest_rect clips further, to an 8x8 corner.
    let oversized_region = ExclusiveRectangle {
        left: 0,
        top: 0,
        right: u16::try_from(frame_w).unwrap(),
        bottom: u16::try_from(frame_h).unwrap(),
    };
    let dest_rect = ExclusiveRectangle {
        left: 0,
        top: 0,
        right: 8,
        bottom: 8,
    };
    send_avc444(
        &mut client,
        1,
        Codec1Type::Avc444,
        dest_rect,
        Encoding::LUMA_AND_CHROMA,
        avc420_stream(vec![oversized_region.clone()], &[0x00, 0x00, 0x00, 0x01, 0x67]),
        Some(avc420_stream(vec![oversized_region], &[0x00, 0x00, 0x00, 0x01, 0x68])),
    )
    .expect("oversized region rects must be clipped, not error or panic");

    let (_, painted_rect, data) = last_painted(&recorder, 1);
    assert!(
        painted_rect.right <= 8 && painted_rect.bottom <= 8,
        "must be clipped to dest_rect"
    );
    assert!(
        painted_rect.right <= surface_w && painted_rect.bottom <= surface_h,
        "must be clipped to the surface"
    );
    let w = usize::from(painted_rect.right - painted_rect.left);
    let h = usize::from(painted_rect.bottom - painted_rect.top);
    assert_eq!(data.len(), w * h * 4);
}

// ============================================================================
// 12. factory returning None
// ============================================================================

#[test]
fn factory_returning_none_skips_without_error() {
    let factory_calls = Arc::new(Mutex::new(0u32));
    let factory = queued_factory(vec![], Arc::clone(&factory_calls));
    let (mut client, recorder) = setup_active_client(Some(factory), 1, 16, 16);
    let rect = full_rect(16, 16);

    for _ in 0..2 {
        send_avc444(
            &mut client,
            1,
            Codec1Type::Avc444,
            rect.clone(),
            Encoding::LUMA_AND_CHROMA,
            avc420_stream(vec![rect.clone()], &[0x00, 0x00, 0x00, 0x01, 0x67]),
            Some(avc420_stream(vec![rect.clone()], &[0x00, 0x00, 0x00, 0x01, 0x68])),
        )
        .expect("a missing decoder must be skipped, not errored");
    }

    assert_eq!(painted_count(&recorder, 1), 0);
}

// ============================================================================
// Fixup: a plane resize must not let a stale `has_luma` survive it
// ============================================================================

#[test]
fn a_frame_size_change_resets_the_luma_state() {
    // Positive: LC=1 at 32x32, then LC=1 at 48x48 (a genuine resize with a
    // non-empty luma rect list), then LC=2 must still paint — the resize's own
    // luma rects re-earn `has_luma` within that same PDU.
    let (src32, _y32, _u32, _v32) = two_level_source(32, 32, 10);
    let (src48, y48, u48, v48) = two_level_source(48, 48, 11);
    let main_frame_32 = split::main_view(&src32);
    let main_frame_48 = split::main_view(&src48);
    let aux_frame_48 = split::aux_view(&src48, ChromaLayout::V1);

    let main_counters = Arc::new(Mutex::new(DecoderCounters::default()));
    let aux_counters = Arc::new(Mutex::new(DecoderCounters::default()));
    let factory_calls = Arc::new(Mutex::new(0u32));
    let main: Box<dyn H264YuvDecoder> = Box::new(MockYuvDecoder::new(
        vec![main_frame_32, main_frame_48],
        &[],
        Arc::clone(&main_counters),
    ));
    let aux: Box<dyn H264YuvDecoder> =
        Box::new(MockYuvDecoder::new(vec![aux_frame_48], &[], Arc::clone(&aux_counters)));
    let factory = queued_factory(vec![main, aux], factory_calls);
    let (mut client, recorder) = setup_active_client(Some(factory), 1, 48, 48);

    send_avc444(
        &mut client,
        1,
        Codec1Type::Avc444,
        full_rect(32, 32),
        Encoding::LUMA,
        avc420_stream(vec![full_rect(32, 32)], &[0x00, 0x00, 0x00, 0x01, 0x67]),
        None,
    )
    .expect("first LC=1 (32x32) should succeed");

    send_avc444(
        &mut client,
        1,
        Codec1Type::Avc444,
        full_rect(48, 48),
        Encoding::LUMA,
        avc420_stream(vec![full_rect(48, 48)], &[0x00, 0x00, 0x00, 0x01, 0x67]),
        None,
    )
    .expect("second LC=1 (48x48, a resize) should succeed");

    send_avc444(
        &mut client,
        1,
        Codec1Type::Avc444,
        full_rect(48, 48),
        Encoding::CHROMA,
        avc420_stream(vec![full_rect(48, 48)], &[0x00, 0x00, 0x00, 0x01, 0x68]),
        None,
    )
    .expect("LC=2 after the resize should still paint");

    let (_, _, data) = last_painted(&recorder, 1);
    assert_eq!(data, direct_rgba(48, 48, &y48, &u48, &v48));

    // Negative: a resize whose luma PDU carries an EMPTY rect list must not
    // re-earn `has_luma` — a following LC=2 must still hit the
    // chroma-before-luma guard and reset both decoders.
    let (src16, _, _, _) = two_level_source(16, 16, 12);
    let main_frame_16 = split::main_view(&src16);
    let decoders2 = mock_decoders(vec![main_frame_16], vec![], &[], &[]);
    let main_counters2 = Arc::clone(&decoders2.main_counters);
    let aux_counters2 = Arc::clone(&decoders2.aux_counters);
    let (mut client2, recorder2) = setup_active_client(Some(decoders2.factory), 2, 16, 16);

    send_avc444(
        &mut client2,
        2,
        Codec1Type::Avc444,
        full_rect(16, 16),
        Encoding::LUMA,
        avc420_stream(vec![], &[0x00, 0x00, 0x00, 0x01, 0x67]),
        None,
    )
    .expect("LC=1 with an empty rect list should still succeed");

    send_avc444(
        &mut client2,
        2,
        Codec1Type::Avc444,
        full_rect(16, 16),
        Encoding::CHROMA,
        avc420_stream(vec![full_rect(16, 16)], &[0x00, 0x00, 0x00, 0x01, 0x68]),
        None,
    )
    .expect("LC=2 must be skipped, not errored");

    assert_eq!(painted_count(&recorder2, 2), 0, "nothing should be painted");
    assert_eq!(main_counters2.lock().unwrap().reset_calls, 1);
    assert_eq!(aux_counters2.lock().unwrap().reset_calls, 1);
}
