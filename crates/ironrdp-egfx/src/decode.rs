//! Codec decoder traits for client-side EGFX processing
//!
//! This module provides pluggable decoder traits that allow consumers
//! to bring their own codec implementations (e.g., openh264, ffmpeg,
//! hardware decoders). The traits are designed for core tier: no I/O,
//! `Send` only. They are intended for use in `std` environments;
//! `no_std` + `alloc` support is not currently guaranteed.
//!
//! # Protocol Context
//!
//! H.264 data arrives inside [RFX_AVC420_BITMAP_STREAM][1] payloads
//! within `RDPGFX_WIRE_TO_SURFACE_PDU_1` messages. The NAL units
//! are in AVC format (4-byte big-endian length prefix per NAL unit),
//! not Annex B (start code prefix).
//!
//! [1]: https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-rdpegfx/d65c3f9c-2088-4302-90c0-53adc0e11a78

use core::fmt;

// ============================================================================
// Decoded Frame
// ============================================================================

/// Decoded bitmap frame from an H.264 decoder
///
/// Contains RGBA pixel data for a decoded H.264 frame.
/// The pixel data is in RGBA format (4 bytes per pixel),
/// row-major, top-to-bottom, left-to-right.
#[derive(Clone)]
#[non_exhaustive]
pub struct DecodedFrame {
    data: Vec<u8>,
    width: u32,
    height: u32,
}

impl DecodedFrame {
    #[expect(
        clippy::as_conversions,
        reason = "usize to u64 is lossless on all supported platforms (32/64-bit)"
    )]
    pub fn new(data: Vec<u8>, width: u32, height: u32) -> Self {
        debug_assert_eq!(
            data.len() as u64,
            u64::from(width).saturating_mul(u64::from(height)).saturating_mul(4),
            "DecodedFrame buffer must be RGBA8888 (width * height * 4 bytes)",
        );
        Self { data, width, height }
    }

    /// RGBA pixel data (4 bytes per pixel).
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Frame width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Frame height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Consume the frame and return the owned RGBA buffer.
    pub fn into_data(self) -> Vec<u8> {
        self.data
    }
}

impl fmt::Debug for DecodedFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecodedFrame")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("data_len", &self.data.len())
            .finish()
    }
}

// ============================================================================
// Decoder Error
// ============================================================================

/// Error type for decoder operations
#[derive(Debug)]
#[non_exhaustive]
pub struct DecoderError {
    context: String,
    source: Option<Box<dyn core::error::Error + Send + Sync>>,
}

impl DecoderError {
    /// Create a decoder error with a source error
    pub fn new(context: impl Into<String>, source: impl core::error::Error + Send + Sync + 'static) -> Self {
        Self {
            context: context.into(),
            source: Some(Box::new(source)),
        }
    }

    /// Create a decoder error with only a message
    pub fn msg(context: impl Into<String>) -> Self {
        Self {
            context: context.into(),
            source: None,
        }
    }
}

impl fmt::Display for DecoderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "decoder error: {}", self.context)?;
        if let Some(ref source) = self.source {
            write!(f, ": {source}")?;
        }
        Ok(())
    }
}

impl core::error::Error for DecoderError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        self.source.as_deref().map(|e| {
            let err: &(dyn core::error::Error + 'static) = e;
            err
        })
    }
}

/// Result type for decoder operations
pub type DecoderResult<T> = Result<T, DecoderError>;

// ============================================================================
// H.264 Decoder Trait
// ============================================================================

/// Trait for H.264 (AVC) decoders
///
/// Implement this trait to provide H.264 decode capability to the
/// EGFX client. The decoder receives AVC-format NAL units (length-prefixed,
/// not Annex B) from `RFX_AVC420_BITMAP_STREAM` payloads.
///
/// # Thread Safety
///
/// Implementations must be `Send` to work with the DVC framework.
///
/// # Example
///
/// ```ignore
/// use ironrdp_egfx::decode::{H264Decoder, DecodedFrame, DecoderResult};
///
/// struct MyH264Decoder { /* ... */ }
///
/// impl H264Decoder for MyH264Decoder {
///     fn decode(&mut self, data: &[u8]) -> DecoderResult<DecodedFrame> {
///         // Decode H.264 NAL units to RGBA
///         todo!()
///     }
/// }
/// ```
pub trait H264Decoder: Send {
    /// Decode AVC-format H.264 NAL units (4-byte BE length prefix, not Annex B)
    /// into an RGBA bitmap.
    ///
    /// Frame dimensions may exceed the destination rectangle due to
    /// macroblock alignment (16x16). The caller crops to fit.
    fn decode(&mut self, data: &[u8]) -> DecoderResult<DecodedFrame>;

    /// Reset the decoder state
    ///
    /// Called when surfaces are reset (e.g., on `ResetGraphics`).
    /// The decoder should drop any internal state and prepare for
    /// a new stream.
    fn reset(&mut self) {
        // Default: no-op
    }
}

// ============================================================================
// Planar YUV Output
// ============================================================================

/// A borrowed planar 4:2:0 picture (I420) as a decoder hands it back.
///
/// `y` holds `height` rows of `y_stride` bytes (at least `width` used per row);
/// `u` and `v` each hold `height.div_ceil(2)` rows of `chroma_stride` bytes
/// (at least `width.div_ceil(2)` used per row). The view is valid until the
/// decoder's next `decode_yuv` call.
#[derive(Clone, Copy)]
pub struct Yuv420View<'a> {
    pub width: u32,
    pub height: u32,
    pub y: &'a [u8],
    pub y_stride: usize,
    pub u: &'a [u8],
    pub v: &'a [u8],
    pub chroma_stride: usize,
}

impl Yuv420View<'_> {
    /// Number of chroma columns: `width.div_ceil(2)`.
    fn chroma_width(&self) -> usize {
        // `u32::div_ceil` result always fits in `usize` on all supported platforms.
        usize::try_from(self.width.div_ceil(2)).unwrap_or(usize::MAX)
    }

    /// Number of chroma rows: `height.div_ceil(2)`.
    fn chroma_height(&self) -> usize {
        usize::try_from(self.height.div_ceil(2)).unwrap_or(usize::MAX)
    }

    /// `true` when every plane is long enough for the declared geometry and
    /// every stride is at least the plane's used width. Never panics and never
    /// overflows: geometry that does not fit in `usize` is simply not well formed.
    pub fn is_well_formed(&self) -> bool {
        if self.width == 0 || self.height == 0 {
            return false;
        }

        let Ok(width) = usize::try_from(self.width) else {
            return false;
        };
        let Ok(height) = usize::try_from(self.height) else {
            return false;
        };
        let chroma_width = self.chroma_width();
        let chroma_height = self.chroma_height();

        Self::plane_is_well_formed(self.y.len(), self.y_stride, width, height)
            && Self::plane_is_well_formed(self.u.len(), self.chroma_stride, chroma_width, chroma_height)
            && Self::plane_is_well_formed(self.v.len(), self.chroma_stride, chroma_width, chroma_height)
    }

    /// Required length is `(rows - 1) * stride + used_width` (the last row need
    /// not be padded to the stride); `stride` must be at least `used_width`.
    fn plane_is_well_formed(plane_len: usize, stride: usize, used_width: usize, rows: usize) -> bool {
        if stride < used_width {
            return false;
        }

        let Some(rows_minus_one) = rows.checked_sub(1) else {
            return false;
        };
        let Some(leading_rows_len) = rows_minus_one.checked_mul(stride) else {
            return false;
        };
        let Some(required_len) = leading_rows_len.checked_add(used_width) else {
            return false;
        };

        plane_len >= required_len
    }
}

impl fmt::Debug for Yuv420View<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Yuv420View")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("y_len", &self.y.len())
            .field("y_stride", &self.y_stride)
            .field("u_len", &self.u.len())
            .field("v_len", &self.v.len())
            .field("chroma_stride", &self.chroma_stride)
            .finish()
    }
}

/// Decoder that returns planar YUV instead of RGBA — used for the two AVC444 streams.
pub trait H264YuvDecoder: Send {
    /// Decode AVC-format H.264 NAL units (4-byte BE length prefix, not Annex B)
    /// into a borrowed planar 4:2:0 picture. Frame dimensions may exceed the
    /// surface because of macroblock alignment; the caller clips.
    fn decode_yuv(&mut self, data: &[u8]) -> DecoderResult<Yuv420View<'_>>;

    /// Reset decoder state (ResetGraphics / resync after a failure); default no-op.
    fn reset(&mut self) {
        // Default: no-op
    }
}

/// Factory the client uses to create the main and auxiliary decoder of one surface.
pub type YuvDecoderFactory = Box<dyn Fn() -> Option<Box<dyn H264YuvDecoder>> + Send>;

// ============================================================================
// OpenH264 Implementation
// ============================================================================

#[cfg(feature = "openh264")]
mod openh264_impl {
    use tracing::warn;

    use super::{DecodedFrame, DecoderError, DecoderResult, H264Decoder};

    /// H.264 decoder backed by Cisco's OpenH264 library
    ///
    /// This decoder converts AVC-format NAL units to Annex B format
    /// (as required by OpenH264), decodes to YUV420p, then converts
    /// to RGBA for the client pipeline.
    ///
    /// # Feature Gates
    ///
    /// Two construction paths are available depending on the feature flags:
    ///
    /// - `openh264-bundled`: compiles OpenH264 from source at build time.
    ///   Use [`OpenH264Decoder::new()`] to construct.
    ///
    /// - `openh264-libloading`: loads a prebuilt Cisco OpenH264 binary at
    ///   runtime. Use [`OpenH264Decoder::from_library_path()`] to construct.
    ///   The library is verified against known Cisco release hashes.
    pub struct OpenH264Decoder {
        decoder: openh264::decoder::Decoder,
        annex_b_buffer: Vec<u8>,
    }

    impl OpenH264Decoder {
        /// Create a decoder using the bundled (source-compiled) OpenH264 library
        ///
        /// This compiles OpenH264 C code at build time. The resulting binary
        /// has no patent coverage from Cisco's license agreement.
        #[cfg(feature = "openh264-bundled")]
        pub fn new() -> DecoderResult<Self> {
            let decoder = openh264::decoder::Decoder::new()
                .map_err(|e| DecoderError::new("failed to create OpenH264 decoder", e))?;

            Ok(Self {
                decoder,
                annex_b_buffer: Vec::new(),
            })
        }

        /// Create a decoder using a dynamically loaded OpenH264 library
        ///
        /// `library_path` should point to a Cisco OpenH264 prebuilt binary,
        /// which is verified against known Cisco release hashes before loading.
        /// Cisco's prebuilt binaries carry patent coverage under their license.
        #[cfg(feature = "openh264-libloading")]
        pub fn from_library_path(library_path: &std::path::Path) -> DecoderResult<Self> {
            let api = openh264::OpenH264API::from_blob_path(library_path)
                .map_err(|e| DecoderError::new("failed to load OpenH264 library", e))?;
            let decoder = openh264::decoder::Decoder::with_api_config(api, Default::default())
                .map_err(|e| DecoderError::new("failed to create OpenH264 decoder", e))?;

            Ok(Self {
                decoder,
                annex_b_buffer: Vec::new(),
            })
        }
    }

    impl H264Decoder for OpenH264Decoder {
        fn decode(&mut self, data: &[u8]) -> DecoderResult<DecodedFrame> {
            crate::pdu::avc_to_annex_b_into(data, &mut self.annex_b_buffer);

            let yuv = self
                .decoder
                .decode(&self.annex_b_buffer)
                .map_err(|e| DecoderError::new("OpenH264 decode failed", e))?
                .ok_or_else(|| DecoderError::msg("OpenH264 returned no picture"))?;

            let (width, height) = openh264::formats::YUVSource::dimensions(&yuv);

            #[expect(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "H.264 frame dimensions are always within u32 range"
            )]
            let (w32, h32) = (width as u32, height as u32);

            let rgba_size = width
                .checked_mul(height)
                .and_then(|s| s.checked_mul(4))
                .ok_or_else(|| DecoderError::msg("frame dimensions too large for RGBA allocation"))?;
            let mut rgba = vec![0u8; rgba_size];
            yuv.write_rgba8(&mut rgba);

            Ok(DecodedFrame::new(rgba, w32, h32))
        }

        fn reset(&mut self) {
            // Recreate decoder from source when available
            #[cfg(feature = "openh264-bundled")]
            match openh264::decoder::Decoder::new() {
                Ok(new_decoder) => self.decoder = new_decoder,
                Err(e) => warn!("Failed to reset OpenH264 decoder, reusing existing state: {e}"),
            }
            // In libloading-only mode, we don't have the library path stored,
            // so we can't recreate. The existing decoder handles new SPS/PPS
            // transparently when the next I-frame arrives.
        }
    }
}

#[cfg(feature = "openh264")]
pub use openh264_impl::OpenH264Decoder;

#[cfg(test)]
mod tests {
    use super::Yuv420View;

    #[test]
    fn well_formed_accepts_an_exact_i420_buffer() {
        let y = [0u8; 24]; // 6x4, stride 6
        let u = [0u8; 6]; // 3x2, stride 3
        let v = [0u8; 6];
        let view = Yuv420View {
            width: 6,
            height: 4,
            y: &y,
            y_stride: 6,
            u: &u,
            v: &v,
            chroma_stride: 3,
        };
        assert!(view.is_well_formed());
    }

    #[test]
    fn well_formed_accepts_an_unpadded_last_row() {
        // width 6, height 4, y_stride 10: buffer sized (rows-1)*stride + width.
        let y = vec![0u8; 3 * 10 + 6];
        let u = vec![0u8; 5 + 3];
        let v = vec![0u8; 5 + 3];
        let view = Yuv420View {
            width: 6,
            height: 4,
            y: &y,
            y_stride: 10,
            u: &u,
            v: &v,
            chroma_stride: 5,
        };
        assert!(view.is_well_formed());
    }

    #[test]
    fn well_formed_rejects_a_short_chroma_plane() {
        let y = [0u8; 24];
        let u = [0u8; 5]; // one byte short of the required 6.
        let v = [0u8; 6];
        let view = Yuv420View {
            width: 6,
            height: 4,
            y: &y,
            y_stride: 6,
            u: &u,
            v: &v,
            chroma_stride: 3,
        };
        assert!(!view.is_well_formed());
    }

    #[test]
    fn well_formed_rejects_a_stride_below_the_width() {
        let y = [0u8; 24];
        let u = [0u8; 6];
        let v = [0u8; 6];
        let view = Yuv420View {
            width: 6,
            height: 4,
            y: &y,
            y_stride: 5, // less than width 6.
            u: &u,
            v: &v,
            chroma_stride: 3,
        };
        assert!(!view.is_well_formed());
    }

    #[test]
    fn well_formed_rejects_zero_dimensions() {
        let y = [0u8; 24];
        let u = [0u8; 6];
        let v = [0u8; 6];

        let zero_width = Yuv420View {
            width: 0,
            height: 4,
            y: &y,
            y_stride: 6,
            u: &u,
            v: &v,
            chroma_stride: 3,
        };
        assert!(!zero_width.is_well_formed());

        let zero_height = Yuv420View {
            width: 6,
            height: 0,
            y: &y,
            y_stride: 6,
            u: &u,
            v: &v,
            chroma_stride: 3,
        };
        assert!(!zero_height.is_well_formed());
    }

    #[test]
    fn well_formed_rejects_u32_max_geometry_without_panicking() {
        let y = [0u8; 4];
        let u = [0u8; 4];
        let v = [0u8; 4];
        let view = Yuv420View {
            width: u32::MAX,
            height: u32::MAX,
            y: &y,
            y_stride: usize::MAX,
            u: &u,
            v: &v,
            chroma_stride: usize::MAX,
        };
        assert!(!view.is_well_formed());
    }
}
