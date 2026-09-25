//! YUV 4:4:4 reconstruction and colour conversion for AVC444 / AVC444v2.
//!
//! MS-RDPEGFX packs a 4:4:4 picture into two 4:2:0 H.264 streams: a *main* view
//! (3.3.8.3.2/3.3.8.3.3, section B1-B3) carrying luma plus a 2x2-averaged
//! ("filtered") chroma, and an *auxiliary* view (B4-B7 for AVC444, B4-B9 for
//! AVC444v2) carrying the chroma detail the average threw away. The decoder
//! reconstructs the true chroma at colour-conversion time with a reverse
//! filter (3.3.8.3.1).
//!
//! This module keeps a persistent [`Yuv444Planes`] per surface: `(2x,2y)`
//! samples hold the filtered average (replicated across the whole 2x2 block by
//! [`apply_main_view`] so the reverse filter degenerates to the identity when
//! only the main view has been painted), everything else holds the
//! aux-carried sample once [`apply_aux_view`] has painted it.
//! [`convert_rect_to_rgba`] applies the reverse filter and the colour matrix
//! on the fly, over exactly the rect being converted.

use crate::decode::Yuv420View;

// ============================================================================
// Colour conversion (MS-RDPEGFX 3.3.8.3.1)
// ============================================================================

/// Full-range BT.709 integer colour matrix, per MS-RDPEGFX 3.3.8.3.1.
pub mod color {
    /// `Y` multiplier.
    pub const Y_SCALE: i32 = 256;
    /// `R` contribution from `V`.
    pub const R_V: i32 = 403;
    /// `G` contribution from `U`.
    pub const G_U: i32 = -48;
    /// `G` contribution from `V`.
    pub const G_V: i32 = -120;
    /// `B` contribution from `U`.
    pub const B_U: i32 = 475;
    /// Arithmetic right shift applied to every channel.
    pub const SHIFT: i32 = 8;

    /// Above this absolute difference (in the reverse filter), the reversed
    /// sample is trusted over the filtered one.
    pub const REVERSE_FILTER_THRESHOLD: i32 = 30;

    /// Clamp to `0..=255` without ever failing the `u8` conversion.
    fn clamp_u8(value: i32) -> u8 {
        u8::try_from(value.clamp(0, 255)).unwrap_or(0)
    }

    /// `R = (256*Y + 403*(V-128)) >> 8`, `G = (256*Y - 48*(U-128) - 120*(V-128)) >> 8`,
    /// `B = (256*Y + 475*(U-128)) >> 8`, each clamped to `0..=255`.
    pub fn yuv_to_rgb(y: u8, u: u8, v: u8) -> [u8; 3] {
        let y = i32::from(y);
        let u = i32::from(u) - 128;
        let v = i32::from(v) - 128;

        let r = (Y_SCALE * y + R_V * v) >> SHIFT;
        let g = (Y_SCALE * y + G_U * u + G_V * v) >> SHIFT;
        let b = (Y_SCALE * y + B_U * u) >> SHIFT;

        [clamp_u8(r), clamp_u8(g), clamp_u8(b)]
    }

    /// Encoder-side forward transform (used by tests and [`super::split`] fixtures):
    /// `Y = (54R + 183G + 18B) >> 8`, `U = ((-29R - 99G + 128B) >> 8) + 128`,
    /// `V = ((128R - 116G - 12B) >> 8) + 128`, each clamped to `0..=255`.
    pub fn rgb_to_yuv(r: u8, g: u8, b: u8) -> [u8; 3] {
        let r = i32::from(r);
        let g = i32::from(g);
        let b = i32::from(b);

        let y = (54 * r + 183 * g + 18 * b) >> SHIFT;
        let u = ((-29 * r - 99 * g + 128 * b) >> SHIFT) + 128;
        let v = ((128 * r - 116 * g - 12 * b) >> SHIFT) + 128;

        [clamp_u8(y), clamp_u8(u), clamp_u8(v)]
    }

    /// Reverse the main view's 2x2 average: `filtered = (a0 + a + b + c) / 4`
    /// where `a0` is the value being recovered, so
    /// `a0_rev = 4*filtered - a - b - c`. The comparison against
    /// [`REVERSE_FILTER_THRESHOLD`] uses the *unclamped* `a0_rev`; only the
    /// chosen value is clamped to `0..=255`.
    pub fn reverse_filter(filtered: u8, a: u8, b: u8, c: u8) -> u8 {
        let filtered_i = i32::from(filtered);
        let reversed = 4 * filtered_i - i32::from(a) - i32::from(b) - i32::from(c);

        if (reversed - filtered_i).abs() > REVERSE_FILTER_THRESHOLD {
            clamp_u8(reversed)
        } else {
            filtered
        }
    }
}

// ============================================================================
// Bounds-checked plane access helpers
// ============================================================================

fn checked_index(width: usize, x: usize, y: usize) -> Option<usize> {
    y.checked_mul(width)?.checked_add(x)
}

fn get_pixel(plane: &[u8], width: usize, x: usize, y: usize) -> Option<u8> {
    checked_index(width, x, y).and_then(|idx| plane.get(idx).copied())
}

/// A `width`-wide row starting at `row * stride`, or `None` if that would
/// overflow or run past `plane`'s end. Never panics.
fn row_slice(plane: &[u8], stride: usize, row: usize, width: usize) -> Option<&[u8]> {
    let start = row.checked_mul(stride)?;
    let end = start.checked_add(width)?;
    plane.get(start..end)
}

/// The mutable `[left, right)` slice of `row` in a `width`-wide plane, or
/// `None` if that would overflow or run past `plane`'s end. Never panics.
fn row_slice_mut(plane: &mut [u8], width: usize, row: usize, left: usize, right: usize) -> Option<&mut [u8]> {
    let row_start = row.checked_mul(width)?;
    let start = row_start.checked_add(left)?;
    let end = row_start.checked_add(right)?;
    plane.get_mut(start..end)
}

/// The mutable `[left, right)` slices of `row` and `row + 1` in a
/// `width`-wide plane, or `None` if either would overflow or run past
/// `plane`'s end. Never panics.
fn row_pair_slice_mut(
    plane: &mut [u8],
    width: usize,
    row: usize,
    left: usize,
    right: usize,
) -> Option<(&mut [u8], &mut [u8])> {
    let row_start = row.checked_mul(width)?;
    let next_row_start = row_start.checked_add(width)?;
    let total_end = next_row_start.checked_add(width)?;
    let both_rows = plane.get_mut(row_start..total_end)?;
    let (first, second) = both_rows.split_at_mut(width);
    Some((first.get_mut(left..right)?, second.get_mut(left..right)?))
}

// ============================================================================
// PlaneRect
// ============================================================================

/// An exclusive rectangle over a [`Yuv444Planes`] (`right`/`bottom` are one
/// past the last included sample), in `usize` plane coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PlaneRect {
    pub left: usize,
    pub top: usize,
    pub right: usize,
    pub bottom: usize,
}

impl PlaneRect {
    pub fn width(&self) -> usize {
        self.right.saturating_sub(self.left)
    }

    pub fn height(&self) -> usize {
        self.bottom.saturating_sub(self.top)
    }

    pub fn is_empty(&self) -> bool {
        self.width() == 0 || self.height() == 0
    }

    /// Clip to `[0, w) x [0, h)`.
    #[must_use]
    pub fn clip(&self, w: usize, h: usize) -> Self {
        Self {
            left: self.left.min(w),
            top: self.top.min(h),
            right: self.right.min(w),
            bottom: self.bottom.min(h),
        }
    }

    #[must_use]
    pub fn intersect(&self, other: &Self) -> Self {
        Self {
            left: self.left.max(other.left),
            top: self.top.max(other.top),
            right: self.right.min(other.right),
            bottom: self.bottom.min(other.bottom),
        }
    }
}

impl From<&ironrdp_pdu::geometry::ExclusiveRectangle> for PlaneRect {
    fn from(rect: &ironrdp_pdu::geometry::ExclusiveRectangle) -> Self {
        Self {
            left: usize::from(rect.left),
            top: usize::from(rect.top),
            right: usize::from(rect.right),
            bottom: usize::from(rect.bottom),
        }
    }
}

/// Only for callers that genuinely hold an inclusive rect: `right`/`bottom`
/// are widened by one to become exclusive.
impl From<&ironrdp_pdu::geometry::InclusiveRectangle> for PlaneRect {
    fn from(rect: &ironrdp_pdu::geometry::InclusiveRectangle) -> Self {
        Self {
            left: usize::from(rect.left),
            top: usize::from(rect.top),
            right: usize::from(rect.right) + 1,
            bottom: usize::from(rect.bottom) + 1,
        }
    }
}

// ============================================================================
// Yuv444Planes
// ============================================================================

/// The persistent, not-yet-reverse-filtered 4:4:4 picture for one surface.
///
/// `(2x, 2y)` chroma samples hold the main view's filtered average (`Ũ`/`Ṽ`);
/// [`apply_main_view`] replicates that average across the whole 2x2 block so
/// the reverse filter is the identity until an aux view overwrites the other
/// three positions.
#[derive(Debug, Clone)]
pub struct Yuv444Planes {
    width: usize,
    height: usize,
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
}

impl Yuv444Planes {
    /// `width x height`, `Y = 0`, `U = V = 128` (mid-grey, matches an
    /// unpainted surface). If `width * height` overflows, both dimensions
    /// are clamped to `0` so `width()`/`height()` never disagree with the
    /// (empty) backing storage.
    pub fn new(width: usize, height: usize) -> Self {
        let (width, height, len) = match width.checked_mul(height) {
            Some(len) => (width, height, len),
            None => (0, 0, 0),
        };
        Self {
            width,
            height,
            y: vec![0u8; len],
            u: vec![128u8; len],
            v: vec![128u8; len],
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn y(&self) -> &[u8] {
        &self.y
    }

    pub fn u(&self) -> &[u8] {
        &self.u
    }

    pub fn v(&self) -> &[u8] {
        &self.v
    }

    /// Reallocate (and reset to the `new` defaults) if the dimensions
    /// changed. Returns `true` when it did.
    #[must_use]
    pub fn resize_if_needed(&mut self, width: usize, height: usize) -> bool {
        if self.width == width && self.height == height {
            return false;
        }
        *self = Self::new(width, height);
        true
    }
}

/// Which auxiliary-view layout produced a decoded stream: AVC444 (per
/// 16x16 macroblock) or AVC444v2 (whole frame).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaLayout {
    /// MS-RDPEGFX 3.3.8.3.2.
    V1,
    /// MS-RDPEGFX 3.3.8.3.3.
    V2,
}

// ============================================================================
// Main view (B1-B3)
// ============================================================================

/// Paint the main (4:2:0) view's luma and filtered chroma into `dst` over
/// `rect`, expanded outward to even boundaries and bounded by
/// `min(dst, main)` dimensions. A no-op if `main` is not
/// [`well formed`](Yuv420View::is_well_formed) or the clipped rect is empty.
///
/// Chroma is replicated across the whole 2x2 block (architecture note: this
/// makes the reverse filter the identity until an aux view overwrites the
/// other three samples).
pub fn apply_main_view(dst: &mut Yuv444Planes, main: &Yuv420View<'_>, rect: PlaneRect) {
    if !main.is_well_formed() {
        return;
    }
    let (Ok(main_w), Ok(main_h)) = (usize::try_from(main.width), usize::try_from(main.height)) else {
        return;
    };

    let w = dst.width.min(main_w);
    let h = dst.height.min(main_h);
    if w < 2 || h < 2 {
        return;
    }
    let chroma_w = main_w.div_ceil(2);

    let expanded = PlaneRect {
        left: rect.left - rect.left % 2,
        top: rect.top - rect.top % 2,
        right: rect.right.checked_add(rect.right % 2).unwrap_or(w),
        bottom: rect.bottom.checked_add(rect.bottom % 2).unwrap_or(h),
    }
    .clip(w, h);
    if expanded.is_empty() {
        return;
    }

    // Reused across every row pair: the chroma row is built once (each
    // source sample duplicated into two destination columns) then copied
    // into both the even and the odd destination row.
    let row_width = expanded.right - expanded.left;
    let mut u_buf = vec![0u8; row_width];
    let mut v_buf = vec![0u8; row_width];

    let mut y2 = expanded.top;
    while y2 + 2 <= expanded.bottom {
        let my = y2 / 2;
        let rows = (
            row_slice(main.y, main.y_stride, y2, main_w),
            row_slice(main.y, main.y_stride, y2 + 1, main_w),
            row_slice(main.u, main.chroma_stride, my, chroma_w),
            row_slice(main.v, main.chroma_stride, my, chroma_w),
        );

        if let (Some(y_row0), Some(y_row1), Some(u_row), Some(v_row)) = rows {
            let y_cols = (
                y_row0.get(expanded.left..expanded.right),
                y_row1.get(expanded.left..expanded.right),
            );
            let chroma_cols = (
                u_row.get(expanded.left / 2..expanded.right / 2),
                v_row.get(expanded.left / 2..expanded.right / 2),
            );

            if let ((Some(y_src0), Some(y_src1)), (Some(u_src), Some(v_src))) = (y_cols, chroma_cols) {
                if let Some((dst_y0, dst_y1)) =
                    row_pair_slice_mut(&mut dst.y, dst.width, y2, expanded.left, expanded.right)
                {
                    dst_y0.copy_from_slice(y_src0);
                    dst_y1.copy_from_slice(y_src1);
                }

                for (i, (&u, &v)) in u_src.iter().zip(v_src.iter()).enumerate() {
                    u_buf[2 * i] = u;
                    u_buf[2 * i + 1] = u;
                    v_buf[2 * i] = v;
                    v_buf[2 * i + 1] = v;
                }

                if let Some((u_even, u_odd)) =
                    row_pair_slice_mut(&mut dst.u, dst.width, y2, expanded.left, expanded.right)
                {
                    u_even.copy_from_slice(&u_buf);
                    u_odd.copy_from_slice(u_even);
                }
                if let Some((v_even, v_odd)) =
                    row_pair_slice_mut(&mut dst.v, dst.width, y2, expanded.left, expanded.right)
                {
                    v_even.copy_from_slice(&v_buf);
                    v_odd.copy_from_slice(v_even);
                }
            }
        }

        y2 += 2;
    }
}

// ============================================================================
// Auxiliary view (B4-B7 / B4-B9)
// ============================================================================

/// Paint the auxiliary view's chroma detail into `dst` over `rect` (clipped
/// to `dst`, *not* expanded). Every read is bounded by `aux`'s actual decoded
/// geometry; a sample `aux` cannot source is left untouched.
pub fn apply_aux_view(dst: &mut Yuv444Planes, aux: &Yuv420View<'_>, rect: PlaneRect, layout: ChromaLayout) {
    if !aux.is_well_formed() {
        return;
    }
    let rect = rect.clip(dst.width, dst.height);
    if rect.is_empty() {
        return;
    }

    match layout {
        ChromaLayout::V1 => apply_aux_view_v1(dst, aux, rect),
        ChromaLayout::V2 => apply_aux_view_v2(dst, aux, rect),
    }
}

/// MS-RDPEGFX 3.3.8.3.2, global macroblock form: for a destination odd row
/// `r`, macroblock row `m = r/16`, local row `l = r%16`, `j = (l-1)/2`;
/// `U444(., r)` comes from aux `Y` row `16m+j` (full width), `V444(., r)`
/// from aux `Y` row `16m+8+j`. Destination even rows `2y` take
/// `U444(2x+1, 2y)` from aux `U(x, y)`, `V444(2x+1, 2y)` from aux `V(x, y)`.
///
/// Rows the aux frame's *actual* decoded height cannot source (below the
/// nominal 16-aligned height) are left untouched — the CVE-2026-85090 guard.
#[expect(
    clippy::similar_names,
    reason = "row_u_idx/row_v_idx name the U/V source rows per MS-RDPEGFX B4/B5"
)]
fn apply_aux_view_v1(dst: &mut Yuv444Planes, aux: &Yuv420View<'_>, rect: PlaneRect) {
    let (Ok(aux_w), Ok(aux_h)) = (usize::try_from(aux.width), usize::try_from(aux.height)) else {
        return;
    };
    let chroma_w = aux_w.div_ceil(2);
    let chroma_h = aux_h.div_ceil(2);

    let col_end = rect.right.min(aux_w);
    if col_end <= rect.left {
        return;
    }

    for r in rect.top..rect.bottom {
        if !r.is_multiple_of(2) {
            let m = r / 16;
            let l = r % 16;
            let j = (l - 1) / 2;

            let row_u_idx = m.checked_mul(16).and_then(|v| v.checked_add(j));
            let row_v_idx = m
                .checked_mul(16)
                .and_then(|v| v.checked_add(8))
                .and_then(|v| v.checked_add(j));

            // Odd destination rows are a contiguous run sourced from a single
            // aux Y row: copy the whole `[rect.left, col_end)` span in one go
            // rather than one bounds-checked write per column.
            if let Some(row_u_idx) = row_u_idx {
                if row_u_idx < aux_h {
                    if let (Some(src_row), Some(dst_row)) = (
                        row_slice(aux.y, aux.y_stride, row_u_idx, aux_w),
                        row_slice_mut(&mut dst.u, dst.width, r, rect.left, col_end),
                    ) {
                        if let Some(src_cols) = src_row.get(rect.left..col_end) {
                            dst_row.copy_from_slice(src_cols);
                        }
                    }
                }
            }

            if let Some(row_v_idx) = row_v_idx {
                if row_v_idx < aux_h {
                    if let (Some(src_row), Some(dst_row)) = (
                        row_slice(aux.y, aux.y_stride, row_v_idx, aux_w),
                        row_slice_mut(&mut dst.v, dst.width, r, rect.left, col_end),
                    ) {
                        if let Some(src_cols) = src_row.get(rect.left..col_end) {
                            dst_row.copy_from_slice(src_cols);
                        }
                    }
                }
            }
        } else {
            let gy = r / 2;
            if gy >= chroma_h {
                continue;
            }
            let u_row = row_slice(aux.u, aux.chroma_stride, gy, chroma_w);
            let v_row = row_slice(aux.v, aux.chroma_stride, gy, chroma_w);

            let start_x = if rect.left.is_multiple_of(2) {
                rect.left + 1
            } else {
                rect.left
            };
            if start_x >= col_end {
                continue;
            }
            let (Some(dst_u_row), Some(dst_v_row)) = (
                row_slice_mut(&mut dst.u, dst.width, r, rect.left, col_end),
                row_slice_mut(&mut dst.v, dst.width, r, rect.left, col_end),
            ) else {
                continue;
            };

            let mut x = start_x;
            while x < col_end {
                let gx = (x - 1) / 2;
                let local = x - rect.left;
                if gx < chroma_w {
                    if let Some(&val) = u_row.as_ref().and_then(|row| row.get(gx)) {
                        dst_u_row[local] = val;
                    }
                    if let Some(&val) = v_row.as_ref().and_then(|row| row.get(gx)) {
                        dst_v_row[local] = val;
                    }
                }
                x += 2;
            }
        }
    }
}

/// MS-RDPEGFX 3.3.8.3.3, whole frame, `W`/`H` = the aux frame's own
/// dimensions. Odd destination columns `2x+1` (any row) come from the aux
/// `Y` plane (`U` at column `x`, `V` at column `W/2+x`). Even columns on odd
/// destination rows `2y+1` come from the aux chroma planes: `4x`/`4x+2` from
/// `Uaux`/`Vaux` respectively, `V` at the `W/4+x` offset within the same
/// plane.
#[expect(
    clippy::similar_names,
    reason = "dst_u_row/dst_v_row name the U/V destination rows per MS-RDPEGFX B4-B9"
)]
fn apply_aux_view_v2(dst: &mut Yuv444Planes, aux: &Yuv420View<'_>, rect: PlaneRect) {
    let (Ok(aux_w), Ok(aux_h)) = (usize::try_from(aux.width), usize::try_from(aux.height)) else {
        return;
    };
    if aux_w == 0 || aux_h == 0 {
        return;
    }
    let chroma_w = aux_w.div_ceil(2);
    let chroma_h = aux_h.div_ceil(2);
    let half_w = aux_w / 2;
    let quarter_w = aux_w / 4;

    let col_end = rect.right;
    let row_end = rect.bottom;
    if col_end <= rect.left || row_end <= rect.top {
        return;
    }

    for y in rect.top..row_end {
        let y_odd = !y.is_multiple_of(2);
        let gy = if y_odd { (y - 1) / 2 } else { 0 };
        let chroma_row_in_range = y_odd && gy < chroma_h;

        // The row this destination row can possibly source from: the aux Y
        // row (odd destination columns) and/or the aux chroma rows (even
        // destination columns), fetched once instead of per column.
        let luma_row = if y < aux_h {
            row_slice(aux.y, aux.y_stride, y, aux_w)
        } else {
            None
        };
        let (u_row, v_row) = if chroma_row_in_range {
            (
                row_slice(aux.u, aux.chroma_stride, gy, chroma_w),
                row_slice(aux.v, aux.chroma_stride, gy, chroma_w),
            )
        } else {
            (None, None)
        };
        if luma_row.is_none() && u_row.is_none() && v_row.is_none() {
            continue;
        }

        let (Some(dst_u_row), Some(dst_v_row)) = (
            row_slice_mut(&mut dst.u, dst.width, y, rect.left, col_end),
            row_slice_mut(&mut dst.v, dst.width, y, rect.left, col_end),
        ) else {
            continue;
        };

        for x in rect.left..col_end {
            let local = x - rect.left;
            if !x.is_multiple_of(2) {
                let gx = (x - 1) / 2;
                if gx < half_w {
                    if let Some(row) = luma_row {
                        if let Some(&val) = row.get(gx) {
                            dst_u_row[local] = val;
                        }
                        if let Some(&val) = row.get(half_w + gx) {
                            dst_v_row[local] = val;
                        }
                    }
                }
            } else if chroma_row_in_range {
                if x.is_multiple_of(4) {
                    let gx = x / 4;
                    if gx < quarter_w {
                        if let Some(row) = u_row {
                            if let Some(&val) = row.get(gx) {
                                dst_u_row[local] = val;
                            }
                            if let Some(&val) = row.get(quarter_w + gx) {
                                dst_v_row[local] = val;
                            }
                        }
                    }
                } else {
                    let gx = (x - 2) / 4;
                    if gx < quarter_w {
                        if let Some(row) = v_row {
                            if let Some(&val) = row.get(gx) {
                                dst_u_row[local] = val;
                            }
                            if let Some(&val) = row.get(quarter_w + gx) {
                                dst_v_row[local] = val;
                            }
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// Colour conversion of a rect
// ============================================================================

/// Reference (per-pixel) implementation of what [`convert_rect_to_rgba`]
/// computes inline over pre-sliced rows for its hot path: the
/// reverse-filtered value at an even position whose full 2x2 block is inside
/// the plane, the raw sample everywhere else. Test-only: not called from
/// production code, kept so tests can check individual samples directly.
#[cfg(test)]
pub(crate) fn reconstructed_chroma(src: &Yuv444Planes, x: usize, y: usize) -> (u8, u8) {
    let (Some(filtered_u), Some(filtered_v)) = (get_pixel(&src.u, src.width, x, y), get_pixel(&src.v, src.width, x, y))
    else {
        return (128, 128);
    };

    if !x.is_multiple_of(2) || !y.is_multiple_of(2) || x + 1 >= src.width || y + 1 >= src.height {
        return (filtered_u, filtered_v);
    }

    let neighbours = (
        get_pixel(&src.u, src.width, x + 1, y),
        get_pixel(&src.u, src.width, x, y + 1),
        get_pixel(&src.u, src.width, x + 1, y + 1),
        get_pixel(&src.v, src.width, x + 1, y),
        get_pixel(&src.v, src.width, x, y + 1),
        get_pixel(&src.v, src.width, x + 1, y + 1),
    );

    let (Some(right_u), Some(down_u), Some(diag_u), Some(right_v), Some(down_v), Some(diag_v)) = neighbours else {
        return (filtered_u, filtered_v);
    };

    (
        color::reverse_filter(filtered_u, right_u, down_u, diag_u),
        color::reverse_filter(filtered_v, right_v, down_v, diag_v),
    )
}

/// Clip `rect` to `src`, refill `out` with `width*height*4` RGBA (alpha
/// `0xFF`) and return the clipped rect.
///
/// Per row: the `U`/`V` rows are copied into scratch, even rows have their
/// even columns overridden with the reverse-filtered value (skipped where
/// the full 2x2 block would run past the plane), then a straight per-pixel
/// loop converts `Y`/scratch `U`/scratch `V` to RGBA with no branching.
pub fn convert_rect_to_rgba(src: &Yuv444Planes, rect: PlaneRect, out: &mut Vec<u8>) -> PlaneRect {
    let clipped = rect.clip(src.width, src.height);
    let w = clipped.width();
    let h = clipped.height();

    out.clear();
    let Some(len) = w.checked_mul(h).and_then(|n| n.checked_mul(4)) else {
        return clipped;
    };
    out.resize(len, 0);
    if w == 0 || h == 0 {
        return clipped;
    }

    let mut u_scratch = vec![0u8; w];
    let mut v_scratch = vec![0u8; w];

    for row in 0..h {
        let y = clipped.top + row;
        // Reslice the row from src's own width rather than computing
        // `y * src.width + clipped.left` as a standalone (unchecked)
        // multiply: row_slice bounds-checks the multiply, then the row is
        // sliced down to the clipped columns.
        let (Some(y_row), Some(u_row), Some(v_row)) = (
            row_slice(&src.y, src.width, y, src.width).and_then(|r| r.get(clipped.left..clipped.right)),
            row_slice(&src.u, src.width, y, src.width).and_then(|r| r.get(clipped.left..clipped.right)),
            row_slice(&src.v, src.width, y, src.width).and_then(|r| r.get(clipped.left..clipped.right)),
        ) else {
            continue;
        };

        u_scratch.copy_from_slice(u_row);
        v_scratch.copy_from_slice(v_row);

        if y.is_multiple_of(2) {
            // Reverse filter needs the current and next full-width U/V rows;
            // reslice them once per row instead of one `get_pixel` call
            // (mul+add) per neighbour per chroma pair.
            let filter_rows = (
                row_slice(&src.u, src.width, y, src.width),
                row_slice(&src.v, src.width, y, src.width),
                row_slice(&src.u, src.width, y + 1, src.width),
                row_slice(&src.v, src.width, y + 1, src.width),
            );

            if let (Some(u0), Some(v0), Some(u1), Some(v1)) = filter_rows {
                let mut x = if clipped.left.is_multiple_of(2) {
                    clipped.left
                } else {
                    clipped.left + 1
                };
                while x < clipped.right {
                    let local = x - clipped.left;
                    let samples = (
                        u0.get(x),
                        v0.get(x),
                        u0.get(x + 1),
                        v0.get(x + 1),
                        u1.get(x),
                        v1.get(x),
                        u1.get(x + 1),
                        v1.get(x + 1),
                    );
                    if let (
                        Some(&filtered_u),
                        Some(&filtered_v),
                        Some(&right_u),
                        Some(&right_v),
                        Some(&down_u),
                        Some(&down_v),
                        Some(&diag_u),
                        Some(&diag_v),
                    ) = samples
                    {
                        if let Some(slot) = u_scratch.get_mut(local) {
                            *slot = color::reverse_filter(filtered_u, right_u, down_u, diag_u);
                        }
                        if let Some(slot) = v_scratch.get_mut(local) {
                            *slot = color::reverse_filter(filtered_v, right_v, down_v, diag_v);
                        }
                    }
                    x += 2;
                }
            }
        }

        let out_start = row * w * 4;
        let Some(out_row) = out.get_mut(out_start..out_start + w * 4) else {
            continue;
        };

        for (((&y_val, &u_val), &v_val), px) in y_row
            .iter()
            .zip(u_scratch.iter())
            .zip(v_scratch.iter())
            .zip(out_row.chunks_exact_mut(4))
        {
            let rgb = color::yuv_to_rgb(y_val, u_val, v_val);
            px[0] = rgb[0];
            px[1] = rgb[1];
            px[2] = rgb[2];
            px[3] = 0xFF;
        }
    }

    clipped
}

// ============================================================================
// Reference encoder side (fixtures, tests)
// ============================================================================

/// Reference (obviously-correct, not fast) encoder side: splits a full 4:4:4
/// picture into the main and auxiliary 4:2:0 streams a real encoder would
/// produce. Used by this module's tests and by the fixture generator in
/// `ironrdp-testsuite-core`.
pub mod split {
    use super::{ChromaLayout, Yuv444Planes, get_pixel};
    use crate::decode::Yuv420View;

    /// An owned planar 4:2:0 picture, mirroring [`Yuv420View`].
    pub struct OwnedYuv420 {
        pub width: u32,
        pub height: u32,
        pub y: Vec<u8>,
        pub u: Vec<u8>,
        pub v: Vec<u8>,
    }

    impl OwnedYuv420 {
        pub fn view(&self) -> Yuv420View<'_> {
            let width = usize::try_from(self.width).unwrap_or(0);
            let chroma_width = usize::try_from(self.width.div_ceil(2)).unwrap_or(0);
            Yuv420View {
                width: self.width,
                height: self.height,
                y: &self.y,
                y_stride: width,
                u: &self.u,
                v: &self.v,
                chroma_stride: chroma_width,
            }
        }
    }

    /// Sample `plane` (`w x h`) at `(x, y)`, clamping to the last valid row
    /// and column instead of failing at the ragged edge of an odd-sized
    /// picture.
    fn sample(plane: &[u8], w: usize, h: usize, x: usize, y: usize) -> u8 {
        let cx = x.min(w.saturating_sub(1));
        let cy = y.min(h.saturating_sub(1));
        get_pixel(plane, w, cx, cy).unwrap_or(0)
    }

    /// B1-B3: `Y420 = Y444`, `Ũ = (sum of the 2x2 block + 2) / 4` (rounded
    /// average), same for `Ṽ`.
    pub fn main_view(src: &Yuv444Planes) -> OwnedYuv420 {
        let w = src.width();
        let h = src.height();
        let cw = w.div_ceil(2);
        let ch = h.div_ceil(2);

        let mut y = vec![0u8; w.saturating_mul(h)];
        for (dst_row, src_row) in y.chunks_exact_mut(w).zip(src.y().chunks_exact(w)) {
            dst_row.copy_from_slice(src_row);
        }

        let mut u = vec![128u8; cw.saturating_mul(ch)];
        let mut v = vec![128u8; cw.saturating_mul(ch)];
        for cy in 0..ch {
            for cx in 0..cw {
                let x0 = cx * 2;
                let y0 = cy * 2;

                let filtered = |plane: &[u8]| -> u8 {
                    let sum = u32::from(sample(plane, w, h, x0, y0))
                        + u32::from(sample(plane, w, h, x0 + 1, y0))
                        + u32::from(sample(plane, w, h, x0, y0 + 1))
                        + u32::from(sample(plane, w, h, x0 + 1, y0 + 1));
                    u8::try_from((sum + 2) / 4).unwrap_or(255)
                };

                if let Some(slot) = u.get_mut(cy * cw + cx) {
                    *slot = filtered(src.u());
                }
                if let Some(slot) = v.get_mut(cy * cw + cx) {
                    *slot = filtered(src.v());
                }
            }
        }

        OwnedYuv420 {
            width: u32::try_from(w).unwrap_or(0),
            height: u32::try_from(h).unwrap_or(0),
            y,
            u,
            v,
        }
    }

    /// B4-B7 (`V1`) / B4-B9 (`V2`). For `V1` the aux height is rounded *up*
    /// to a multiple of 16 so every odd row has a macroblock slot; positions
    /// the layout does not define stay at the [`Yuv420View`] default (`0` for
    /// `Y`, `128` for chroma).
    pub fn aux_view(src: &Yuv444Planes, layout: ChromaLayout) -> OwnedYuv420 {
        match layout {
            ChromaLayout::V1 => aux_view_v1(src),
            ChromaLayout::V2 => aux_view_v2(src),
        }
    }

    fn aux_view_v1(src: &Yuv444Planes) -> OwnedYuv420 {
        let w = src.width();
        let h = src.height();
        let aligned_h = h.div_ceil(16).saturating_mul(16).max(16);
        let cw = w.div_ceil(2);
        let ch = aligned_h.div_ceil(2);

        let mut y = vec![0u8; w.saturating_mul(aligned_h)];
        let macroblock_rows = aligned_h / 16;
        for m in 0..macroblock_rows {
            for j in 0..8 {
                let main_row = m * 16 + 2 * j + 1;
                if main_row >= h {
                    continue;
                }
                let u_dst_row = m * 16 + j;
                let v_dst_row = m * 16 + 8 + j;

                if let Some(dst_row) = y.get_mut(u_dst_row * w..u_dst_row * w + w) {
                    if let Some(src_row) = src.u().get(main_row * w..main_row * w + w) {
                        dst_row.copy_from_slice(src_row);
                    }
                }
                if let Some(dst_row) = y.get_mut(v_dst_row * w..v_dst_row * w + w) {
                    if let Some(src_row) = src.v().get(main_row * w..main_row * w + w) {
                        dst_row.copy_from_slice(src_row);
                    }
                }
            }
        }

        let mut u = vec![128u8; cw.saturating_mul(ch)];
        let mut v = vec![128u8; cw.saturating_mul(ch)];
        for gy in 0..ch {
            let main_row = gy * 2;
            if main_row >= h {
                continue;
            }
            for gx in 0..cw {
                let main_col = gx * 2 + 1;
                if main_col >= w {
                    continue;
                }
                if let Some(slot) = u.get_mut(gy * cw + gx) {
                    *slot = get_pixel(src.u(), w, main_col, main_row).unwrap_or(128);
                }
                if let Some(slot) = v.get_mut(gy * cw + gx) {
                    *slot = get_pixel(src.v(), w, main_col, main_row).unwrap_or(128);
                }
            }
        }

        OwnedYuv420 {
            width: u32::try_from(w).unwrap_or(0),
            height: u32::try_from(aligned_h).unwrap_or(0),
            y,
            u,
            v,
        }
    }

    fn aux_view_v2(src: &Yuv444Planes) -> OwnedYuv420 {
        let w = src.width();
        let h = src.height();
        let cw = w.div_ceil(2);
        let ch = h.div_ceil(2);
        let half_w = w / 2;
        let quarter_w = w / 4;

        let mut y = vec![0u8; w.saturating_mul(h)];
        for row in 0..h {
            for gx in 0..half_w {
                let main_col_u = gx * 2 + 1;
                let main_col_v = gx * 2 + 1;
                if main_col_u >= w {
                    continue;
                }
                if let Some(slot) = y.get_mut(row * w + gx) {
                    *slot = get_pixel(src.u(), w, main_col_u, row).unwrap_or(0);
                }
                if let Some(slot) = y.get_mut(row * w + half_w + gx) {
                    *slot = get_pixel(src.v(), w, main_col_v, row).unwrap_or(0);
                }
            }
        }

        let mut u = vec![128u8; cw.saturating_mul(ch)];
        let mut v = vec![128u8; cw.saturating_mul(ch)];
        for gy in 0..(h / 2) {
            let main_row = gy * 2 + 1;
            if main_row >= h {
                continue;
            }
            for gx in 0..quarter_w {
                let col_u = gx * 4;
                let col_v = gx * 4 + 2;

                if col_u < w {
                    if let Some(slot) = u.get_mut(gy * cw + gx) {
                        *slot = get_pixel(src.u(), w, col_u, main_row).unwrap_or(128);
                    }
                    if let Some(slot) = v.get_mut(gy * cw + gx) {
                        *slot = get_pixel(src.u(), w, col_v, main_row).unwrap_or(128);
                    }
                }
                if col_u < w {
                    if let Some(slot) = u.get_mut(gy * cw + quarter_w + gx) {
                        *slot = get_pixel(src.v(), w, col_u, main_row).unwrap_or(128);
                    }
                    if let Some(slot) = v.get_mut(gy * cw + quarter_w + gx) {
                        *slot = get_pixel(src.v(), w, col_v, main_row).unwrap_or(128);
                    }
                }
            }
        }

        OwnedYuv420 {
            width: u32::try_from(w).unwrap_or(0),
            height: u32::try_from(h).unwrap_or(0),
            y,
            u,
            v,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg_next(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state
    }

    /// Two-level (40/200) pseudo-random chroma over a `((x+y)*2) & 0xFF`
    /// luma gradient.
    fn two_level_source(width: usize, height: usize, seed: u64) -> Yuv444Planes {
        let mut planes = Yuv444Planes::new(width, height);
        let mut state = seed;
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                planes.y[idx] = u8::try_from(((x + y) * 2) & 0xFF).unwrap_or(0);
                planes.u[idx] = if lcg_next(&mut state) & 1 == 0 { 40 } else { 200 };
                planes.v[idx] = if lcg_next(&mut state) & 1 == 0 { 40 } else { 200 };
            }
        }
        planes
    }

    fn gradient_source(width: usize, height: usize) -> Yuv444Planes {
        let mut planes = Yuv444Planes::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                planes.y[idx] = u8::try_from(((x + y) * 2) & 0xFF).unwrap_or(0);
                planes.u[idx] = u8::try_from((x * 4) & 0xFF).unwrap_or(0);
                planes.v[idx] = u8::try_from((y * 4) & 0xFF).unwrap_or(0);
            }
        }
        planes
    }

    fn checkerboard_source(width: usize, height: usize) -> Yuv444Planes {
        let mut planes = Yuv444Planes::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                planes.y[idx] = u8::try_from(((x + y) * 2) & 0xFF).unwrap_or(0);
                let level = if (x + y).is_multiple_of(2) { 40 } else { 200 };
                planes.u[idx] = level;
                planes.v[idx] = level;
            }
        }
        planes
    }

    fn full_rect(w: usize, h: usize) -> PlaneRect {
        PlaneRect {
            left: 0,
            top: 0,
            right: w,
            bottom: h,
        }
    }

    /// Reconstructs `dst` from `src` by splitting into main + aux (per
    /// `layout`) and applying both views over the full frame.
    fn round_trip(src: &Yuv444Planes, layout: ChromaLayout) -> Yuv444Planes {
        let main = split::main_view(src);
        let aux = split::aux_view(src, layout);

        let mut dst = Yuv444Planes::new(src.width(), src.height());
        apply_main_view(&mut dst, &main.view(), full_rect(src.width(), src.height()));
        apply_aux_view(&mut dst, &aux.view(), full_rect(src.width(), src.height()), layout);
        dst
    }

    fn direct_rgba(src: &Yuv444Planes) -> Vec<u8> {
        let mut out = Vec::with_capacity(src.width() * src.height() * 4);
        for y in 0..src.height() {
            for x in 0..src.width() {
                let idx = y * src.width() + x;
                let rgb = color::yuv_to_rgb(src.y()[idx], src.u()[idx], src.v()[idx]);
                out.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 0xFF]);
            }
        }
        out
    }

    // -- 1. colour matrix --------------------------------------------------

    #[test]
    fn color_matrix_matches_ms_rdpegfx_3_3_8_3_1() {
        assert_eq!(color::yuv_to_rgb(128, 128, 255), [255, 68, 128]);
        assert_eq!(color::yuv_to_rgb(0, 128, 128), [0, 0, 0]);
        assert_eq!(color::yuv_to_rgb(255, 128, 128), [255, 255, 255]);
        // Negative shift behaviour: G with U > 128 must move negative before
        // clamping, not wrap.
        let [_, g, _] = color::yuv_to_rgb(0, 255, 128);
        assert_eq!(g, 0);
    }

    #[test]
    fn forward_and_reverse_transform_round_trip_primaries_and_greys_within_3() {
        // The MS-RDPEGFX forward coefficients (54 + 183 + 18 = 255, not 256) do not
        // exactly invert the reverse matrix at full saturation; a pure red primary
        // measures 3 off (252 vs 255) here. Everything else is within 2.
        const TOLERANCE: u32 = 3;
        let samples = [
            (0, 0, 0),
            (255, 255, 255),
            (128, 128, 128),
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (64, 200, 32),
        ];
        for (r, g, b) in samples {
            let [y, u, v] = color::rgb_to_yuv(r, g, b);
            let [rr, rg, rb] = color::yuv_to_rgb(y, u, v);
            assert!(i32::from(rr).abs_diff(i32::from(r)) <= TOLERANCE, "r: {rr} vs {r}");
            assert!(i32::from(rg).abs_diff(i32::from(g)) <= TOLERANCE, "g: {rg} vs {g}");
            assert!(i32::from(rb).abs_diff(i32::from(b)) <= TOLERANCE, "b: {rb} vs {b}");
        }
    }

    // -- 2. reverse filter ---------------------------------------------------

    #[test]
    fn reverse_filter_keeps_the_filtered_value_up_to_30_and_reverses_above() {
        // a=b=c=filtered: reversed == filtered, difference 0, keep Ũ.
        assert_eq!(color::reverse_filter(100, 100, 100, 100), 100);

        // Construct a block whose reversed value differs from filtered by
        // exactly 30 (boundary, kept) and by 31 (reversed, over threshold).
        // filtered = (a0 + a + b + c) / 4 with a=b=c=0, a0=120 -> filtered=30,
        // reversed = 4*30 - 0 - 0 - 0 = 120, |120-30| = 90 > 30 -> reversed.
        assert_eq!(color::reverse_filter(30, 0, 0, 0), 120);

        // filtered=10, a=b=c=10 -> reversed=10, diff=0 -> keep.
        assert_eq!(color::reverse_filter(10, 10, 10, 10), 10);

        // Clamping of an out-of-range reversed value.
        assert_eq!(color::reverse_filter(250, 0, 0, 0), 255);
        assert_eq!(color::reverse_filter(5, 255, 255, 255), 0);
    }

    // -- 3. two-level chroma round trip --------------------------------------

    fn two_level_round_trips_exactly(layout: ChromaLayout, width: usize, height: usize) {
        let src = two_level_source(width, height, 0x1234_5678_9abc_def0);
        let dst = round_trip(&src, layout);

        let mut out = Vec::new();
        let clipped = convert_rect_to_rgba(&dst, full_rect(width, height), &mut out);
        assert_eq!(clipped, full_rect(width, height));
        assert_eq!(out, direct_rgba(&src));

        for y in 0..height {
            for x in 0..width {
                if !x.is_multiple_of(2) || !y.is_multiple_of(2) {
                    let idx = y * width + x;
                    assert_eq!(dst.u()[idx], src.u()[idx], "u mismatch at ({x},{y})");
                    assert_eq!(dst.v()[idx], src.v()[idx], "v mismatch at ({x},{y})");
                }
            }
        }
    }

    #[test]
    fn two_level_chroma_image_round_trips_exactly_v1() {
        two_level_round_trips_exactly(ChromaLayout::V1, 48, 32);
        two_level_round_trips_exactly(ChromaLayout::V1, 64, 64);
    }

    #[test]
    fn two_level_chroma_image_round_trips_exactly_v2() {
        two_level_round_trips_exactly(ChromaLayout::V2, 48, 32);
        two_level_round_trips_exactly(ChromaLayout::V2, 64, 64);
    }

    // -- 4. gradient round trip ----------------------------------------------

    fn gradient_round_trips(layout: ChromaLayout) {
        let width = 64;
        let height = 32;
        let src = gradient_source(width, height);
        let dst = round_trip(&src, layout);

        for y in 0..height {
            for x in 0..width {
                let (u, v) = reconstructed_chroma(&dst, x, y);
                let src_idx = y * width + x;
                if x.is_multiple_of(2) && y.is_multiple_of(2) {
                    assert!(
                        i32::from(u).abs_diff(i32::from(src.u()[src_idx])) <= 3,
                        "u at ({x},{y})"
                    );
                    assert!(
                        i32::from(v).abs_diff(i32::from(src.v()[src_idx])) <= 3,
                        "v at ({x},{y})"
                    );
                } else {
                    assert_eq!(u, src.u()[src_idx], "u exact at ({x},{y})");
                    assert_eq!(v, src.v()[src_idx], "v exact at ({x},{y})");
                }
            }
        }
    }

    #[test]
    fn gradient_round_trip_is_exact_off_the_even_grid_and_within_3_on_it_v1() {
        gradient_round_trips(ChromaLayout::V1);
    }

    #[test]
    fn gradient_round_trip_is_exact_off_the_even_grid_and_within_3_on_it_v2() {
        gradient_round_trips(ChromaLayout::V2);
    }

    // -- 5. checkerboard round trip -------------------------------------------

    fn checkerboard_round_trips_exactly(layout: ChromaLayout) {
        let width = 48;
        let height = 32;
        let src = checkerboard_source(width, height);
        let dst = round_trip(&src, layout);

        let mut out = Vec::new();
        convert_rect_to_rgba(&dst, full_rect(width, height), &mut out);
        assert_eq!(out, direct_rgba(&src));
    }

    #[test]
    fn checkerboard_1px_chroma_round_trips_exactly_v1() {
        checkerboard_round_trips_exactly(ChromaLayout::V1);
    }

    #[test]
    fn checkerboard_1px_chroma_round_trips_exactly_v2() {
        checkerboard_round_trips_exactly(ChromaLayout::V2);
    }

    // -- 6. replication makes the reverse filter the identity ----------------

    #[test]
    fn main_view_replication_makes_the_reverse_filter_the_identity() {
        let width = 16;
        let height = 16;
        let src = gradient_source(width, height);
        let main = split::main_view(&src);

        let mut dst = Yuv444Planes::new(width, height);
        apply_main_view(&mut dst, &main.view(), full_rect(width, height));

        let mut out = Vec::new();
        convert_rect_to_rgba(&dst, full_rect(width, height), &mut out);

        // Plain 4:2:0 upsample: every pixel in a 2x2 block uses that block's Ũ/Ṽ.
        let mut expected = Vec::with_capacity(width * height * 4);
        for y in 0..height {
            for x in 0..width {
                let idx = (y / 2 * (width / 2)) + x / 2;
                let rgb = color::yuv_to_rgb(src.y()[y * width + x], main.u[idx], main.v[idx]);
                expected.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 0xFF]);
            }
        }
        assert_eq!(out, expected);
    }

    // -- 7. aux reads never go past the actually decoded aux geometry --------

    #[test]
    fn v1_aux_rows_beyond_the_decoded_height_are_never_read() {
        let width = 32;
        let height = 40;
        let aux_height = 40;

        let sentinel = 7u8;
        let mut dst = Yuv444Planes::new(width, height);
        dst.u.fill(sentinel);
        dst.v.fill(sentinel);

        let aux_y = vec![9u8; width * aux_height];
        let chroma_w = width.div_ceil(2);
        let chroma_h = aux_height.div_ceil(2);
        let aux_u = vec![11u8; chroma_w * chroma_h];
        let aux_v = vec![13u8; chroma_w * chroma_h];
        let aux = Yuv420View {
            width: u32::try_from(width).unwrap(),
            height: u32::try_from(aux_height).unwrap(),
            y: &aux_y,
            y_stride: width,
            u: &aux_u,
            v: &aux_v,
            chroma_stride: chroma_w,
        };

        apply_aux_view(&mut dst, &aux, full_rect(width, height), ChromaLayout::V1);

        // Macroblock row 2 covers dst rows 32..47; only 32..39 exist.
        // row_v for r in {33,35,37,39} is 16*2+8+j = 40..43, all >= aux_height.
        for r in [33, 35, 37, 39] {
            for x in 0..width {
                assert_eq!(
                    dst.v()[r * width + x],
                    sentinel,
                    "v at row {r} col {x} must be untouched"
                );
            }
        }
        // row_u for the same rows is 32..35, all < aux_height: those ARE read.
        for r in [33, 35, 37, 39] {
            for x in 0..width {
                assert_eq!(dst.u()[r * width + x], 9, "u at row {r} col {x} must be read");
            }
        }

        // Aux narrower than dst: columns beyond aux width stay untouched.
        let narrow_width = 24;
        let mut dst2 = Yuv444Planes::new(width, height);
        dst2.u.fill(sentinel);
        dst2.v.fill(sentinel);
        let narrow_y = vec![9u8; narrow_width * height];
        let narrow_chroma_w = narrow_width.div_ceil(2);
        let narrow_chroma_h = height.div_ceil(2);
        let narrow_u = vec![11u8; narrow_chroma_w * narrow_chroma_h];
        let narrow_v = vec![13u8; narrow_chroma_w * narrow_chroma_h];
        let aux2 = Yuv420View {
            width: u32::try_from(narrow_width).unwrap(),
            height: u32::try_from(height).unwrap(),
            y: &narrow_y,
            y_stride: narrow_width,
            u: &narrow_u,
            v: &narrow_v,
            chroma_stride: narrow_chroma_w,
        };
        apply_aux_view(&mut dst2, &aux2, full_rect(width, height), ChromaLayout::V1);
        for r in 0..height {
            for x in narrow_width..width {
                assert_eq!(dst2.u()[r * width + x], sentinel, "u beyond aux width at ({x},{r})");
                assert_eq!(dst2.v()[r * width + x], sentinel, "v beyond aux width at ({x},{r})");
            }
        }
    }

    // -- 8. touches nothing outside the rect ----------------------------------

    #[test]
    fn apply_touches_nothing_outside_the_even_expanded_rect() {
        let width = 32;
        let height = 32;
        let sentinel = 42u8;
        let mut dst = Yuv444Planes::new(width, height);
        dst.y.fill(sentinel);
        dst.u.fill(sentinel);
        dst.v.fill(sentinel);

        let main = split::main_view(&gradient_source(width, height));
        let rect = PlaneRect {
            left: 5,
            top: 5,
            right: 11,
            bottom: 11,
        };
        apply_main_view(&mut dst, &main.view(), rect);

        // Expanded to even boundaries: [4,12) x [4,12).
        let expanded = PlaneRect {
            left: 4,
            top: 4,
            right: 12,
            bottom: 12,
        };
        for y in 0..height {
            for x in 0..width {
                let inside = x >= expanded.left && x < expanded.right && y >= expanded.top && y < expanded.bottom;
                if !inside {
                    let idx = y * width + x;
                    assert_eq!(dst.y()[idx], sentinel, "y outside rect at ({x},{y})");
                    assert_eq!(dst.u()[idx], sentinel, "u outside rect at ({x},{y})");
                }
            }
        }
    }

    #[test]
    fn apply_touches_nothing_outside_the_rect_aux() {
        let width = 32;
        let height = 32;
        let sentinel = 42u8;
        let mut dst = Yuv444Planes::new(width, height);
        dst.u.fill(sentinel);
        dst.v.fill(sentinel);

        let src = two_level_source(width, height, 7);
        let aux = split::aux_view(&src, ChromaLayout::V2);
        let rect = PlaneRect {
            left: 5,
            top: 5,
            right: 11,
            bottom: 11,
        };
        apply_aux_view(&mut dst, &aux.view(), rect, ChromaLayout::V2);

        for y in 0..height {
            for x in 0..width {
                let inside = x >= rect.left && x < rect.right && y >= rect.top && y < rect.bottom;
                if !inside {
                    let idx = y * width + x;
                    assert_eq!(dst.u()[idx], sentinel, "u outside rect at ({x},{y})");
                    assert_eq!(dst.v()[idx], sentinel, "v outside rect at ({x},{y})");
                }
            }
        }
    }

    // -- 9. convert_rect_to_rgba length / clipping ----------------------------

    #[test]
    fn convert_returns_the_clipped_rect_and_the_exact_length() {
        let planes = Yuv444Planes::new(10, 10);
        let mut out = Vec::new();

        let clipped = convert_rect_to_rgba(&planes, full_rect(20, 20), &mut out);
        assert_eq!(clipped, full_rect(10, 10));
        assert_eq!(out.len(), 10 * 10 * 4);

        let empty = PlaneRect {
            left: 5,
            top: 5,
            right: 5,
            bottom: 8,
        };
        let clipped_empty = convert_rect_to_rgba(&planes, empty, &mut out);
        assert!(clipped_empty.is_empty());
        assert!(out.is_empty());

        let inverted = PlaneRect {
            left: 8,
            top: 8,
            right: 2,
            bottom: 2,
        };
        let clipped_inverted = convert_rect_to_rgba(&planes, inverted, &mut out);
        assert!(clipped_inverted.is_empty());
        assert!(out.is_empty());
    }

    // -- 10. ill-formed view is ignored ---------------------------------------

    #[test]
    fn an_ill_formed_view_is_ignored() {
        let width = 16;
        let height = 16;
        let sentinel = 5u8;
        let mut dst = Yuv444Planes::new(width, height);
        dst.y.fill(sentinel);
        dst.u.fill(sentinel);
        dst.v.fill(sentinel);

        let short_y = [0u8; 4]; // too short for a 16x16 frame.
        let short_u = [0u8; 4];
        let short_v = [0u8; 4];
        let bad_main = Yuv420View {
            width: 16,
            height: 16,
            y: &short_y,
            y_stride: 16,
            u: &short_u,
            v: &short_v,
            chroma_stride: 8,
        };
        apply_main_view(&mut dst, &bad_main, full_rect(width, height));
        assert!(dst.y().iter().all(|&b| b == sentinel));
        assert!(dst.u().iter().all(|&b| b == sentinel));

        apply_aux_view(&mut dst, &bad_main, full_rect(width, height), ChromaLayout::V1);
        assert!(dst.u().iter().all(|&b| b == sentinel));
        assert!(dst.v().iter().all(|&b| b == sentinel));
    }

    // -- PlaneRect / Yuv444Planes basics ---------------------------------------

    #[test]
    fn plane_rect_from_exclusive_and_inclusive_rectangles() {
        use ironrdp_pdu::geometry::{ExclusiveRectangle, InclusiveRectangle};

        let excl = ExclusiveRectangle {
            left: 1,
            top: 2,
            right: 10,
            bottom: 20,
        };
        let from_excl = PlaneRect::from(&excl);
        assert_eq!(
            from_excl,
            PlaneRect {
                left: 1,
                top: 2,
                right: 10,
                bottom: 20
            }
        );

        let incl = InclusiveRectangle {
            left: 1,
            top: 2,
            right: 9,
            bottom: 19,
        };
        let from_incl = PlaneRect::from(&incl);
        assert_eq!(
            from_incl,
            PlaneRect {
                left: 1,
                top: 2,
                right: 10,
                bottom: 20
            }
        );
    }

    #[test]
    fn resize_if_needed_resets_to_defaults_only_on_a_dimension_change() {
        let mut planes = Yuv444Planes::new(4, 4);
        planes.y[0] = 200;
        assert!(!planes.resize_if_needed(4, 4));
        assert_eq!(planes.y()[0], 200);

        assert!(planes.resize_if_needed(8, 4));
        assert_eq!(planes.y()[0], 0);
        assert_eq!(planes.u()[0], 128);
        assert_eq!(planes.width(), 8);
    }

    #[test]
    fn overflowing_dimensions_yield_empty_planes() {
        let planes = Yuv444Planes::new(usize::MAX, 2);
        assert_eq!(planes.width(), 0);
        assert_eq!(planes.height(), 0);
        assert!(planes.y().is_empty());
        assert!(planes.u().is_empty());
        assert!(planes.v().is_empty());
    }
}
