//! Property tests for `ironrdp_egfx::yuv444`: the AVC444/AVC444v2
//! reconstruction and colour conversion never panics, respects its
//! documented output size and touched-region bounds, and reproduces the
//! source picture at odd positions after a full main+aux round trip, across
//! arbitrary (including tiny, empty, and inverted) rects and dimensions.

use ironrdp_egfx::yuv444::split::{self, OwnedYuv420};
use ironrdp_egfx::yuv444::{
    ChromaLayout, PlaneRect, Yuv444Planes, apply_aux_view, apply_main_view, convert_rect_to_rgba,
};

// ============================================================================
// Helpers
// ============================================================================

fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

/// A deterministically-filled `OwnedYuv420` of the given dimensions. Every
/// sample avoids `0` and `128` so those values keep working as unambiguous
/// "untouched" sentinels (`Yuv444Planes::new`'s Y/U/V defaults) in the tests
/// below.
fn filled_yuv420(width: usize, height: usize, seed: u64) -> OwnedYuv420 {
    let chroma_w = width.div_ceil(2);
    let chroma_h = height.div_ceil(2);
    let mut state = seed;
    let mut fill = |len: usize| -> Vec<u8> {
        core::iter::repeat_with(|| {
            loop {
                let val = u8::try_from(lcg_next(&mut state) & 0xFF).unwrap();
                if val != 0 && val != 128 {
                    break val;
                }
            }
        })
        .take(len)
        .collect()
    };

    OwnedYuv420 {
        width: u32::try_from(width).unwrap(),
        height: u32::try_from(height).unwrap(),
        y: fill(width * height),
        u: fill(chroma_w * chroma_h),
        v: fill(chroma_w * chroma_h),
    }
}

fn full_rect(w: usize, h: usize) -> PlaneRect {
    PlaneRect {
        left: 0,
        top: 0,
        right: w,
        bottom: h,
    }
}

/// Mirrors `apply_main_view`'s own even-expansion of `rect`, bounded by
/// `min(dst, main)`. Returns an empty rect where `apply_main_view` would
/// have been a no-op (either dimension below 2).
fn expected_main_touched_rect(rect: PlaneRect, dst_w: usize, dst_h: usize, main_w: usize, main_h: usize) -> PlaneRect {
    let w = dst_w.min(main_w);
    let h = dst_h.min(main_h);
    if w < 2 || h < 2 {
        return PlaneRect::default();
    }
    PlaneRect {
        left: rect.left - rect.left % 2,
        top: rect.top - rect.top % 2,
        right: rect.right.checked_add(rect.right % 2).unwrap_or(w),
        bottom: rect.bottom.checked_add(rect.bottom % 2).unwrap_or(h),
    }
    .clip(w, h)
}

fn dim_strategy() -> impl proptest::strategy::Strategy<Value = usize> {
    1usize..=33
}

fn edge_strategy() -> impl proptest::strategy::Strategy<Value = usize> {
    0usize..=40
}

fn layout_strategy() -> impl proptest::strategy::Strategy<Value = ChromaLayout> {
    proptest::prop_oneof![
        proptest::strategy::Just(ChromaLayout::V1),
        proptest::strategy::Just(ChromaLayout::V2),
    ]
}

// ============================================================================
// Properties
// ============================================================================

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(200))]

    /// `apply_main_view`, `apply_aux_view` and `convert_rect_to_rgba` never
    /// panic, for any combination of tiny/large dimensions and rects that
    /// touch, cross, or fall entirely outside every edge (including empty
    /// and inverted rects).
    #[test]
    fn applying_and_converting_never_panics(
        dst_w in dim_strategy(), dst_h in dim_strategy(),
        main_w in dim_strategy(), main_h in dim_strategy(),
        aux_w in dim_strategy(), aux_h in dim_strategy(),
        left in edge_strategy(), top in edge_strategy(), right in edge_strategy(), bottom in edge_strategy(),
        layout in layout_strategy(),
    ) {
        let main = filled_yuv420(main_w, main_h, 1);
        let aux = filled_yuv420(aux_w, aux_h, 2);
        let mut dst = Yuv444Planes::new(dst_w, dst_h);
        let rect = PlaneRect { left, top, right, bottom };

        apply_main_view(&mut dst, &main.view(), rect);
        apply_aux_view(&mut dst, &aux.view(), rect, layout);

        let mut out = Vec::new();
        convert_rect_to_rgba(&dst, rect, &mut out);
    }
}

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(200))]

    /// `convert_rect_to_rgba`'s output is always exactly `clipped.w * clipped.h * 4` bytes.
    #[test]
    fn convert_rect_to_rgba_output_length_matches_the_clipped_rect(
        dst_w in dim_strategy(), dst_h in dim_strategy(),
        left in edge_strategy(), top in edge_strategy(), right in edge_strategy(), bottom in edge_strategy(),
    ) {
        let dst = Yuv444Planes::new(dst_w, dst_h);
        let rect = PlaneRect { left, top, right, bottom };

        let mut out = Vec::new();
        let clipped = convert_rect_to_rgba(&dst, rect, &mut out);

        proptest::prop_assert_eq!(out.len(), clipped.width() * clipped.height() * 4);
    }
}

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(200))]

    /// `apply_main_view` never writes outside its even-expanded, `min(dst, main)`-clipped rect.
    /// `Yuv444Planes::new`'s defaults (`Y = 0`, `U = V = 128`) work as sentinels because
    /// `filled_yuv420` never produces those values.
    #[test]
    fn apply_main_view_never_touches_outside_the_expanded_rect(
        dst_w in dim_strategy(), dst_h in dim_strategy(),
        main_w in dim_strategy(), main_h in dim_strategy(),
        left in edge_strategy(), top in edge_strategy(), right in edge_strategy(), bottom in edge_strategy(),
    ) {
        let main = filled_yuv420(main_w, main_h, 3);
        let mut dst = Yuv444Planes::new(dst_w, dst_h);
        let rect = PlaneRect { left, top, right, bottom };

        apply_main_view(&mut dst, &main.view(), rect);

        let touched = expected_main_touched_rect(rect, dst_w, dst_h, main_w, main_h);
        for y in 0..dst_h {
            for x in 0..dst_w {
                let inside = x >= touched.left && x < touched.right && y >= touched.top && y < touched.bottom;
                if !inside {
                    let idx = y * dst_w + x;
                    proptest::prop_assert_eq!(dst.y()[idx], 0, "y outside rect at ({}, {})", x, y);
                    proptest::prop_assert_eq!(dst.u()[idx], 128, "u outside rect at ({}, {})", x, y);
                    proptest::prop_assert_eq!(dst.v()[idx], 128, "v outside rect at ({}, {})", x, y);
                }
            }
        }
    }
}

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(200))]

    /// `apply_aux_view` never writes outside `rect` clipped to `dst` (it does not expand it).
    #[test]
    fn apply_aux_view_never_touches_outside_the_rect(
        dst_w in dim_strategy(), dst_h in dim_strategy(),
        aux_w in dim_strategy(), aux_h in dim_strategy(),
        left in edge_strategy(), top in edge_strategy(), right in edge_strategy(), bottom in edge_strategy(),
        layout in layout_strategy(),
    ) {
        let aux = filled_yuv420(aux_w, aux_h, 4);
        let mut dst = Yuv444Planes::new(dst_w, dst_h);
        let rect = PlaneRect { left, top, right, bottom };

        apply_aux_view(&mut dst, &aux.view(), rect, layout);

        let touched = rect.clip(dst_w, dst_h);
        for y in 0..dst_h {
            for x in 0..dst_w {
                let inside = x >= touched.left && x < touched.right && y >= touched.top && y < touched.bottom;
                if !inside {
                    let idx = y * dst_w + x;
                    proptest::prop_assert_eq!(dst.u()[idx], 128, "u outside rect at ({}, {})", x, y);
                    proptest::prop_assert_eq!(dst.v()[idx], 128, "v outside rect at ({}, {})", x, y);
                }
            }
        }
    }
}

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(100))]

    /// Layout property under odd sizes: build a reference picture `src` by
    /// applying LCG-filled main/aux views over its full frame, then split it
    /// back into main/aux (`split::main_view`/`split::aux_view`) and apply
    /// those into a fresh `dst` of the same size. Every position with an odd
    /// `x` (never touched by the main view's block replication, and always
    /// aux-carried in both layouts, per MS-RDPEGFX B4/B6-B7 for V1 and
    /// B4/B5 for V2) inside `min(dst, aux)` must equal `src` exactly, as
    /// must every position with an odd `y` for V1 (whole rows are
    /// aux-carried per B4/B5) and, for V2, whenever the frame is wide
    /// enough (`aux_w >= 4`) for its B6-B9 chroma detail to exist at all
    /// (`quarter_w = aux_w / 4`) - aux-sourced samples are copied, not
    /// colour-converted, so there is no rounding involved.
    #[test]
    fn split_and_apply_round_trip_is_exact_at_odd_positions(
        src_w in 2usize..=33, src_h in 2usize..=33,
        main_w in dim_strategy(), main_h in dim_strategy(),
        aux_w in dim_strategy(), aux_h in dim_strategy(),
        layout in layout_strategy(),
    ) {
        let main0 = filled_yuv420(main_w, main_h, 5);
        let aux0 = filled_yuv420(aux_w, aux_h, 6);

        let mut src = Yuv444Planes::new(src_w, src_h);
        apply_main_view(&mut src, &main0.view(), full_rect(src_w, src_h));
        apply_aux_view(&mut src, &aux0.view(), full_rect(src_w, src_h), layout);

        let main1 = split::main_view(&src);
        let aux1 = split::aux_view(&src, layout);

        let mut dst = Yuv444Planes::new(src_w, src_h);
        apply_main_view(&mut dst, &main1.view(), full_rect(src_w, src_h));
        apply_aux_view(&mut dst, &aux1.view(), full_rect(src_w, src_h), layout);

        let min_w = dst.width().min(usize::try_from(aux1.width).unwrap());
        let min_h = dst.height().min(usize::try_from(aux1.height).unwrap());
        // V2's B6-B9 chroma detail packs 2 chroma columns' worth of detail
        // per 4 luma columns (`quarter_w = aux_w / 4`): an (even x, odd y)
        // position is aux-exact only where its quarter-column actually
        // exists, which a width that is not a multiple of 4 does not
        // guarantee for every such position (a ragged last group).
        let quarter_w = src_w / 4;
        for y in 0..min_h {
            for x in 0..min_w {
                let even_x_odd_y_is_aux_exact = match layout {
                    ChromaLayout::V1 => true,
                    ChromaLayout::V2 => {
                        let gx = if x % 4 == 0 { x / 4 } else { x.saturating_sub(2) / 4 };
                        gx < quarter_w
                    }
                };
                let checked = x % 2 == 1 || (y % 2 == 1 && even_x_odd_y_is_aux_exact);
                if checked {
                    let idx = y * src_w + x;
                    proptest::prop_assert_eq!(dst.u()[idx], src.u()[idx], "u mismatch at ({}, {})", x, y);
                    proptest::prop_assert_eq!(dst.v()[idx], src.v()[idx], "v mismatch at ({}, {})", x, y);
                }
            }
        }
    }
}
