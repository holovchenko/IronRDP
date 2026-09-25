//! Client-side surface compositor for the Graphics Pipeline (EGFX).
//!
//! MS-RDPEGFX surfaces are persistent pixel canvases: the server creates them,
//! writes decoded bitmaps into sub-rectangles, fills and scrolls regions, caches
//! tiles, and maps surfaces onto the graphics output. A client that only decodes
//! `WireToSurface1` or `WireToSurface2` bitmaps and forwards the remaining surface commands renders
//! incorrectly against any server that uses them (GNOME Remote Desktop and
//! Windows both do).
//!
//! [`Compositor`] holds the surface and cache pixel state, applies each command
//! into it, and accumulates the regions of the *output* that changed. Updates are
//! committed per logical frame (`EndFrame`) and drained by the consumer, so a
//! partially-built frame is never observed.
//!
//! All canvases are stored as tightly-packed RGBA8888, matching
//! [`BitmapUpdate`](crate::client::BitmapUpdate) and the session's decoded image,
//! so a drained region is ready to blit without conversion.

use std::collections::{BTreeMap, BTreeSet};

use ironrdp_pdu::geometry::ExclusiveRectangle;
use tracing::{debug, trace};

use crate::client::{CacheImportOutcome, CacheImportTile, MAX_CACHE_IMPORT_BYTES, MAX_CACHE_IMPORT_ENTRIES};
use crate::pdu::{CacheEntryMetadata, Color, Point};

const BYTES_PER_PIXEL: usize = 4;

/// Upper bound on a surface edge, in pixels.
///
/// Larger than any real display dimension (8K is 7680x4320); its purpose is to
/// bound the per-surface allocation so a malformed or hostile `CreateSurface`
/// cannot request an unbounded buffer.
const MAX_SURFACE_DIM: u16 = 16384;

/// Upper bound on the pixel bytes the compositor will hold at once, across every
/// surface and every cache slot.
///
/// [`MAX_SURFACE_DIM`] bounds one allocation, which is not the property that
/// matters: surfaces are keyed by `u16` and cache slots likewise, so a peer that
/// stays under the per-edge limit can still ask for tens of thousands of them.
/// This bounds the total instead.
///
/// 256 MiB is chosen against what a real session needs rather than as a round
/// number. A 4K surface is 3840*2160*4, about 33 MiB; MS-RDPEGFX servers
/// legitimately keep a handful of those plus an offscreen cache, so this leaves
/// room for roughly eight full-screen 4K canvases while making the 1 GiB
/// single-surface case (16384*16384*4) impossible to reach.
const MAX_COMPOSITOR_BYTES: usize = 256 * 1024 * 1024;

/// A rectangular region of the graphics output whose pixels changed within a frame.
///
/// `region` is in output space (after a surface-to-output mapping), using the
/// egfx exclusive-rectangle convention (`right`/`bottom` are one past the edge).
/// `data` is tightly-packed RGBA8888, row-major, exactly
/// `region.width() * region.height() * 4` bytes.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct OutputUpdate {
    /// Changed region, in output-buffer coordinates.
    pub region: ExclusiveRectangle,
    /// RGBA8888 pixels for `region`, row-major.
    pub data: Vec<u8>,
}

/// A surface-space rectangle that changed within the frame being built.
///
/// Regions rather than pixels: one PDU may carry 65,535 rectangles or destination
/// points, so copying eagerly at each command lets a small wire payload amplify
/// into an unbounded queue of full-surface buffers. Reading the pixels once at
/// `EndFrame` also means the consumer receives the frame's final state instead of
/// intermediate states it would immediately paint over.
#[derive(Debug, Clone)]
struct DirtyRegion {
    surface_id: u16,
    /// Already clipped to the surface, so materialization can trust it.
    rect: ExclusiveRectangle,
}

/// A persistent client surface: a pixel canvas and its optional output mapping.
#[derive(Debug)]
struct Surface {
    width: u16,
    height: u16,
    /// RGBA8888, `width * height * 4` bytes.
    data: Vec<u8>,
    /// Output mapping if this surface is visible, else `None`.
    mapping: Option<OutputMapping>,
}

/// A surface-to-output mapping, including the scaled output dimensions.
#[derive(Debug, Clone, Copy)]
struct OutputMapping {
    origin: (u32, u32),
    target_width: u32,
    target_height: u32,
}

impl Surface {
    /// A mutable view of this surface's pixel buffer, for the blit/fill helpers.
    fn canvas_mut(&mut self) -> CanvasMut<'_> {
        CanvasMut {
            data: &mut self.data,
            width: self.width,
            height: self.height,
        }
    }
}

/// A mutable view of an RGBA8888 canvas: its pixel buffer and dimensions.
struct CanvasMut<'a> {
    data: &'a mut [u8],
    width: u16,
    height: u16,
}

/// A cached bitmap tile (MS-RDPEGFX bitmap cache, 2.2.2.10 / 2.2.2.11).
#[derive(Debug)]
struct CachedTile {
    width: u16,
    height: u16,
    /// RGBA8888, `width * height * 4` bytes.
    data: Vec<u8>,
}

/// A tile held for a pending `CacheImportOffer`, already charged against the budget.
#[derive(Debug)]
struct StagedTile {
    cache_key: u64,
    tile: CachedTile,
}

/// Client-side EGFX surface compositor.
///
/// Applies surface commands into persistent RGBA8888 buffers and accumulates the
/// output regions that change. The
/// [`GraphicsPipelineClient`](crate::client::GraphicsPipelineClient) feeds it each
/// command and drains completed frames via [`drain_output`](Self::drain_output).
#[derive(Debug, Default)]
pub(crate) struct Compositor {
    surfaces: BTreeMap<u16, Surface>,
    cache: BTreeMap<u16, CachedTile>,
    /// Tiles offered in `CacheImportOffer`, in offer order, awaiting the reply.
    /// `None` once the reply has moved the tile into a slot.
    staged_import: Vec<Option<StagedTile>>,
    /// Pixel bytes currently held across `surfaces`, `cache`, `staged_import` and
    /// `ready`, charged against [`MAX_COMPOSITOR_BYTES`]. Kept as a running total
    /// rather than recomputed so the check before an allocation stays O(1).
    allocated_bytes: usize,
    output_width: u16,
    output_height: u16,
    /// Regions dirtied by the frame currently being built (between
    /// `StartFrame`/`EndFrame`), materialized into `ready` at commit.
    frame: Vec<DirtyRegion>,
    /// Deltas from completed frames, awaiting drain.
    ready: Vec<OutputUpdate>,
}

impl Compositor {
    /// Handle `ResetGraphics`: resize the graphics output and discard pending output.
    ///
    /// Per MS-RDPEGFX 3.3.5.14, the only client requirement on receiving
    /// `RDPGFX_RESET_GRAPHICS_PDU` is to "resize the Graphics Output Buffer ...
    /// ADM element"; 2.2.2.14 likewise describes it only as changing the output
    /// buffer's dimensions and monitor layout. Neither mentions surfaces or the
    /// bitmap cache, and cache slots are removed only by
    /// `RDPGFX_EVICT_CACHE_ENTRY_PDU` (3.3.5.8). FreeRDP matches this:
    /// `rdpgfx_recv_reset_graphics_pdu` never touches `CacheSlots` or the surface
    /// table, and `gdi_ResetGraphics` resizes the desktop without freeing any
    /// surface. A Windows server relies on this: after a resize it sends
    /// `ResetGraphics` + `CreateSurface` + `StartFrame` and then keeps painting
    /// through `CacheToSurface` with slots it filled before the reset, without
    /// re-sending them, so dropping the cache (or the surfaces) here breaks
    /// rendering until reconnect.
    ///
    /// `frame` and `ready` are discarded ONLY when the output dimensions actually
    /// change. The reason for discarding them is that those deltas were clipped
    /// against the previous output, so painting them into the new one repaints
    /// stale pixels and, after a shrink, addresses a region the new output no
    /// longer contains — and every word of that is about a *different* output.
    ///
    /// A same-size reset is not a different output. Windows sends one whenever it
    /// rebuilds its own surfaces, and the consumer side agrees: the session's
    /// `apply_reset_graphics` returns without touching the image when the
    /// dimensions are unchanged, deliberately, to avoid a blank flash. Clearing
    /// the queue there loses every delta of the payload that carried the reset —
    /// the `EndFrame` immediately before it commits into `ready`, the reset wipes
    /// `ready`, `drain_output` returns nothing, and the consumer keeps an image
    /// that is one frame stale in exactly the region that frame repainted. The
    /// loss is silent and never repaired: the server considers those pixels sent.
    pub(crate) fn reset(&mut self, width: u32, height: u32) {
        // Compared before clamping: two different oversized dimensions can both
        // clamp to `u16::MAX`, and comparing the clamped values would then wrongly
        // treat that as a same-size reset.
        let resized = (width, height) != (u32::from(self.output_width), u32::from(self.output_height));
        let width = u16::try_from(width).unwrap_or(u16::MAX);
        let height = u16::try_from(height).unwrap_or(u16::MAX);
        self.output_width = width;
        self.output_height = height;
        if !resized {
            return;
        }
        let released = self.ready.iter().map(|update| update.data.len()).sum::<usize>()
            + self.frame.len() * size_of::<DirtyRegion>();
        self.frame.clear();
        self.ready.clear();
        // Release only what was discarded: surfaces and the cache survive, so
        // their pixel bytes stay charged.
        self.allocated_bytes = self.allocated_bytes.saturating_sub(released);
    }

    /// Reserve `len` pixel bytes, or refuse if that would exceed the budget.
    ///
    /// Refusing rather than erroring keeps the existing contract for allocations
    /// the compositor declines: the surface or slot simply does not exist, and
    /// commands targeting it become no-ops.
    fn charge(&mut self, len: usize) -> bool {
        match self.allocated_bytes.checked_add(len) {
            Some(total) if total <= MAX_COMPOSITOR_BYTES => {
                self.allocated_bytes = total;
                true
            }
            _ => {
                debug!(
                    len,
                    allocated = self.allocated_bytes,
                    budget = MAX_COMPOSITOR_BYTES,
                    "compositor allocation refused: budget exhausted"
                );
                false
            }
        }
    }

    /// Release a charge taken by [`Self::charge`].
    fn release(&mut self, len: usize) {
        // Saturating rather than wrapping: an accounting slip must not underflow
        // into a huge budget that disables the limit entirely.
        self.allocated_bytes = self.allocated_bytes.saturating_sub(len);
    }

    /// Handle `CreateSurface`: allocate a zeroed RGBA8888 canvas.
    ///
    /// Surfaces beyond [`MAX_SURFACE_DIM`] on either edge are skipped rather than
    /// allocated; subsequent commands targeting them become no-ops (the metadata
    /// tracked by the client is unaffected).
    pub(crate) fn create_surface(&mut self, id: u16, width: u16, height: u16) {
        if width == 0 || height == 0 || width > MAX_SURFACE_DIM || height > MAX_SURFACE_DIM {
            return;
        }
        let len = usize::from(width) * usize::from(height) * BYTES_PER_PIXEL;

        // Release first: `BTreeMap::insert` drops any surface already under this id,
        // so re-creating one must not charge twice for the same slot.
        if let Some(previous) = self.surfaces.remove(&id) {
            self.release(previous.data.len());
        }

        if !self.charge(len) {
            return;
        }

        self.surfaces.insert(
            id,
            Surface {
                width,
                height,
                data: vec![0; len],
                mapping: None,
            },
        );
    }

    /// Handle `DeleteSurface`.
    pub(crate) fn delete_surface(&mut self, id: u16) {
        if let Some(surface) = self.surfaces.remove(&id) {
            self.release(surface.data.len());
        }
    }

    /// Handle `MapSurfaceToOutput`: record the mapping and make the surface's
    /// current contents visible at `(origin_x, origin_y)`.
    pub(crate) fn map_surface(&mut self, id: u16, origin_x: u32, origin_y: u32) {
        let Some((width, height)) = self.surfaces.get(&id).map(|surface| (surface.width, surface.height)) else {
            return;
        };

        self.map_surface_scaled(id, origin_x, origin_y, u32::from(width), u32::from(height));
    }

    /// Handle `MapSurfaceToScaledOutput`: record the scaled mapping and make the
    /// surface's current contents visible at `(origin_x, origin_y)`.
    pub(crate) fn map_surface_scaled(
        &mut self,
        id: u16,
        origin_x: u32,
        origin_y: u32,
        target_width: u32,
        target_height: u32,
    ) {
        let mapping = OutputMapping {
            origin: (origin_x, origin_y),
            target_width,
            target_height,
        };
        let Some(surface) = self.surfaces.get_mut(&id) else {
            return;
        };
        surface.mapping = Some(mapping);
        let (w, h) = (surface.width, surface.height);
        // The whole surface becomes visible at its new origin.
        self.record_dirty(id, 0, 0, w, h);
    }

    /// Write a decoded RGBA8888 bitmap into a surface sub-rectangle (the output of
    /// a `WireToSurface1` or `WireToSurface2` decode) and record the resulting output delta.
    ///
    /// `rgba` is expected to be `dest.width() * dest.height() * 4` bytes; short or
    /// long buffers are handled defensively by the row-wise blit.
    pub(crate) fn apply_bitmap(&mut self, surface_id: u16, dest: &ExclusiveRectangle, rgba: &[u8]) {
        let x = dest.left;
        let y = dest.top;
        let w = dest.right.saturating_sub(dest.left);
        let h = dest.bottom.saturating_sub(dest.top);
        if let Some(surface) = self.surfaces.get_mut(&surface_id) {
            blit_region(surface.canvas_mut(), x, y, w, h, rgba);
        }
        self.record_dirty(surface_id, x, y, w, h);
    }

    /// Fill each rectangle of a surface with a solid, opaque color (`SolidFill`).
    pub(crate) fn solid_fill(&mut self, surface_id: u16, color: &Color, rects: &[ExclusiveRectangle]) {
        let px = [color.r, color.g, color.b, 0xFF];
        for rect in rects {
            let x = rect.left;
            let y = rect.top;
            let w = rect.right.saturating_sub(rect.left);
            let h = rect.bottom.saturating_sub(rect.top);
            if let Some(surface) = self.surfaces.get_mut(&surface_id) {
                fill_region(surface.canvas_mut(), x, y, w, h, px);
            }
            self.record_dirty(surface_id, x, y, w, h);
        }
    }

    /// Copy a rectangle from one surface to points on another (`SurfaceToSurface`;
    /// a copy or an on-screen scroll when source and destination are the same
    /// surface). The source region is read once; each destination point receives a
    /// copy.
    pub(crate) fn surface_to_surface(
        &mut self,
        src_id: u16,
        dst_id: u16,
        src_rect: &ExclusiveRectangle,
        dst_points: &[Point],
    ) {
        // Read the source region into an owned tile first, so a scroll within one
        // surface (src_id == dst_id) reads pre-move pixels and never aliases.
        let Some(src) = self.surfaces.get(&src_id) else {
            return;
        };
        // Clamp the copy to the source surface: the tile is exactly the valid
        // source pixels, so a malformed rectangle can't drive an oversized
        // allocation. It is deliberately not charged against the budget, since it
        // can be no larger than the source surface, which is, bounding the peak at
        // two surfaces rather than leaving it open.
        let w = src_rect
            .right
            .saturating_sub(src_rect.left)
            .min(src.width.saturating_sub(src_rect.left));
        let h = src_rect
            .bottom
            .saturating_sub(src_rect.top)
            .min(src.height.saturating_sub(src_rect.top));
        let tile = copy_region(&src.data, src.width, src_rect.left, src_rect.top, w, h);

        if let Some(dst) = self.surfaces.get_mut(&dst_id) {
            for point in dst_points {
                blit_region(dst.canvas_mut(), point.x, point.y, w, h, &tile);
            }
        }
        for point in dst_points {
            self.record_dirty(dst_id, point.x, point.y, w, h);
        }
    }

    /// Copy a surface rectangle into a cache slot (`SurfaceToCache`). The cache is
    /// off-screen, so this produces no output delta.
    pub(crate) fn surface_to_cache(&mut self, surface_id: u16, cache_slot: u16, src_rect: &ExclusiveRectangle) {
        // Size the tile in a scope that ends the surface borrow, so the budget calls
        // below can take `&mut self` while the charge still precedes the allocation.
        let (w, h) = {
            let Some(surface) = self.surfaces.get(&surface_id) else {
                return;
            };
            // Clamp to the source surface so a malformed rectangle can't drive an
            // oversized allocation.
            let w = src_rect
                .right
                .saturating_sub(src_rect.left)
                .min(surface.width.saturating_sub(src_rect.left));
            let h = src_rect
                .bottom
                .saturating_sub(src_rect.top)
                .min(surface.height.saturating_sub(src_rect.top));
            (w, h)
        };

        if w == 0 || h == 0 {
            // The clamp above can collapse the tile to nothing when the source
            // rectangle starts outside the surface. The slot is then emptied below
            // and every later `CacheToSurface` on it is a silent no-op.
            debug!(
                surface_id,
                cache_slot,
                src_left = src_rect.left,
                src_top = src_rect.top,
                "SurfaceToCache clamped to an empty tile - the slot will paint nothing"
            );
        }

        let len = usize::from(w) * usize::from(h) * BYTES_PER_PIXEL;

        // Release first, for the same reason as `create_surface`: `BTreeMap::insert`
        // drops whatever occupied the slot, so caching repeatedly into one slot must
        // not accumulate charge.
        if let Some(previous) = self.cache.remove(&cache_slot) {
            self.release(previous.data.len());
        }

        // Charged before `copy_region`, not around the insert: the copy is the
        // allocation, so checking afterwards would still let a peer drive the
        // transient the budget exists to prevent. The cache shares the surfaces'
        // budget because both draw on the same process memory.
        if !self.charge(len) {
            return;
        }

        // Nothing above removes a surface, so this lookup cannot fail. Handle it by
        // releasing rather than unwrapping: a charge without a matching insert would
        // leak budget permanently, which is a worse failure than a skipped tile.
        let Some(surface) = self.surfaces.get(&surface_id) else {
            self.release(len);
            return;
        };

        let data = copy_region(&surface.data, surface.width, src_rect.left, src_rect.top, w, h);
        self.cache.insert(
            cache_slot,
            CachedTile {
                width: w,
                height: h,
                data,
            },
        );
    }

    /// Render a cached tile onto a surface at each destination point
    /// (`CacheToSurface`).
    pub(crate) fn cache_to_surface(&mut self, cache_slot: u16, surface_id: u16, dst_points: &[Point]) {
        let Some(tile) = self.cache.get(&cache_slot) else {
            // Silent before: the server paints from a slot this client never filled
            // (or emptied), nothing lands, and no dirty rectangle is recorded — the
            // region simply keeps whatever it held. Diagnosing that needs a line.
            debug!(
                cache_slot,
                surface_id, "CacheToSurface on an empty cache slot - nothing painted"
            );
            return;
        };
        let (w, h) = (tile.width, tile.height);

        // `cache` and `surfaces` are disjoint fields, so the tile stays borrowed
        // across the blits rather than being copied to satisfy the borrow checker.
        // Recording is what needs all of `&mut self`, so it follows in its own loop
        // once the tile borrow has ended.
        if let Some(surface) = self.surfaces.get_mut(&surface_id) {
            for point in dst_points {
                blit_region(surface.canvas_mut(), point.x, point.y, w, h, &tile.data);
            }
        }
        for point in dst_points {
            self.record_dirty(surface_id, point.x, point.y, w, h);
        }
    }

    /// Evict a cache slot (`EvictCacheEntry`).
    pub(crate) fn evict_cache_entry(&mut self, cache_slot: u16) {
        if let Some(tile) = self.cache.remove(&cache_slot) {
            self.release(tile.data.len());
        }
    }

    /// Hold `tiles` for a `CacheImportOffer` and return the offer's entries, in order.
    ///
    /// Every staged tile is charged here, before the offer exists. A slot the server
    /// believes filled but the client left empty paints nothing on every later
    /// `CacheToSurface` (the stale-mosaic class), and no PDU lets the client withdraw an
    /// entry the server accepted — so the budget must say no now, while saying no only
    /// shortens the offer. Staging stops at the first tile the entry cap, the byte cap or
    /// the budget refuses, keeping the offer a prefix of `tiles` (the reply answers a
    /// prefix too, MS-RDPEGFX 2.2.2.17). Malformed tiles (zero-sized, or `data` not
    /// `width * height * 4` bytes) and repeated keys are skipped. Anything staged
    /// earlier is released first.
    pub(crate) fn stage_import(&mut self, tiles: Vec<CacheImportTile>) -> Vec<CacheEntryMetadata> {
        self.discard_import();
        let mut entries = Vec::new();
        let mut keys = BTreeSet::new();
        let mut total = 0usize;
        for tile in tiles {
            if entries.len() == MAX_CACHE_IMPORT_ENTRIES {
                break;
            }
            let len = usize::from(tile.width) * usize::from(tile.height) * BYTES_PER_PIXEL;
            if len == 0 || tile.data.len() != len || !keys.insert(tile.cache_key) {
                debug!(
                    cache_key = tile.cache_key,
                    "skipping a malformed or repeated cache import tile"
                );
                continue;
            }
            #[expect(
                clippy::arithmetic_side_effects,
                reason = "total is bounded by MAX_CACHE_IMPORT_BYTES via the check below every iteration"
            )]
            let would_total = total + len;
            if would_total > MAX_CACHE_IMPORT_BYTES || !self.charge(len) {
                break;
            }
            total = would_total;
            let Ok(bitmap_len) = u32::try_from(len) else {
                // Unreachable: `len` is bounded by MAX_CACHE_IMPORT_BYTES. Release rather
                // than unwrap, so an accounting slip cannot leak budget.
                self.release(len);
                break;
            };
            entries.push(CacheEntryMetadata {
                cache_key: tile.cache_key,
                bitmap_len,
            });
            self.staged_import.push(Some(StagedTile {
                cache_key: tile.cache_key,
                tile: CachedTile {
                    width: tile.width,
                    height: tile.height,
                    data: tile.data,
                },
            }));
        }
        entries
    }

    /// Apply a `CacheImportReply`: `cache_slots[i]` is the slot the server assigned to
    /// offer entry `i`, and `0` means "not imported" (FreeRDP's `gdi_ImportCacheEntry`).
    ///
    /// A named tile moves into its slot carrying the charge taken at staging, so
    /// filling cannot be refused here. A slot named with no staged tile behind it — an
    /// index past the offer, a reply with nothing staged, a slot named twice — cannot be
    /// filled by anything the client holds and is reported in `unfilled_slots`. Every
    /// staged tile the reply does not name is released.
    pub(crate) fn commit_import(&mut self, cache_slots: &[u16]) -> CacheImportOutcome {
        let mut staged = core::mem::take(&mut self.staged_import);
        let mut outcome = CacheImportOutcome {
            offered: staged.len(),
            ..CacheImportOutcome::default()
        };
        let mut assigned = BTreeSet::new();
        for (index, &slot) in cache_slots.iter().enumerate() {
            if slot == 0 {
                continue;
            }
            match staged.get_mut(index).and_then(Option::take) {
                Some(StagedTile { cache_key, tile }) if assigned.insert(slot) => {
                    if let Some(previous) = self.cache.remove(&slot) {
                        self.release(previous.data.len());
                    }
                    self.cache.insert(slot, tile);
                    outcome.imported.push((cache_key, slot));
                }
                Some(StagedTile { tile, .. }) => {
                    // Two entries claim one slot; which one the server believes it holds
                    // is unknowable, so the slot counts as unfilled.
                    self.release(tile.data.len());
                    outcome.unfilled_slots.push(slot);
                }
                None => outcome.unfilled_slots.push(slot),
            }
        }
        for leftover in staged.into_iter().flatten() {
            self.release(leftover.tile.data.len());
        }
        outcome
    }

    /// Release every staged import tile (channel closed, or a new offer replaces it).
    pub(crate) fn discard_import(&mut self) {
        for staged in core::mem::take(&mut self.staged_import).into_iter().flatten() {
            self.release(staged.tile.data.len());
        }
    }

    /// The tile in `cache_slot`, as `(width, height, rgba)`.
    pub(crate) fn cache_tile(&self, cache_slot: u16) -> Option<(u16, u16, &[u8])> {
        self.cache
            .get(&cache_slot)
            .map(|tile| (tile.width, tile.height, tile.data.as_slice()))
    }

    /// Commit the current frame's deltas (`EndFrame`), making them drainable.
    ///
    /// The pixels are read here rather than when each command ran, so a surface
    /// deleted or remapped mid-frame contributes nothing: its output region is
    /// repainted by whatever maps there next.
    pub(crate) fn end_frame(&mut self) {
        let frame = core::mem::take(&mut self.frame);
        // Released before materializing so the pixel copies can spend the budget the
        // metadata was holding.
        self.release(frame.len() * size_of::<DirtyRegion>());
        for dirty in frame {
            self.materialize(&dirty);
        }
    }

    /// Take all committed output updates, leaving the pending set empty.
    pub(crate) fn drain_output(&mut self) -> Vec<OutputUpdate> {
        let drained = core::mem::take(&mut self.ready);
        let released = drained.iter().map(|update| update.data.len()).sum();
        self.release(released);
        drained
    }

    /// Copy the pixels for one dirty region into the drainable queue.
    ///
    /// The region arrives clipped to its surface and is mapped against the output
    /// mapping in force now rather than the one active when it was drawn.
    fn materialize(&mut self, dirty: &DirtyRegion) {
        let Some((mapping, source_width, source_height)) = self
            .surfaces
            .get(&dirty.surface_id)
            .and_then(|surface| surface.mapping.map(|mapping| (mapping, surface.width, surface.height)))
        else {
            return;
        };
        if mapping.target_width == 0 || mapping.target_height == 0 {
            return;
        }

        let mapped_left = scaled_edge(dirty.rect.left, source_width, mapping.target_width);
        let mapped_top = scaled_edge(dirty.rect.top, source_height, mapping.target_height);
        let mapped_right = scaled_edge(dirty.rect.right, source_width, mapping.target_width);
        let mapped_bottom = scaled_edge(dirty.rect.bottom, source_height, mapping.target_height);

        // Saturating adds and output clipping bound the output rectangle before
        // allocating its RGBA8888 pixels.
        let left = mapping
            .origin
            .0
            .saturating_add(mapped_left)
            .min(u32::from(self.output_width));
        let top = mapping
            .origin
            .1
            .saturating_add(mapped_top)
            .min(u32::from(self.output_height));
        let right = mapping
            .origin
            .0
            .saturating_add(mapped_right)
            .min(u32::from(self.output_width));
        let bottom = mapping
            .origin
            .1
            .saturating_add(mapped_bottom)
            .min(u32::from(self.output_height));
        if right <= left || bottom <= top {
            return;
        }

        let left = u16::try_from(left).unwrap_or(u16::MAX);
        let top = u16::try_from(top).unwrap_or(u16::MAX);
        let right = u16::try_from(right).unwrap_or(u16::MAX);
        let bottom = u16::try_from(bottom).unwrap_or(u16::MAX);
        let w = right - left;
        let h = bottom - top;
        let Some(len) = pixel_data_len(w, h) else {
            return;
        };
        if !self.charge(len) {
            return;
        }
        let Some(surface) = self.surfaces.get(&dirty.surface_id) else {
            self.release(len);
            return;
        };
        let output_rect = ExclusiveRectangle {
            left,
            top,
            right,
            bottom,
        };
        let data =
            if mapping.target_width == u32::from(source_width) && mapping.target_height == u32::from(source_height) {
                copy_region(&surface.data, source_width, dirty.rect.left, dirty.rect.top, w, h)
            } else {
                copy_scaled_region(&surface.data, (source_width, source_height), mapping, &output_rect)
            };
        self.ready.push(OutputUpdate {
            region: output_rect,
            data,
        });
    }

    /// Record that a surface-space rectangle changed, for materialization at
    /// `EndFrame`.
    ///
    /// A no-op if the surface is gone. The rectangle is clipped to the surface here,
    /// while the surface is already in hand, so a malformed server rectangle can
    /// shrink the update to empty but never reads out of range.
    fn record_dirty(&mut self, surface_id: u16, sx: u16, sy: u16, sw: u16, sh: u16) {
        let Some(surface) = self.surfaces.get(&surface_id) else {
            return;
        };
        let left = sx.min(surface.width);
        let top = sy.min(surface.height);
        let right = sx.saturating_add(sw).min(surface.width);
        let bottom = sy.saturating_add(sh).min(surface.height);
        if right <= left || bottom <= top {
            return;
        }
        let rect = ExclusiveRectangle {
            left,
            top,
            right,
            bottom,
        };

        // Repeated paints of one region are the cheap half of the amplification
        // deferral addresses: a PDU naming the same rectangle 65,535 times collapses
        // to a single entry. Comparing against only the previous entry keeps this
        // O(1); it is a repeat filter, not region coalescing.
        if let Some(last) = self.frame.last() {
            if last.surface_id == surface_id && covers(&last.rect, &rect) {
                trace!(surface_id, ?rect, "dirty already covered by the previous entry");
                return;
            }
        }

        // The filter above only collapses consecutive repeats, so alternating
        // rectangles still land one entry each and the frame stays open until the
        // peer sends `EndFrame`. Charging the entry puts that growth under the same
        // ceiling as the pixel buffers: `RDPGFX_POINT16` is four bytes on the wire
        // against ten resident here, so an uncharged queue would let a peer trade
        // its own bandwidth for 2.5 times as much of the client's memory.
        if !self.charge(size_of::<DirtyRegion>()) {
            return;
        }
        trace!(surface_id, ?rect, "dirty recorded");
        self.frame.push(DirtyRegion { surface_id, rect });
    }
}

/// Whether `outer` fully contains `inner`, both in the exclusive convention.
fn covers(outer: &ExclusiveRectangle, inner: &ExclusiveRectangle) -> bool {
    outer.left <= inner.left && outer.top <= inner.top && outer.right >= inner.right && outer.bottom >= inner.bottom
}

/// Return the target-space edge for a source-space edge with nearest-neighbor
/// scaling. Rounding up means a dirty source rectangle covers every output pixel
/// that samples from it.
fn scaled_edge(source_edge: u16, source_length: u16, target_length: u32) -> u32 {
    if source_length == 0 {
        return 0;
    }

    let scaled = (u64::from(source_edge) * u64::from(target_length)).div_ceil(u64::from(source_length));
    u32::try_from(scaled).unwrap_or(u32::MAX)
}

/// Return the byte length of a tightly-packed RGBA8888 rectangle.
fn pixel_data_len(width: u16, height: u16) -> Option<usize> {
    usize::from(width)
        .checked_mul(usize::from(height))?
        .checked_mul(BYTES_PER_PIXEL)
}

/// Copy a sub-rectangle out of an RGBA8888 canvas into a tightly-packed buffer.
///
/// Rows that fall outside the source (short buffer, out-of-range rect) are padded
/// with zeroes so the result is always `w * h * 4` bytes. Callers must bound
/// `w`/`h` to the source surface, since the returned buffer is allocated at that
/// size regardless of how much source actually exists.
fn copy_region(src: &[u8], src_width: u16, x: u16, y: u16, w: u16, h: u16) -> Vec<u8> {
    let src_stride = usize::from(src_width) * BYTES_PER_PIXEL;
    let row_bytes = usize::from(w) * BYTES_PER_PIXEL;
    let mut out = Vec::with_capacity(row_bytes * usize::from(h));
    for row in 0..usize::from(h) {
        let src_y = usize::from(y) + row;
        let start = src_y * src_stride + usize::from(x) * BYTES_PER_PIXEL;
        let end = start + row_bytes;
        if end <= src.len() {
            out.extend_from_slice(&src[start..end]);
        } else {
            out.resize(out.len() + row_bytes, 0);
        }
    }
    out
}

/// Scale a target-space rectangle from a source RGBA8888 canvas using nearest
/// neighbor sampling.
fn copy_scaled_region(
    src: &[u8],
    (src_width, src_height): (u16, u16),
    mapping: OutputMapping,
    output_rect: &ExclusiveRectangle,
) -> Vec<u8> {
    let width = output_rect.right.saturating_sub(output_rect.left);
    let height = output_rect.bottom.saturating_sub(output_rect.top);
    let Some(len) = pixel_data_len(width, height) else {
        return Vec::new();
    };
    let mut out = vec![0; len];
    let target_left = u32::from(output_rect.left).saturating_sub(mapping.origin.0);
    let target_top = u32::from(output_rect.top).saturating_sub(mapping.origin.1);
    let source_column_offsets = (0..width)
        .map(|x| {
            usize::from(
                u16::try_from(
                    (u64::from(target_left) + u64::from(x)) * u64::from(src_width) / u64::from(mapping.target_width),
                )
                .unwrap_or(u16::MAX),
            ) * BYTES_PER_PIXEL
        })
        .collect::<Vec<_>>();

    for y in 0..height {
        let source_y = u16::try_from(
            (u64::from(target_top) + u64::from(y)) * u64::from(src_height) / u64::from(mapping.target_height),
        )
        .unwrap_or(u16::MAX);
        let source_row_offset = usize::from(source_y) * usize::from(src_width) * BYTES_PER_PIXEL;
        for x in 0..width {
            let source_offset = source_row_offset + source_column_offsets[usize::from(x)];
            let target_offset = (usize::from(y) * usize::from(width) + usize::from(x)) * BYTES_PER_PIXEL;
            if source_offset + BYTES_PER_PIXEL <= src.len() {
                out[target_offset..target_offset + BYTES_PER_PIXEL]
                    .copy_from_slice(&src[source_offset..source_offset + BYTES_PER_PIXEL]);
            }
        }
    }

    out
}

/// Write a tightly-packed RGBA8888 buffer into a sub-rectangle of a canvas,
/// clipping to the canvas bounds.
fn blit_region(dst: CanvasMut<'_>, x: u16, y: u16, w: u16, h: u16, src: &[u8]) {
    let dst_stride = usize::from(dst.width) * BYTES_PER_PIXEL;
    let src_stride = usize::from(w) * BYTES_PER_PIXEL;
    let copy_cols = usize::from(w).min(usize::from(dst.width).saturating_sub(usize::from(x)));
    let copy_rows = usize::from(h).min(usize::from(dst.height).saturating_sub(usize::from(y)));
    let copy_bytes = copy_cols * BYTES_PER_PIXEL;
    if copy_bytes == 0 {
        return;
    }
    for row in 0..copy_rows {
        let dst_y = usize::from(y) + row;
        let dstart = dst_y * dst_stride + usize::from(x) * BYTES_PER_PIXEL;
        let sstart = row * src_stride;
        if dstart + copy_bytes <= dst.data.len() && sstart + copy_bytes <= src.len() {
            dst.data[dstart..dstart + copy_bytes].copy_from_slice(&src[sstart..sstart + copy_bytes]);
        }
    }
}

/// Fill a sub-rectangle of a canvas with a single RGBA8888 pixel, clipping to the
/// canvas bounds.
fn fill_region(dst: CanvasMut<'_>, x: u16, y: u16, w: u16, h: u16, px: [u8; 4]) {
    let dst_stride = usize::from(dst.width) * BYTES_PER_PIXEL;
    let copy_cols = usize::from(w).min(usize::from(dst.width).saturating_sub(usize::from(x)));
    let copy_rows = usize::from(h).min(usize::from(dst.height).saturating_sub(usize::from(y)));
    for row in 0..copy_rows {
        let dst_y = usize::from(y) + row;
        let base = dst_y * dst_stride + usize::from(x) * BYTES_PER_PIXEL;
        for col in 0..copy_cols {
            let offset = base + col * BYTES_PER_PIXEL;
            if offset + BYTES_PER_PIXEL <= dst.data.len() {
                dst.data[offset..offset + BYTES_PER_PIXEL].copy_from_slice(&px);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(left: u16, top: u16, right: u16, bottom: u16) -> ExclusiveRectangle {
        ExclusiveRectangle {
            left,
            top,
            right,
            bottom,
        }
    }

    /// A mapped surface filled with a solid color drains one output update, in
    /// output coordinates, with the expected pixels.
    #[test]
    fn solid_fill_on_mapped_surface_drains_output() {
        let mut c = Compositor::default();
        c.reset(200, 100);
        c.create_surface(1, 64, 64);
        c.map_surface(1, 10, 20);
        // map made the (zeroed) surface visible; commit and discard that delta.
        c.end_frame();
        let _ = c.drain_output();

        c.solid_fill(
            1,
            &Color {
                b: 0x33,
                g: 0x22,
                r: 0x11,
                xa: 0,
            },
            &[rect(0, 0, 4, 2)],
        );
        c.end_frame();
        let updates = c.drain_output();

        assert_eq!(updates.len(), 1);
        let u = &updates[0];
        assert_eq!(
            (u.region.left, u.region.top, u.region.right, u.region.bottom),
            (10, 20, 14, 22)
        );
        assert_eq!(u.data.len(), 4 * 2 * BYTES_PER_PIXEL);
        assert_eq!(&u.data[0..4], &[0x11, 0x22, 0x33, 0xFF]);
    }

    #[test]
    fn scaled_surface_materializes_nearest_neighbor_pixels_and_dirty_bounds() {
        let mut c = Compositor::default();
        c.reset(10, 10);
        c.create_surface(1, 2, 2);
        c.apply_bitmap(
            1,
            &rect(0, 0, 2, 2),
            &[
                0x10, 0x11, 0x12, 0xFF, 0x20, 0x21, 0x22, 0xFF, 0x30, 0x31, 0x32, 0xFF, 0x40, 0x41, 0x42, 0xFF,
            ],
        );
        c.end_frame();

        c.map_surface_scaled(1, 4, 3, 3, 3);
        c.end_frame();
        let updates = c.drain_output();
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].region, rect(4, 3, 7, 6));
        assert_eq!(
            updates[0].data,
            [
                0x10, 0x11, 0x12, 0xFF, 0x10, 0x11, 0x12, 0xFF, 0x20, 0x21, 0x22, 0xFF, 0x10, 0x11, 0x12, 0xFF, 0x10,
                0x11, 0x12, 0xFF, 0x20, 0x21, 0x22, 0xFF, 0x30, 0x31, 0x32, 0xFF, 0x30, 0x31, 0x32, 0xFF, 0x40, 0x41,
                0x42, 0xFF,
            ]
        );

        c.solid_fill(
            1,
            &Color {
                b: 0x53,
                g: 0x52,
                r: 0x51,
                xa: 0,
            },
            &[rect(1, 1, 2, 2)],
        );
        c.end_frame();
        let updates = c.drain_output();
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].region, rect(6, 5, 7, 6));
        assert_eq!(updates[0].data, [0x51, 0x52, 0x53, 0xFF]);
    }

    #[test]
    fn scaled_mapping_clips_without_discarding_other_output() {
        let mut c = Compositor::default();
        c.reset(8, 8);
        c.create_surface(1, 2, 2);
        c.create_surface(2, 1, 1);
        c.map_surface(2, 0, 0);
        c.end_frame();
        c.map_surface_scaled(1, 6, 5, 3, 3);
        c.end_frame();

        let updates = c.drain_output();
        assert_eq!(updates.len(), 2);
        assert_eq!(updates[0].region, rect(0, 0, 1, 1));
        assert_eq!(updates[1].region, rect(6, 5, 8, 8));
    }

    #[test]
    fn oversized_scaled_mapping_preserves_the_wire_scale_factor() {
        let mut c = Compositor::default();
        c.reset(u32::from(u16::MAX), 1);
        c.create_surface(1, 2, 1);
        c.apply_bitmap(1, &rect(0, 0, 2, 1), &[0x10, 0x11, 0x12, 0xFF, 0x20, 0x21, 0x22, 0xFF]);
        c.end_frame();

        c.map_surface_scaled(1, 0, 0, u32::MAX, 1);
        c.end_frame();
        let updates = c.drain_output();

        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].region, rect(0, 0, u16::MAX, 1));
        assert_eq!(&updates[0].data[0..4], &[0x10, 0x11, 0x12, 0xFF]);
        assert_eq!(
            &updates[0].data[updates[0].data.len() - BYTES_PER_PIXEL..],
            &[0x10, 0x11, 0x12, 0xFF]
        );
    }

    #[test]
    fn zero_scale_mapping_produces_no_output() {
        let mut c = Compositor::default();
        c.reset(8, 8);
        c.create_surface(1, 2, 2);
        c.map_surface_scaled(1, 0, 0, 0, 2);
        c.end_frame();

        assert!(c.drain_output().is_empty());
    }

    /// Deltas only become drainable after `EndFrame`.
    #[test]
    fn frame_is_atomic() {
        let mut c = Compositor::default();
        c.reset(200, 100);
        c.create_surface(1, 16, 16);
        c.map_surface(1, 0, 0);
        c.end_frame();
        let _ = c.drain_output();

        c.solid_fill(
            1,
            &Color {
                b: 1,
                g: 2,
                r: 3,
                xa: 0,
            },
            &[rect(0, 0, 8, 8)],
        );
        // Not committed yet.
        assert!(c.drain_output().is_empty());
        c.end_frame();
        assert_eq!(c.drain_output().len(), 1);
    }

    /// An unmapped surface produces no output.
    #[test]
    fn unmapped_surface_produces_no_output() {
        let mut c = Compositor::default();
        c.reset(200, 100);
        c.create_surface(1, 16, 16);
        c.solid_fill(
            1,
            &Color {
                b: 1,
                g: 2,
                r: 3,
                xa: 0,
            },
            &[rect(0, 0, 8, 8)],
        );
        c.end_frame();
        assert!(c.drain_output().is_empty());
    }

    /// `apply_bitmap` writes into the surface and drains the mapped region.
    #[test]
    fn apply_bitmap_composites_and_drains() {
        let mut c = Compositor::default();
        c.reset(64, 64);
        c.create_surface(1, 32, 32);
        c.map_surface(1, 0, 0);
        c.end_frame();
        let _ = c.drain_output();

        let px = vec![0xAAu8; 2 * 2 * BYTES_PER_PIXEL];
        c.apply_bitmap(1, &rect(4, 4, 6, 6), &px);
        c.end_frame();
        let updates = c.drain_output();
        assert_eq!(updates.len(), 1);
        assert_eq!(
            (
                updates[0].region.left,
                updates[0].region.top,
                updates[0].region.right,
                updates[0].region.bottom
            ),
            (4, 4, 6, 6)
        );
        assert_eq!(&updates[0].data[0..4], &[0xAA, 0xAA, 0xAA, 0xAA]);
    }

    /// A cache round-trip (surface -> cache -> other surface) reproduces the pixels.
    #[test]
    fn cache_round_trip() {
        let mut c = Compositor::default();
        c.reset(128, 128);
        c.create_surface(1, 16, 16);
        c.create_surface(2, 16, 16);
        c.map_surface(2, 0, 0);
        c.end_frame();
        let _ = c.drain_output();

        c.solid_fill(
            1,
            &Color {
                b: 0x10,
                g: 0x20,
                r: 0x30,
                xa: 0,
            },
            &[rect(0, 0, 8, 8)],
        );
        c.surface_to_cache(1, 7, &rect(0, 0, 8, 8));
        c.cache_to_surface(7, 2, &[Point { x: 2, y: 2 }]);
        c.end_frame();

        let updates = c.drain_output();
        // One update on surface 2 (surface 1 is unmapped, so its fill drained nothing).
        assert_eq!(updates.len(), 1);
        let u = &updates[0];
        assert_eq!(
            (u.region.left, u.region.top, u.region.right, u.region.bottom),
            (2, 2, 10, 10)
        );
        assert_eq!(&u.data[0..4], &[0x30, 0x20, 0x10, 0xFF]);
    }

    /// A destination rectangle larger than the surface is clipped, not panicked.
    #[test]
    fn oversized_rect_is_clipped() {
        let mut c = Compositor::default();
        c.reset(64, 64);
        c.create_surface(1, 8, 8);
        c.map_surface(1, 0, 0);
        c.end_frame();
        let _ = c.drain_output();

        c.solid_fill(
            1,
            &Color {
                b: 1,
                g: 2,
                r: 3,
                xa: 0,
            },
            &[rect(0, 0, 1000, 1000)],
        );
        c.end_frame();
        let updates = c.drain_output();
        assert_eq!(updates.len(), 1);
        // Clipped to the 8x8 surface.
        assert_eq!((updates[0].region.right, updates[0].region.bottom), (8, 8));
    }

    /// `ResetGraphics` only resizes the output (MS-RDPEGFX 3.3.5.14); surfaces
    /// survive it, so commands issued against a surface created before the reset
    /// still paint once the surface is remapped in the new output.
    #[test]
    fn surfaces_survive_a_reset() {
        let mut c = Compositor::default();
        c.reset(128, 128);
        c.create_surface(1, 16, 16);
        c.map_surface(1, 0, 0);
        c.end_frame();
        let _ = c.drain_output();

        // Different size than before: a server resize commonly changes it.
        c.reset(64, 64);

        c.solid_fill(
            1,
            &Color {
                b: 1,
                g: 2,
                r: 3,
                xa: 0,
            },
            &[rect(0, 0, 8, 8)],
        );
        c.end_frame();
        let updates = c.drain_output();
        assert_eq!(updates.len(), 1, "the surface must still be mapped after the reset");
        assert_eq!(&updates[0].data[0..4], &[3, 2, 1, 0xFF]);
    }

    /// Cache slots are removed only by `EvictCacheEntry` (MS-RDPEGFX 3.3.5.8), never
    /// by `ResetGraphics`. A Windows server relies on this: after a resize it sends
    /// ResetGraphics + CreateSurface + StartFrame and then paints through
    /// `CacheToSurface` with slots filled before the reset, without re-sending them.
    #[test]
    fn cache_slots_survive_a_reset() {
        let mut c = Compositor::default();
        c.reset(128, 128);
        c.create_surface(1, 16, 16);
        c.solid_fill(
            1,
            &Color {
                b: 0x10,
                g: 0x20,
                r: 0x30,
                xa: 0,
            },
            &[rect(0, 0, 8, 8)],
        );
        c.surface_to_cache(1, 7, &rect(0, 0, 8, 8));

        c.reset(256, 256);
        c.create_surface(2, 16, 16);
        c.map_surface(2, 0, 0);
        c.end_frame();
        let _ = c.drain_output(); // discard the mapping delta

        c.cache_to_surface(7, 2, &[Point { x: 2, y: 2 }]);
        c.end_frame();
        let updates = c.drain_output();

        assert_eq!(updates.len(), 1, "the cache slot must still exist after the reset");
        let u = &updates[0];
        assert_eq!(
            (u.region.left, u.region.top, u.region.right, u.region.bottom),
            (2, 2, 10, 10)
        );
        assert_eq!(&u.data[0..4], &[0x30, 0x20, 0x10, 0xFF]);
    }

    /// A `SurfaceToSurface` source rectangle larger than the source surface is
    /// clamped to it, so the copied tile is bounded (no oversized allocation).
    #[test]
    fn oversized_source_rect_is_clamped_to_source() {
        let mut c = Compositor::default();
        c.reset(128, 128);
        c.create_surface(1, 8, 8);
        c.create_surface(2, 64, 64);
        c.map_surface(2, 0, 0);
        c.end_frame();
        let _ = c.drain_output();

        // Source rectangle wildly exceeds surface 1 (8x8); it must clamp to 8x8.
        c.surface_to_surface(1, 2, &rect(0, 0, 60000, 60000), &[Point { x: 0, y: 0 }]);
        c.end_frame();
        let updates = c.drain_output();
        assert_eq!(updates.len(), 1);
        assert_eq!((updates[0].region.right, updates[0].region.bottom), (8, 8));
        assert_eq!(updates[0].data.len(), 8 * 8 * BYTES_PER_PIXEL);
    }

    /// A single surface at the per-edge limit is 1 GiB, which the aggregate budget
    /// must refuse outright. The edge limit alone never did.
    #[test]
    fn a_surface_larger_than_the_budget_is_refused() {
        let mut c = Compositor::default();
        c.reset(1920, 1080);
        c.create_surface(1, MAX_SURFACE_DIM, MAX_SURFACE_DIM);

        assert!(c.surfaces.is_empty(), "a 1 GiB surface must not be allocated");
        assert_eq!(c.allocated_bytes, 0);
    }

    /// Surfaces that each fit are still refused once their total would not.
    /// Bounding one allocation is not the property that matters when the peer
    /// chooses how many to ask for.
    #[test]
    fn surfaces_are_refused_once_the_total_would_exceed_the_budget() {
        const EDGE: u16 = 4096; // 4096 * 4096 * 4 == 64 MiB
        let per_surface = usize::from(EDGE) * usize::from(EDGE) * BYTES_PER_PIXEL;
        let fit = MAX_COMPOSITOR_BYTES / per_surface;

        let mut c = Compositor::default();
        c.reset(1920, 1080);

        for id in 0..u16::try_from(fit).unwrap() {
            c.create_surface(id, EDGE, EDGE);
        }
        assert_eq!(c.surfaces.len(), fit, "every surface within budget must be allocated");

        c.create_surface(u16::try_from(fit).unwrap(), EDGE, EDGE);
        assert_eq!(c.surfaces.len(), fit, "the surface crossing the budget must be refused");
        assert!(c.allocated_bytes <= MAX_COMPOSITOR_BYTES);
    }

    /// Deleting a surface returns its bytes, so the budget is a live figure rather
    /// than a high-water mark that permanently poisons the session.
    #[test]
    fn deleting_a_surface_releases_its_charge() {
        const EDGE: u16 = 4096;
        let mut c = Compositor::default();
        c.reset(1920, 1080);

        c.create_surface(1, EDGE, EDGE);
        let charged = c.allocated_bytes;
        assert!(charged > 0);

        c.delete_surface(1);
        assert_eq!(c.allocated_bytes, 0, "the charge must be released on delete");

        // And the freed budget is usable again.
        c.create_surface(2, EDGE, EDGE);
        assert_eq!(c.allocated_bytes, charged);
    }

    /// `BTreeMap::insert` drops whatever was under the id, so re-creating a surface
    /// must not charge twice for one slot. Without the release this leaks budget
    /// until the compositor refuses everything.
    #[test]
    fn recreating_a_surface_does_not_double_charge() {
        const EDGE: u16 = 4096;
        let mut c = Compositor::default();
        c.reset(1920, 1080);

        c.create_surface(1, EDGE, EDGE);
        let once = c.allocated_bytes;

        for _ in 0..8 {
            c.create_surface(1, EDGE, EDGE);
        }

        assert_eq!(c.allocated_bytes, once, "re-creating one id must not accumulate charge");
        assert_eq!(c.surfaces.len(), 1);
    }

    /// `ResetGraphics` keeps surfaces and the cache, so the charge after a reset
    /// must equal exactly their pixel bytes, not zero; only the queued (not yet
    /// drained) deltas in `ready` are discarded, since they were clipped against
    /// the output size that no longer applies.
    #[test]
    fn reset_releases_only_the_queued_deltas() {
        const EDGE: u16 = 4096;
        let mut c = Compositor::default();
        c.reset(1920, 1080);
        c.create_surface(1, EDGE, EDGE);
        c.create_surface(2, EDGE, EDGE);
        c.surface_to_cache(1, 7, &rect(0, 0, 8, 8));
        c.map_surface(1, 0, 0);
        c.end_frame(); // commits a delta into `ready`
        assert!(!c.ready.is_empty(), "the mapped surface must have committed a delta");

        let surfaces_and_cache_bytes = c.allocated_bytes - c.ready.iter().map(|u| u.data.len()).sum::<usize>();

        c.reset(1280, 720);

        assert_eq!(
            c.allocated_bytes, surfaces_and_cache_bytes,
            "reset must keep the surfaces' and cache's charge, discarding only the queued deltas"
        );
        assert!(c.ready.is_empty(), "queued deltas must not survive a resizing reset");
    }

    /// The other half of the pair: a reset that does NOT change the output is not a
    /// different output, so the deltas clipped against it are still valid and must
    /// survive. Windows sends same-size `ResetGraphics` whenever it rebuilds its own
    /// surfaces, and the session leaves the consumer's image untouched in that case,
    /// so a cleared queue is a frame lost with nothing to repaint it.
    #[test]
    fn a_same_size_reset_keeps_the_queued_deltas() {
        let mut c = Compositor::default();
        c.reset(1920, 1080);
        c.create_surface(1, 64, 64);
        c.map_surface(1, 0, 0);
        c.end_frame();
        assert!(!c.ready.is_empty(), "the mapped surface must have committed a delta");
        let charged = c.allocated_bytes;

        c.reset(1920, 1080);

        assert!(
            !c.ready.is_empty(),
            "a same-size reset must not discard committed deltas"
        );
        assert_eq!(
            c.allocated_bytes, charged,
            "a same-size reset releases nothing, because it discards nothing"
        );
        assert_eq!(
            c.drain_output().len(),
            1,
            "the delta must still be drainable after a same-size reset"
        );
    }

    /// Strengthens `reset_releases_only_the_queued_deltas` by also exercising the
    /// `frame` term: a payload can carry `ResetGraphics` while a frame is still
    /// open (before `EndFrame`), and the uncommitted dirty metadata queued in
    /// `frame` must be released too, not just the committed deltas in `ready`.
    #[test]
    fn reset_mid_frame_releases_the_dirty_metadata_too() {
        const EDGE: u16 = 4096;
        const TILE: u16 = 8;
        const OUTPUT_W: u16 = 1920;
        const OUTPUT_H: u16 = 1080;
        const DELTA_W: u16 = 8;
        const DELTA_H: u16 = 8;

        let mut c = Compositor::default();
        c.reset(u32::from(OUTPUT_W), u32::from(OUTPUT_H));
        c.create_surface(1, EDGE, EDGE);
        c.create_surface(2, EDGE, EDGE);
        c.surface_to_cache(1, 7, &rect(0, 0, TILE, TILE));

        // Mapping itself dirties the whole surface (it just became visible), clipped
        // to the output. Commit that delta first so it is the test's one known
        // `ready` entry, sized by the output rather than the surface.
        c.map_surface(1, 0, 0);
        c.end_frame();
        assert_eq!(c.ready.len(), 1, "mapping the surface must have committed one delta");

        let color = Color {
            r: 0,
            g: 0,
            b: 0,
            xa: 0xFF,
        };

        // An uncommitted dirty entry left in `frame` (no following `end_frame`).
        c.solid_fill(1, &color, &[rect(10, 10, 10 + DELTA_W, 10 + DELTA_H)]);
        assert_eq!(
            c.frame.len(),
            1,
            "the fill must have queued one uncommitted dirty entry"
        );

        let surfaces_bytes = 2 * usize::from(EDGE) * usize::from(EDGE) * BYTES_PER_PIXEL;
        let cache_bytes = usize::from(TILE) * usize::from(TILE) * BYTES_PER_PIXEL;
        // The mapping delta is clipped to the output, which is smaller than the surface.
        let ready_bytes = usize::from(OUTPUT_W) * usize::from(OUTPUT_H) * BYTES_PER_PIXEL;
        let frame_bytes = size_of::<DirtyRegion>(); // one uncommitted dirty entry

        assert_eq!(
            c.allocated_bytes,
            surfaces_bytes + cache_bytes + ready_bytes + frame_bytes,
            "allocated_bytes must equal the explicit sum of surfaces, cache, ready and frame charges"
        );

        c.reset(u32::from(OUTPUT_W) / 2, u32::from(OUTPUT_H) / 2);

        assert!(c.ready.is_empty(), "queued deltas must not survive a resizing reset");
        assert!(
            c.frame.is_empty(),
            "uncommitted dirty metadata must not survive a resizing reset"
        );
        assert_eq!(
            c.allocated_bytes,
            surfaces_bytes + cache_bytes,
            "reset must release both the ready deltas and the uncommitted frame metadata"
        );
    }

    /// Sibling of `a_same_size_reset_keeps_the_queued_deltas`, but for the `frame`
    /// term instead of `ready`: a same-size `ResetGraphics` can also arrive mid-frame
    /// (before `EndFrame`), and the uncommitted dirty metadata queued in `frame` must
    /// survive it too, exactly as the committed deltas in `ready` do — a reset that
    /// changes nothing invalidates nothing (MS-RDPEGFX 3.3.5.14: reset resizes the
    /// Graphics Output Buffer).
    #[test]
    fn a_same_size_reset_mid_frame_keeps_the_dirty_metadata_too() {
        let mut c = Compositor::default();
        c.reset(1920, 1080);
        c.create_surface(1, 64, 64);
        c.map_surface(1, 0, 0);
        c.end_frame();
        let _ = c.drain_output(); // discard the mapping delta

        let color = Color {
            r: 1,
            g: 2,
            b: 3,
            xa: 0,
        };
        c.solid_fill(1, &color, &[rect(0, 0, 8, 8)]);
        assert_eq!(
            c.frame.len(),
            1,
            "the fill must have queued one uncommitted dirty entry"
        );

        c.reset(1920, 1080);

        assert!(
            !c.frame.is_empty(),
            "a same-size reset must not discard uncommitted dirty metadata"
        );

        c.end_frame();
        assert_eq!(
            c.drain_output().len(),
            1,
            "the uncommitted delta must still be drainable after a same-size reset"
        );
    }

    /// Cache slots are a second allocation pool keyed by `u16`. Charging them against
    /// the same budget is what stops a peer from bypassing the surface limit by
    /// parking the same pixels in tens of thousands of slots instead.
    #[test]
    fn cache_entries_are_refused_once_the_total_would_exceed_the_budget() {
        const EDGE: u16 = 4096; // 64 MiB per surface, and per full-surface tile
        let mut c = Compositor::default();
        c.reset(1920, 1080);
        c.create_surface(1, EDGE, EDGE);

        let after_surface = c.allocated_bytes;
        let slots_that_fit = (MAX_COMPOSITOR_BYTES - after_surface) / after_surface;

        for slot in 0..u16::try_from(slots_that_fit).unwrap() {
            c.surface_to_cache(1, slot, &rect(0, 0, EDGE, EDGE));
        }
        assert_eq!(c.cache.len(), slots_that_fit);

        c.surface_to_cache(1, u16::try_from(slots_that_fit).unwrap(), &rect(0, 0, EDGE, EDGE));
        assert_eq!(
            c.cache.len(),
            slots_that_fit,
            "the tile crossing the shared budget must be refused"
        );
        assert!(c.allocated_bytes <= MAX_COMPOSITOR_BYTES);
    }

    /// The cache twin of the surface double-charge: repeatedly caching into one slot
    /// replaces the tile, so it must not accumulate charge.
    #[test]
    fn recaching_into_one_slot_does_not_double_charge() {
        const EDGE: u16 = 2048;
        let mut c = Compositor::default();
        c.reset(1920, 1080);
        c.create_surface(1, EDGE, EDGE);

        c.surface_to_cache(1, 7, &rect(0, 0, EDGE, EDGE));
        let once = c.allocated_bytes;

        for _ in 0..8 {
            c.surface_to_cache(1, 7, &rect(0, 0, EDGE, EDGE));
        }

        assert_eq!(c.allocated_bytes, once, "one slot must hold one charge");
        assert_eq!(c.cache.len(), 1);
    }

    /// Evicting a slot returns its bytes, so a session that cycles the cache does not
    /// ratchet the budget upward.
    #[test]
    fn evicting_a_cache_entry_releases_its_charge() {
        const EDGE: u16 = 2048;
        let mut c = Compositor::default();
        c.reset(1920, 1080);
        c.create_surface(1, EDGE, EDGE);
        let surface_only = c.allocated_bytes;

        c.surface_to_cache(1, 7, &rect(0, 0, EDGE, EDGE));
        assert!(c.allocated_bytes > surface_only);

        c.evict_cache_entry(7);
        assert_eq!(
            c.allocated_bytes, surface_only,
            "eviction must return the tile's bytes to the budget"
        );
    }

    /// Every destination point receives the cached pixels. The blits share one
    /// borrow of the tile and one surface lookup, so this pins that the loop still
    /// paints each point rather than only the first.
    #[test]
    fn cache_to_surface_paints_every_destination_point() {
        let mut c = Compositor::default();
        c.reset(128, 128);
        c.create_surface(1, 16, 16);
        c.create_surface(2, 16, 16);
        c.map_surface(2, 0, 0);
        c.end_frame();
        let _ = c.drain_output();

        c.solid_fill(
            1,
            &Color {
                b: 0x10,
                g: 0x20,
                r: 0x30,
                xa: 0,
            },
            &[rect(0, 0, 8, 8)],
        );
        c.surface_to_cache(1, 7, &rect(0, 0, 8, 8));
        c.cache_to_surface(7, 2, &[Point { x: 0, y: 0 }, Point { x: 8, y: 8 }]);
        c.end_frame();

        let updates = c.drain_output();
        assert_eq!(updates.len(), 2, "each destination point produces its own delta");
        for update in &updates {
            assert_eq!(&update.data[0..4], &[0x30, 0x20, 0x10, 0xFF]);
        }
        assert_eq!(
            (updates[1].region.left, updates[1].region.top),
            (8, 8),
            "the second point must be painted, not skipped"
        );
    }

    /// A PDU that names one rectangle repeatedly queues one delta, not one per
    /// entry: `rectCount` is a `u16`, so eager copies would amplify a small
    /// payload into 65,535 full-region buffers.
    #[test]
    fn repeated_paints_of_one_region_queue_a_single_delta() {
        let mut c = Compositor::default();
        c.reset(1920, 1080);
        c.create_surface(1, 512, 512);
        c.map_surface(1, 0, 0);
        c.end_frame();
        let _ = c.drain_output();

        let color = Color {
            b: 0x11,
            g: 0x22,
            r: 0x33,
            xa: 0,
        };
        let repeated = vec![rect(0, 0, 512, 512); 4096];
        c.solid_fill(1, &color, &repeated);
        c.end_frame();

        let updates = c.drain_output();
        assert_eq!(updates.len(), 1, "identical rectangles must collapse to one delta");
    }

    /// Pixels are read at `EndFrame`, so a region painted twice within a frame
    /// yields one delta carrying the second color rather than two the consumer
    /// would paint over each other.
    #[test]
    fn deferred_pixels_carry_the_frames_final_state() {
        let mut c = Compositor::default();
        c.reset(1920, 1080);
        c.create_surface(1, 16, 16);
        c.map_surface(1, 0, 0);
        c.end_frame();
        let _ = c.drain_output();

        let first = Color {
            b: 0x01,
            g: 0x02,
            r: 0x03,
            xa: 0,
        };
        let second = Color {
            b: 0xF1,
            g: 0xF2,
            r: 0xF3,
            xa: 0,
        };
        c.solid_fill(1, &first, &[rect(0, 0, 8, 8)]);
        c.solid_fill(1, &second, &[rect(0, 0, 8, 8)]);
        c.end_frame();

        let updates = c.drain_output();
        assert_eq!(updates.len(), 1);
        assert_eq!(&updates[0].data[0..4], &[0xF3, 0xF2, 0xF1, 0xFF]);
    }

    /// Deferral reads the surface as it stands at commit, so a surface deleted
    /// mid-frame contributes nothing: its output region is repainted by whatever
    /// maps there next.
    #[test]
    fn a_surface_deleted_mid_frame_contributes_no_delta() {
        let mut c = Compositor::default();
        c.reset(1920, 1080);
        c.create_surface(1, 16, 16);
        c.map_surface(1, 0, 0);
        c.end_frame();
        let _ = c.drain_output();

        c.solid_fill(
            1,
            &Color {
                b: 0x11,
                g: 0x22,
                r: 0x33,
                xa: 0,
            },
            &[rect(0, 0, 8, 8)],
        );
        c.delete_surface(1);
        c.end_frame();

        assert!(c.drain_output().is_empty());
    }

    /// Distinct rectangles still materialize one copy each, so the queue is
    /// charged against the same budget as the surfaces and refuses past it.
    #[test]
    fn queued_deltas_are_refused_once_they_would_exceed_the_budget() {
        const EDGE: u16 = 4096; // 4096 * 4096 * 4 == 64 MiB
        let mut c = Compositor::default();
        c.reset(u32::from(EDGE), u32::from(EDGE));
        c.create_surface(1, EDGE, EDGE);
        c.map_surface(1, 0, 0);
        c.end_frame();
        let _ = c.drain_output();

        // Neither rectangle contains the other, so the repeat filter keeps all four
        // as distinct dirty regions. Each is one pixel row short of the surface.
        c.solid_fill(
            1,
            &Color {
                b: 0,
                g: 0,
                r: 0,
                xa: 0,
            },
            &[
                rect(0, 0, EDGE, EDGE - 1),
                rect(0, 1, EDGE, EDGE),
                rect(0, 0, EDGE, EDGE - 1),
                rect(0, 1, EDGE, EDGE),
            ],
        );
        c.end_frame();

        // 64 MiB of surface leaves room for three of the four deltas, each one row
        // short of 64 MiB. An exact count also proves the repeat filter left all
        // four regions distinct rather than collapsing them.
        let updates = c.drain_output();
        assert_eq!(
            updates.len(),
            3,
            "the queue must refuse deltas once the shared budget is exhausted"
        );
        assert!(c.allocated_bytes <= MAX_COMPOSITOR_BYTES);
    }

    /// The repeat filter only collapses consecutive repeats, so a peer that never
    /// sends `EndFrame` can alternate two rectangles and add a dirty entry per
    /// rectangle indefinitely. The entries are charged, so that growth stops at the
    /// budget instead of running to exhaustion.
    #[test]
    fn alternating_dirty_rectangles_stop_at_the_budget() {
        const ENTRY: usize = size_of::<DirtyRegion>();
        const FIT: usize = 64;
        let mut c = Compositor::default();
        c.reset(64, 64);
        c.create_surface(1, 64, 64);

        // Spend the budget down to a known headroom rather than allocating 256 MiB
        // of surfaces to reach it, so the loop below stays short.
        let reserve = MAX_COMPOSITOR_BYTES - c.allocated_bytes - FIT * ENTRY;
        assert!(c.charge(reserve));

        // Neither rectangle contains the other, so every one of these survives the
        // repeat filter and reaches the push. No `EndFrame` in between: this is the
        // single unbounded frame the filter cannot collapse.
        let black = Color {
            b: 0,
            g: 0,
            r: 0,
            xa: 0,
        };
        for _ in 0..FIT {
            c.solid_fill(1, &black, &[rect(0, 0, 64, 63), rect(0, 1, 64, 64)]);
        }

        assert_eq!(c.frame.len(), FIT, "the frame must stop growing at the budget");
        assert_eq!(c.allocated_bytes, MAX_COMPOSITOR_BYTES);
    }

    /// Draining hands the pixels to the consumer, so the queue's charge returns to
    /// the budget rather than ratcheting it upward frame after frame.
    #[test]
    fn draining_releases_the_queue_charge() {
        let mut c = Compositor::default();
        c.reset(1920, 1080);
        c.create_surface(1, 512, 512);
        let surface_only = c.allocated_bytes;

        c.map_surface(1, 0, 0);
        c.end_frame();
        assert!(c.allocated_bytes > surface_only, "the queued delta must be charged");

        let _ = c.drain_output();
        assert_eq!(
            c.allocated_bytes, surface_only,
            "draining must return the delta's bytes to the budget"
        );
    }

    /// A payload may carry `EndFrame` and `ResetGraphics` together, since `process`
    /// handles every PDU in it before the consumer can drain. Deltas committed
    /// before the reset were clipped against the previous output, so they must not
    /// survive into the new one.
    #[test]
    fn reset_discards_deltas_committed_before_it() {
        let mut c = Compositor::default();
        c.reset(4096, 4096);
        c.create_surface(1, 2048, 2048);
        let surface_only = c.allocated_bytes;
        c.map_surface(1, 2048, 2048);
        c.end_frame();
        assert!(!c.ready.is_empty(), "the mapped surface must have committed a delta");
        assert!(c.allocated_bytes > surface_only, "the queued delta must be charged");

        // Shrinking the output is the case that matters: the queued region sits
        // outside the new bounds entirely.
        c.reset(1920, 1080);
        assert!(c.drain_output().is_empty(), "pre-reset deltas must not be drainable");
        // The surface itself survives the reset (MS-RDPEGFX 3.3.5.14), so its charge
        // remains; only the discarded queued delta's charge is released.
        assert_eq!(c.allocated_bytes, surface_only);
    }

    fn tile(cache_key: u64, width: u16, height: u16, fill: u8) -> CacheImportTile {
        CacheImportTile {
            cache_key,
            width,
            height,
            data: vec![fill; usize::from(width) * usize::from(height) * BYTES_PER_PIXEL],
        }
    }

    /// Staging charges every staged tile and offers them in the order given, with
    /// `bitmap_len = width * height * 4`.
    #[test]
    fn staged_import_is_charged_and_offered_in_order() {
        let mut c = Compositor::default();
        let entries = c.stage_import(vec![tile(10, 2, 2, 1), tile(20, 3, 1, 2), tile(30, 1, 1, 3)]);
        let offered: Vec<(u64, u32)> = entries.iter().map(|e| (e.cache_key, e.bitmap_len)).collect();
        assert_eq!(offered, vec![(10, 16), (20, 12), (30, 4)]);
        assert_eq!(c.allocated_bytes, 16 + 12 + 4);
    }

    #[test]
    fn staging_skips_malformed_and_repeated_tiles() {
        let mut c = Compositor::default();
        let mut short = tile(11, 2, 2, 1);
        short.data.pop();
        let entries = c.stage_import(vec![
            short,
            tile(12, 0, 4, 1),
            tile(13, 1, 1, 1),
            tile(13, 1, 1, 9),
            tile(14, 1, 1, 1),
        ]);
        let keys: Vec<u64> = entries.iter().map(|e| e.cache_key).collect();
        assert_eq!(keys, vec![13, 14]);
        assert_eq!(c.allocated_bytes, 4 + 4);
    }

    #[test]
    fn staging_stops_at_the_entry_cap() {
        let mut c = Compositor::default();
        let tiles = (0..u64::try_from(MAX_CACHE_IMPORT_ENTRIES + 3).unwrap())
            .map(|k| tile(k + 1, 1, 1, 1))
            .collect();
        assert_eq!(c.stage_import(tiles).len(), MAX_CACHE_IMPORT_ENTRIES);
    }

    #[test]
    fn staging_stops_at_the_byte_cap() {
        const EDGE: u16 = 2048; // 16 MiB per tile
        let per_tile = usize::from(EDGE) * usize::from(EDGE) * BYTES_PER_PIXEL;
        let fit = MAX_CACHE_IMPORT_BYTES / per_tile; // 6
        let mut c = Compositor::default();
        let tiles = (0..u64::try_from(fit + 1).unwrap())
            .map(|k| tile(k + 1, EDGE, EDGE, 1))
            .collect();
        assert_eq!(c.stage_import(tiles).len(), fit);
        assert_eq!(c.allocated_bytes, fit * per_tile);
    }

    /// The budget is checked before the offer exists: a tile the budget refuses is
    /// never offered, so the reply can never name a slot the compositor then refuses.
    #[test]
    fn staging_stops_where_the_budget_refuses() {
        const SURFACE_EDGE: u16 = 4096; // 64 MiB per surface
        const TILE_EDGE: u16 = 2048; // 16 MiB per tile
        let mut c = Compositor::default();
        for id in 1..=3 {
            c.create_surface(id, SURFACE_EDGE, SURFACE_EDGE);
        }
        let surfaces = 3 * usize::from(SURFACE_EDGE) * usize::from(SURFACE_EDGE) * BYTES_PER_PIXEL;
        let per_tile = usize::from(TILE_EDGE) * usize::from(TILE_EDGE) * BYTES_PER_PIXEL;
        let fit = (MAX_COMPOSITOR_BYTES - surfaces) / per_tile; // 4, below the byte cap's 6
        let tiles = (0..6).map(|k| tile(k + 1, TILE_EDGE, TILE_EDGE, 1)).collect();
        assert_eq!(c.stage_import(tiles).len(), fit);
        assert_eq!(c.allocated_bytes, surfaces + fit * per_tile);
    }

    /// The reply fills exactly the slots it names, with the tile at the same offer
    /// index, and releases every tile it does not name.
    #[test]
    fn commit_fills_exactly_the_named_slots() {
        let mut c = Compositor::default();
        c.stage_import(vec![tile(10, 2, 2, 0xA1), tile(20, 2, 2, 0xB2), tile(30, 1, 1, 0xC3)]);
        let outcome = c.commit_import(&[5, 0, 9]);
        assert_eq!(outcome.offered, 3);
        assert_eq!(outcome.imported, vec![(10, 5), (30, 9)]);
        assert!(outcome.unfilled_slots.is_empty());
        assert_eq!(c.cache.keys().copied().collect::<Vec<_>>(), vec![5, 9]);
        assert_eq!(c.cache_tile(5), Some((2, 2, &[0xA1; 16][..])));
        assert_eq!(c.cache_tile(9), Some((1, 1, &[0xC3; 4][..])));
        assert_eq!(c.allocated_bytes, 16 + 4, "the unnamed tile's charge must be released");
    }

    /// An imported tile paints through `CacheToSurface` like a tile the server cached
    /// in this session.
    #[test]
    fn imported_tile_paints_through_cache_to_surface() {
        let mut c = Compositor::default();
        c.reset(16, 16);
        c.create_surface(1, 16, 16);
        c.map_surface(1, 0, 0);
        c.end_frame();
        let _ = c.drain_output();
        c.stage_import(vec![tile(10, 2, 2, 0x7F)]);
        c.commit_import(&[7]);
        c.cache_to_surface(7, 1, &[Point { x: 4, y: 4 }]);
        c.end_frame();
        let updates = c.drain_output();
        assert_eq!(updates.len(), 1);
        assert_eq!((updates[0].region.left, updates[0].region.top), (4, 4));
        assert_eq!(updates[0].data, vec![0x7F; 16]);
    }

    #[test]
    fn commit_reports_slots_it_cannot_fill() {
        let mut c = Compositor::default();
        c.stage_import(vec![tile(10, 1, 1, 1)]);
        let outcome = c.commit_import(&[4, 7]);
        assert_eq!(outcome.imported, vec![(10, 4)]);
        assert_eq!(outcome.unfilled_slots, vec![7]);
        // A second reply has nothing staged behind it.
        let again = c.commit_import(&[3]);
        assert_eq!(again.offered, 0);
        assert!(again.imported.is_empty());
        assert_eq!(again.unfilled_slots, vec![3]);
        assert_eq!(c.cache.keys().copied().collect::<Vec<_>>(), vec![4]);
    }

    #[test]
    fn commit_reports_a_slot_named_twice() {
        let mut c = Compositor::default();
        c.stage_import(vec![tile(10, 1, 1, 1), tile(20, 1, 1, 2)]);
        let outcome = c.commit_import(&[4, 4]);
        assert_eq!(outcome.imported, vec![(10, 4)]);
        assert_eq!(outcome.unfilled_slots, vec![4]);
        assert_eq!(c.allocated_bytes, 4);
    }

    /// Importing into an occupied slot replaces the occupant and releases its charge.
    #[test]
    fn commit_replaces_an_occupied_slot() {
        let mut c = Compositor::default();
        c.create_surface(1, 8, 8);
        c.surface_to_cache(1, 5, &rect(0, 0, 8, 8));
        c.stage_import(vec![tile(10, 1, 1, 1)]);
        c.commit_import(&[5]);
        assert_eq!(c.allocated_bytes, 8 * 8 * BYTES_PER_PIXEL + 4);
        assert_eq!(c.cache_tile(5), Some((1, 1, &[1u8; 4][..])));
    }

    #[test]
    fn staged_import_survives_a_reset() {
        let mut c = Compositor::default();
        c.reset(100, 100);
        c.stage_import(vec![tile(10, 1, 1, 1)]);
        c.reset(200, 150);
        assert_eq!(c.commit_import(&[3]).imported, vec![(10, 3)]);
    }

    #[test]
    fn discard_releases_the_staged_charge() {
        let mut c = Compositor::default();
        c.stage_import(vec![tile(10, 2, 2, 1), tile(20, 1, 1, 1)]);
        c.discard_import();
        assert_eq!(c.allocated_bytes, 0);
        let outcome = c.commit_import(&[1, 2]);
        assert!(outcome.imported.is_empty());
        assert_eq!(outcome.unfilled_slots, vec![1, 2]);
    }

    /// A tile the budget refuses leaves no tile behind for `cache_tile` to report.
    #[test]
    fn a_refused_surface_to_cache_leaves_no_tile() {
        let mut c = Compositor::default();
        c.surface_to_cache(99, 5, &rect(0, 0, 8, 8)); // no such surface
        assert_eq!(c.cache_tile(5), None);
    }
}
