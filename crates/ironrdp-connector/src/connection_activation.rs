use core::mem;

use ironrdp_pdu::rdp;
use ironrdp_pdu::rdp::capability_sets::{
    CapabilitySet, InputFlags, Rail, RailSupportLevel, WindowList, WindowSupportLevel,
};
use tracing::{debug, warn};

use crate::{
    Config, ConnectionFinalizationSequence, ConnectorError, ConnectorErrorExt as _, ConnectorResult, DesktopSize,
    MonotonicInstant, Sequence, State, Written, encode_send_data_request, general_err, reason_err,
};

/// TESSERA PATCH — mirrors `license_exchange::SKIP_WARN_THRESHOLD`: past this
/// many answered auto-detect probes in `CapabilitiesExchange`, logging
/// escalates from `debug!` to `warn!`.
const CAPABILITIES_PROBE_WARN_THRESHOLD: u32 = 3;

/// TESSERA PATCH — ceiling on virtual-channel data buffered while a
/// connection activation runs, before the sequence starts dropping it.
///
/// A reactivation completes in milliseconds in practice (3ms and 50ms in the
/// two captured incidents), but `tessera-rdp-worker`'s frame pump gives it up
/// to 15s before giving up, and a saturated link would hand this buffer
/// hundreds of megabytes in that window. 16 MiB covers a reactivation several
/// orders of magnitude slower than any observed one while staying far below
/// what the worker can afford to hold.
pub const DEFERRED_PDU_BUDGET: usize = 16 * 1024 * 1024;

/// Represents the Capability Exchange and Connection Finalization phases
/// of the connection sequence (section [1.3.1.1]).
///
/// This is abstracted into its own struct to allow it to be used for the ordinary
/// RDP connection sequence [`ClientConnector`] that occurs for every RDP connection,
/// as well as the Deactivation-Reactivation Sequence ([1.3.1.3]) that occurs when
/// a [Server Deactivate All PDU] is received.
///
/// [1.3.1.1]: https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-rdpbcgr/023f1e69-cfe8-4ee6-9ee0-7e759fb4e4ee
/// [1.3.1.3]: https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-rdpbcgr/dfc234ce-481a-4674-9a5d-2a7bafb14432
/// [`ClientConnector`]: crate::ClientConnector
/// [Server Deactivate All PDU]: https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-rdpbcgr/8a29971a-df3c-48da-add2-8ed9a05edc89
#[derive(Debug, Clone)]
pub struct ConnectionActivationSequence {
    state: ConnectionActivationState,
    monitor_layout: Option<rdp::finalization_messages::MonitorLayoutPdu>,
    config: Config,
    // The MCS channel IDs are invariant for the whole life of the sequence: they are negotiated
    // once and never change, even across a Deactivation-Reactivation Sequence. They are stored
    // here (rather than duplicated into every state variant).
    io_channel_id: u16,
    user_channel_id: u16,
    /// TESSERA PATCH — virtual-channel PDUs that arrived mid-sequence, in
    /// wire order, for the caller to replay once the sequence finalizes.
    /// See the guard at the top of [`Sequence::step`].
    deferred_virtual_channel_pdus: Vec<Vec<u8>>,
    /// Running total of [`Self::deferred_virtual_channel_pdus`], against
    /// [`DEFERRED_PDU_BUDGET`].
    deferred_bytes: usize,
    /// PDUs dropped after the budget was exhausted. Non-zero means the
    /// replay is incomplete — see the "Scope" note in `rdp/ironrdp-fork.md`'s
    /// Guard 2 section for how a drop mid-fragment can leave the cliprdr
    /// channel's chunk reassembler wedged for the rest of the session.
    ///
    /// TESSERA PATCH — also doubles as the drop latch (see
    /// [`Self::defer_virtual_channel_pdu`]): once this is non-zero, every
    /// later virtual-channel PDU is dropped too, never just the ones that
    /// individually overflow the budget. That is why it is scoped to this
    /// sequence's lifetime and deliberately NOT reset by
    /// [`Self::take_deferred_virtual_channel_pdus`] — a fresh sequence (and
    /// therefore a fresh counter) is constructed per reactivation, in
    /// `tessera-rdp-worker`'s `frame_pump.rs`, inside the `DeactivateAll` arm.
    /// Resetting it here would let a sequence resume deferring after it had
    /// already started dropping, reopening the gap Fix 1 closes.
    dropped_virtual_channel_pdus: u32,
    /// TESSERA PATCH — the MCS message channel id, when one was negotiated.
    ///
    /// Exists solely to keep the auto-detect probe-answer branch in the
    /// `CapabilitiesExchange` arm of [`Sequence::step`] reachable: that
    /// branch answers connect-time Auto-Detect Requests a Windows RDS host
    /// keeps sending on the message channel after `ConnectTimeAutoDetection`
    /// has closed. The message channel is not the I/O channel, so without
    /// this field the virtual-channel deferral guard at the top of `step`
    /// would swallow every probe before that branch ever saw it.
    message_channel_id: Option<u16>,
}

/// TESSERA PATCH — why [`ConnectionActivationSequence::virtual_channel_id_to_defer`]
/// returned `None` while the sequence was in a phase that reads from the
/// wire, i.e. why [`ConnectionActivationSequence::declined_defer_reason`]
/// returned `Some`.
///
/// Diagnostic only: nothing branches on this. It exists because the three
/// reasons are indistinguishable in a log otherwise, and two of them lead to
/// completely different fixes.
///
/// `pub`, not `pub(crate)`: exercised from `tessera-rdp-worker`'s own
/// integration test (`reactivation_survives_virtual_channel_pdu.rs`), which
/// lives in a different crate and cargo workspace — the same reason
/// [`ConnectionActivationState`] and [`DEFERRED_PDU_BUDGET`] are `pub`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeclinedDefer {
    /// `input` is not a Send Data Indication — it never reached a channel id.
    NotSendDataIndication,
    /// A Send Data Indication for the I/O channel: handled, not deferred.
    IoChannel,
    /// A Send Data Indication for the message channel: answered in place.
    MessageChannel,
}

impl ConnectionActivationSequence {
    pub fn new(config: Config, io_channel_id: u16, user_channel_id: u16, message_channel_id: Option<u16>) -> Self {
        // TODO/FIXME: Investigate whether we really need to carry around the whole `Config` struct.
        // RATIONALE(@CBenoit): Not very convenient when building in isolation.
        //   I doubt this type really needs every field there.
        Self {
            state: ConnectionActivationState::CapabilitiesExchange { probe_count: 0 },
            monitor_layout: None,
            config,
            io_channel_id,
            user_channel_id,
            deferred_virtual_channel_pdus: Vec::new(),
            deferred_bytes: 0,
            dropped_virtual_channel_pdus: 0,
            message_channel_id,
        }
    }

    /// TESSERA PATCH — the channel id of `input`, when it is a PDU this
    /// sequence must set aside rather than decode: a Send Data Indication for
    /// a channel other than the I/O channel, arriving in one of the two
    /// phases that read from the wire.
    ///
    /// `None` covers everything the sequence should handle normally,
    /// including an `input` that does not decode as a Send Data Indication at
    /// all — deciding on the channel id and never on "decoding failed" is
    /// what keeps a malformed I/O-channel PDU a protocol error.
    ///
    /// `pub`, not `pub(crate)` (TESSERA PATCH): exercised directly from
    /// `tessera-rdp-worker`'s `declined_defer_reason_agrees_with_virtual_channel_id_to_defer`
    /// test, which pins that this function and [`Self::declined_defer_reason`]
    /// — two independent re-implementations of the same MCS decode and
    /// channel-id comparison — never drift apart.
    pub fn virtual_channel_id_to_defer(&self, input: &[u8]) -> Option<u16> {
        if !matches!(
            self.state,
            ConnectionActivationState::CapabilitiesExchange { .. }
                | ConnectionActivationState::ConnectionFinalization { .. }
        ) {
            return None;
        }

        let ctx = ironrdp_pdu::mcs::decode_send_data_indication(input).ok()?;

        // TESSERA PATCH — the message channel is exempt: it carries
        // connect-time auto-detect probes the `CapabilitiesExchange` arm
        // below answers itself, not virtual-channel data to defer. Deferring
        // it here would make that branch unreachable and, on the initial
        // connection (where nothing calls `take_deferred_virtual_channel_pdus`),
        // silently drop the probe instead of answering it.
        if Some(ctx.channel_id) == self.message_channel_id {
            return None;
        }

        (ctx.channel_id != self.io_channel_id).then_some(ctx.channel_id)
    }

    /// TESSERA PATCH — why [`Self::virtual_channel_id_to_defer`] returned `None`
    /// while the sequence was in a phase that reads from the wire.
    ///
    /// Diagnostic only: nothing branches on this. It exists because the three
    /// reasons are indistinguishable in a log otherwise, and two of them lead
    /// to completely different fixes.
    pub fn declined_defer_reason(&self, input: &[u8]) -> Option<DeclinedDefer> {
        if !matches!(
            self.state,
            ConnectionActivationState::CapabilitiesExchange { .. }
                | ConnectionActivationState::ConnectionFinalization { .. }
        ) {
            return None;
        }
        self.classify_declined_defer(input)
    }

    /// TESSERA PATCH — the state-independent half of
    /// [`Self::declined_defer_reason`]: the phase check above needs `self.state`
    /// as it stood BEFORE `step`'s `mem::take` replaces it with `Consumed`,
    /// but the actual work — decoding `input` and comparing channel ids — is
    /// only worth paying for on an error path, which runs AFTER the take.
    /// Splitting the two lets `step` capture the phase for free (the match
    /// arm it is already inside IS that capture — see `step`'s
    /// `CapabilitiesExchange`/`ConnectionFinalization` arms) and defer this
    /// half into the error closures, where `self.io_channel_id` and
    /// `self.message_channel_id` are still valid: they are plain `Copy`
    /// fields the take never touches, only `self.state` is replaced.
    fn classify_declined_defer(&self, input: &[u8]) -> Option<DeclinedDefer> {
        let Ok(ctx) = ironrdp_pdu::mcs::decode_send_data_indication(input) else {
            return Some(DeclinedDefer::NotSendDataIndication);
        };
        if Some(ctx.channel_id) == self.message_channel_id {
            return Some(DeclinedDefer::MessageChannel);
        }
        if ctx.channel_id == self.io_channel_id {
            return Some(DeclinedDefer::IoChannel);
        }
        None
    }

    /// TESSERA PATCH — records a virtual-channel PDU that arrived while the
    /// sequence was running, to be replayed by the caller afterwards.
    ///
    /// Bounded by [`DEFERRED_PDU_BUDGET`]: a reactivation normally completes
    /// in milliseconds, but its caller allows it seconds, and a file transfer
    /// saturating the link for that long would otherwise be buffered in full.
    /// Past the budget the PDU is dropped and counted — losing channel data
    /// is bad, but it is recoverable in a way running the process out of
    /// memory is not.
    ///
    /// TESSERA PATCH — the first drop LATCHES: once
    /// [`Self::dropped_virtual_channel_pdus`] is non-zero, every later PDU is
    /// dropped too, even one small enough to fit in whatever budget remains.
    /// The gate is deliberately "have we dropped anything yet", not "does
    /// this PDU fit" — the latter lets a large PDU get rejected while a
    /// smaller one after it still fits, so what comes back from
    /// [`Self::take_deferred_virtual_channel_pdus`] would have a HOLE in the
    /// middle (PDU N-1 and N+1 present, N missing) rather than a truncated
    /// tail. The invariant this buys: what
    /// [`Self::take_deferred_virtual_channel_pdus`] returns is always a
    /// PREFIX of what arrived on the wire, never a stream with a gap.
    ///
    /// Reusing [`Self::dropped_virtual_channel_pdus`] as the latch (instead of
    /// a separate bool) is deliberate: it is already the field that tracks
    /// "has this sequence started dropping", and it is never reset except by
    /// constructing a fresh sequence, which is exactly the latch's required
    /// lifetime.
    fn defer_virtual_channel_pdu(&mut self, channel_id: u16, input: &[u8]) {
        if self.dropped_virtual_channel_pdus > 0 || self.deferred_bytes + input.len() > DEFERRED_PDU_BUDGET {
            self.dropped_virtual_channel_pdus += 1;
            // TESSERA PATCH — one-shot: this fires exactly once per sequence,
            // on the drop that flips the latch from 0 to 1. Every drop after
            // that is the same fact restated, so logging it again would flood
            // a saturated link with thousands of identical lines in
            // milliseconds. The running total is reported once, by the frame
            // pump's own summary `warn!` when the sequence finalizes — this
            // line only needs to announce that dropping has started and why.
            if self.dropped_virtual_channel_pdus == 1 {
                warn!(
                    channel_id,
                    budget = DEFERRED_PDU_BUDGET,
                    "Virtual-channel PDU dropped mid-activation: deferred-PDU budget exhausted; every subsequent virtual-channel PDU this sequence sees will be dropped too"
                );
            }
            return;
        }

        debug!(
            channel_id,
            len = input.len(),
            "Deferring a virtual-channel PDU received during connection activation"
        );
        self.deferred_bytes += input.len();
        self.deferred_virtual_channel_pdus.push(input.to_vec());
    }

    /// TESSERA PATCH — takes the virtual-channel PDUs deferred during the
    /// sequence, in wire order, leaving the buffer empty.
    ///
    /// The caller must feed these to the active stage once the sequence has
    /// finalized: they are ordinary channel data that merely arrived while
    /// the I/O channel was busy re-negotiating, and nothing will resend them.
    ///
    /// TESSERA PATCH — thanks to the drop latch in
    /// [`Self::defer_virtual_channel_pdu`], what this returns is always a
    /// PREFIX of what arrived on the wire, never a stream with a gap in the
    /// middle. This method deliberately does NOT reset
    /// [`Self::dropped_virtual_channel_pdus`] — that counter is scoped to
    /// this sequence's whole lifetime, not to one `take`, because it doubles
    /// as the latch that keeps the prefix invariant true for every PDU still
    /// to come. A fresh sequence (and therefore a fresh, unlatched counter)
    /// is constructed per reactivation in `tessera-rdp-worker`'s
    /// `frame_pump.rs`, inside the `DeactivateAll` arm.
    #[must_use]
    pub fn take_deferred_virtual_channel_pdus(&mut self) -> Vec<Vec<u8>> {
        // Zeroing the byte counter here while leaving `dropped_virtual_channel_pdus`
        // (the latch) armed is safe, not an asymmetry: `deferred_bytes` only
        // gates whether a FUTURE PDU still fits under `DEFERRED_PDU_BUDGET`,
        // and once the latch is armed, `defer_virtual_channel_pdu` drops
        // every later PDU before it ever reaches that accounting — so no
        // later PDU can observe `deferred_bytes` having been reset.
        self.deferred_bytes = 0;
        mem::take(&mut self.deferred_virtual_channel_pdus)
    }

    /// TESSERA PATCH — how many virtual-channel PDUs were dropped for want of
    /// budget. Non-zero means [`Self::take_deferred_virtual_channel_pdus`]
    /// returns an incomplete replay — specifically a PREFIX of the wire
    /// stream (see [`Self::defer_virtual_channel_pdu`]'s drop latch), never a
    /// stream with a gap. Scoped to this sequence's whole lifetime: it is not
    /// reset by `take`, only by constructing a fresh sequence, which is what
    /// `tessera-rdp-worker`'s frame pump does per reactivation (the
    /// `DeactivateAll` arm in `frame_pump.rs`).
    #[must_use]
    pub fn dropped_virtual_channel_pdus(&self) -> u32 {
        self.dropped_virtual_channel_pdus
    }

    pub fn io_channel_id(&self) -> u16 {
        self.io_channel_id
    }

    pub fn user_channel_id(&self) -> u16 {
        self.user_channel_id
    }

    /// Returns the current state as a distinct type, rather than `&dyn State` provided by [`Self::state`].
    pub fn connection_activation_state(&self) -> ConnectionActivationState {
        self.state.clone()
    }

    /// Returns the server-reported monitor layout received during this activation.
    pub fn monitor_layout(&self) -> Option<rdp::finalization_messages::MonitorLayoutPdu> {
        self.monitor_layout.clone()
    }
}

/// Factory producing fresh [`ConnectionActivationSequence`] instances.
///
/// The [`Config`] and MCS channel IDs required to build a connection activation sequence are
/// invariant for the whole lifetime of the connection: they are negotiated once and never change,
/// even across a [Deactivation-Reactivation Sequence]. This factory captures them so that a fresh,
/// correctly-initialized sequence can be produced each time one is needed, driven until it is
/// finalized, then dropped.
///
/// [Deactivation-Reactivation Sequence]: https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-rdpbcgr/dfc234ce-481a-4674-9a5d-2a7bafb14432
#[derive(Debug, Clone)]
pub struct ConnectionActivationFactory {
    config: Config,
    io_channel_id: u16,
    user_channel_id: u16,
    /// TESSERA PATCH — see [`ConnectionActivationSequence::message_channel_id`];
    /// threaded through so every sequence [`Self::create`] produces (including
    /// the ones driven across a Deactivation-Reactivation Sequence) keeps the
    /// auto-detect probe-answer branch reachable.
    message_channel_id: Option<u16>,
}

impl ConnectionActivationFactory {
    pub fn new(config: Config, io_channel_id: u16, user_channel_id: u16, message_channel_id: Option<u16>) -> Self {
        Self {
            config,
            io_channel_id,
            user_channel_id,
            message_channel_id,
        }
    }

    pub fn io_channel_id(&self) -> u16 {
        self.io_channel_id
    }

    pub fn user_channel_id(&self) -> u16 {
        self.user_channel_id
    }

    /// Produces a fresh [`ConnectionActivationSequence`] in the initial `CapabilitiesExchange` state.
    #[must_use]
    pub fn create(&self) -> ConnectionActivationSequence {
        ConnectionActivationSequence::new(
            self.config.clone(),
            self.io_channel_id,
            self.user_channel_id,
            self.message_channel_id,
        )
    }
}

impl Sequence for ConnectionActivationSequence {
    fn next_pdu_hint(&self) -> Option<&dyn ironrdp_pdu::PduHint> {
        match &self.state {
            ConnectionActivationState::Consumed => None,
            ConnectionActivationState::Finalized { .. } => None,
            ConnectionActivationState::CapabilitiesExchange { .. } => Some(&ironrdp_pdu::X224_HINT),
            ConnectionActivationState::ConnectionFinalization {
                connection_finalization,
                ..
            } => connection_finalization.next_pdu_hint(),
        }
    }

    fn state(&self) -> &dyn State {
        &self.state
    }

    fn step(
        &mut self,
        input: &[u8],
        received_at: Option<MonotonicInstant>,
        output: &mut ironrdp_core::WriteBuf,
    ) -> ConnectorResult<Written> {
        // TESSERA PATCH — see `rdp/ironrdp-fork.md`.
        //
        // A Deactivation-Reactivation Sequence (§1.3.1.3) runs on the SAME
        // wire the static virtual channels use, and the server does not stop
        // servicing them while it runs. Both phases below decode whatever
        // they read as an I/O-channel PDU — `CapabilitiesExchange` through
        // `decode_share_control`, `ConnectionFinalization` through
        // `decode_share_data` — and only look at the channel id afterwards
        // (or not at all). A cliprdr chunk arriving mid-reactivation
        // therefore died at the decode:
        //
        //     InvalidField { field: "pdu_type", reason: "invalid pdu type" }
        //
        // which is what killed two live sessions mid file-copy (worker logs
        // 2026-08-14 06:09:46 and 12:36:02). Skip-and-stay is the shape this
        // file already applies to `ServerDeactivateAll` and auto-detect
        // probes; the difference is that a virtual-channel PDU carries data
        // nobody else will resend, so it is DEFERRED rather than dropped and
        // replayed into the active stage once reactivation finishes.
        //
        // Keyed on the channel id, never on "decoding failed": garbage on the
        // I/O channel must stay a protocol error, or a malformed Share
        // Control PDU would silently stall the sequence until its caller's
        // reactivation timeout.
        if let Some(channel_id) = self.virtual_channel_id_to_defer(input) {
            self.defer_virtual_channel_pdu(channel_id, input);
            return Ok(Written::Nothing);
        }

        // TESSERA PATCH — diagnostic. Only `self.state` is invalidated by the
        // `mem::take` below — `self.io_channel_id`/`self.message_channel_id`
        // are plain `Copy` fields it never touches — so only the state check
        // needs to happen before it, and that check is already free: the
        // match arms below (`CapabilitiesExchange { .. }` /
        // `ConnectionFinalization { .. }`) ARE that check, encoded in which
        // arm ends up running. Each error closure that needs the
        // classification calls `self.classify_declined_defer(input)` — the
        // decode-and-classify half with no state check of its own — directly,
        // lazily, only on the error path. No `warn!` here either: logging at
        // this point would fire on every ordinary I/O-channel PDU during
        // Capabilities Exchange / Connection Finalization, the dominant
        // SUCCESSFUL path of every connection (not just a reactivation).
        let (written, next_state) = match mem::take(&mut self.state) {
            ConnectionActivationState::Consumed | ConnectionActivationState::Finalized { .. } => {
                return Err(general_err!(
                    "connector sequence state is finalized or consumed (this is a bug)"
                ));
            }
            ConnectionActivationState::CapabilitiesExchange { probe_count } => {
                debug!("Capabilities Exchange");

                let send_data_indication_ctx = ironrdp_pdu::mcs::decode_send_data_indication(input).map_err(|e| {
                    // TESSERA PATCH — diagnostic, on the error path only.
                    // `ConnectorErrorKind::Decode` displays as the bare
                    // string "decode error" and drops `source()` on
                    // `to_string()`, so without this a crash here says
                    // nothing about WHICH decode failed or what was on the
                    // wire.
                    if let Some(reason) = self.classify_declined_defer(input) {
                        let prefix_len = input.len().min(16);
                        warn!(
                            ?reason,
                            len = input.len(),
                            prefix = %hex_prefix(&input[..prefix_len]),
                            "Not deferring a PDU received during connection activation"
                        );
                    }
                    ConnectorError::decode(e)
                })?;

                // TESSERA PATCH — see `rdp/ironrdp-fork.md`.
                //
                // `ClientConnector` answers auto-detect only in its own
                // `ConnectTimeAutoDetection` phase, which ends at the first PDU
                // off the message channel. A Windows RDS host here keeps probing
                // afterwards. `ClientConnector::respond_to_connect_time_autodetect`
                // is a private method on `ClientConnector` that this sequence
                // cannot reach (it takes `&mut self` to mutate the connect-time
                // bandwidth window: see #1530/#1559), so the RTT request — the
                // only variant that needs an unconditional answer here and needs
                // no connector state to produce one — is answered locally.
                // [MS-RDPBCGR] 3.2.5.14 routes the Bandwidth Measure variants
                // that can arrive outside connect-time to a multitransport
                // procedure with a sequence-number correlation this phase does
                // not track, so ignoring them is the conservative response — the
                // same conclusion upstream reached in #1559 for the connect-time
                // phase. Answer and stay in this state, still waiting for Server
                // Demand Active — the same skip-and-stay shape already applied
                // to `ServerDeactivateAll` a few lines below.
                //
                // `probe_count` mirrors `license_exchange::log_skipped_non_license_pdu`'s
                // skip counter: nothing bounds how many times this branch can
                // fire (that is `CONNECT_PHASE_TIMEOUT` in tessera-rdp-worker's
                // job), so escalate to `warn!` once it repeats a few times, to
                // leave a visible trail in a release run.
                if let Some(message_channel_id) = self.message_channel_id {
                    if send_data_indication_ctx.channel_id == message_channel_id {
                        if let Ok(request) =
                            send_data_indication_ctx.decode_user_data::<rdp::autodetect::AutoDetectReqPdu>()
                        {
                            if let rdp::autodetect::AutoDetectRequest::RttRequest { sequence_number, .. } =
                                request.request
                            {
                                let probe_count = probe_count + 1;
                                if probe_count > CAPABILITIES_PROBE_WARN_THRESHOLD {
                                    warn!(
                                        probe_count,
                                        sequence_number,
                                        "repeatedly answering RTT auto-detect requests received during Capabilities Exchange; still waiting for Server Demand Active"
                                    );
                                } else {
                                    debug!(
                                        probe_count,
                                        sequence_number,
                                        "Answering an RTT auto-detect request received during Capabilities Exchange"
                                    );
                                }
                                let response = rdp::autodetect::AutoDetectRspPdu::new(
                                    rdp::autodetect::AutoDetectResponse::RttResponse { sequence_number },
                                );
                                let written = encode_send_data_request(
                                    self.user_channel_id,
                                    message_channel_id,
                                    &response,
                                    output,
                                )?;
                                self.state = ConnectionActivationState::CapabilitiesExchange { probe_count };
                                return Written::from_size(written);
                            }
                        }
                        self.state = ConnectionActivationState::CapabilitiesExchange { probe_count };
                        return Ok(Written::Nothing);
                    }
                }

                let share_control_ctx = rdp::headers::decode_share_control(send_data_indication_ctx).map_err(|e| {
                    // TESSERA PATCH — diagnostic, on the error path only; see
                    // the comment above the decode a few lines up.
                    if let Some(reason) = self.classify_declined_defer(input) {
                        let prefix_len = input.len().min(16);
                        warn!(
                            ?reason,
                            len = input.len(),
                            prefix = %hex_prefix(&input[..prefix_len]),
                            "Not deferring a PDU received during connection activation"
                        );
                    }
                    ConnectorError::decode(e)
                })?;

                debug!(message = ?share_control_ctx.pdu, "Received");

                if share_control_ctx.channel_id != self.io_channel_id {
                    warn!(
                        io_channel_id = self.io_channel_id,
                        share_control_ctx.channel_id, "Unexpected channel ID for received Share Control Pdu"
                    );
                }

                // Some servers (e.g. GNOME Remote Desktop) send a ServerDeactivateAll PDU
                // before ServerDemandActive as part of a Deactivation-Reactivation Sequence
                // (MS-RDPBCGR §1.3.1.3). Skip it and stay in the same state to wait for
                // the actual DemandActive PDU.
                //
                // The decoded PDU is intentionally discarded: the DeactivateAll body carries
                // no payload we need during initial activation.
                if matches!(
                    share_control_ctx.pdu,
                    rdp::headers::ShareControlPdu::ServerDeactivateAll(_)
                ) {
                    debug!(
                        "Skipping Server Deactivate All PDU received during Capabilities Exchange, awaiting Server Demand Active"
                    );
                    self.state = ConnectionActivationState::CapabilitiesExchange { probe_count };
                    return Ok(Written::Nothing);
                }

                let capability_sets = if let rdp::headers::ShareControlPdu::ServerDemandActive(server_demand_active) =
                    share_control_ctx.pdu
                {
                    server_demand_active.pdu.capability_sets
                } else {
                    // Instead of reactivating after a Deactivate-All, a server may end the
                    // session (MS-RDPBCGR §1.3.1.3) by sending a Set Error Info PDU carrying the
                    // disconnect reason. FreeRDP-based servers such as GNOME Remote Desktop do
                    // this, for example when the backend screencast session cannot be created.
                    // Surface that reason so the disconnect is explained rather than reported as
                    // an unexpected PDU.
                    if let rdp::headers::ShareControlPdu::Data(rdp::headers::ShareDataHeader {
                        share_data_pdu:
                            rdp::headers::ShareDataPdu::ServerSetErrorInfo(rdp::server_error_info::ServerSetErrorInfoPdu(
                                error_info,
                            )),
                        ..
                    }) = share_control_ctx.pdu
                    {
                        // ERRINFO_NONE is informational (it clears a previously reported error),
                        // not a disconnect. Skip it and keep waiting for the Demand Active PDU,
                        // matching how the connection finalization sequence treats it.
                        if let rdp::server_error_info::ErrorInfo::ProtocolIndependentCode(
                            rdp::server_error_info::ProtocolIndependentCode::None,
                        ) = error_info
                        {
                            self.state = ConnectionActivationState::CapabilitiesExchange { probe_count };
                            return Ok(Written::Nothing);
                        }

                        return Err(reason_err!(
                            "ConnectionActivation::CapabilitiesExchange",
                            "server ended the session with error info: {}",
                            error_info.description()
                        ));
                    }

                    return Err(reason_err!(
                        "ConnectionActivation::CapabilitiesExchange",
                        "unexpected Share Control PDU during capabilities exchange: got {} (expected Server Demand Active PDU)",
                        rdp::headers::describe_unexpected_share_control_pdu(&share_control_ctx.pdu),
                    ));
                };

                let window_list = server_window_list(&capability_sets);
                let window_support_level = negotiated_window_support_level(window_list.as_ref());

                let (refresh_rect_support, suppress_output_support) = capability_sets
                    .iter()
                    .find_map(|capability_set| {
                        let CapabilitySet::General(general) = capability_set else {
                            return None;
                        };
                        if general.protocol_version != rdp::capability_sets::PROTOCOL_VER {
                            warn!(version = general.protocol_version, "Unexpected protocol version");
                        }
                        Some((general.refresh_rect_support, general.suppress_output_support))
                    })
                    .unwrap_or((false, false));

                // Keep the server's Input capability flags so the session layer can tell whether
                // fast-path input was negotiated: per [MS-RDPBCGR] 2.2.8.1.2, the client MUST NOT
                // send fast-path input events unless the server advertised
                // `INPUT_FLAG_FASTPATH_INPUT` or `INPUT_FLAG_FASTPATH_INPUT2`. Servers that omit
                // both (e.g. VirtualBox VRDP) may reject fast-path input PDUs outright.
                let input_flags = capability_sets
                    .iter()
                    .find_map(|c| match c {
                        CapabilitySet::Input(i) => Some(i.input_flags),
                        _ => None,
                    })
                    .unwrap_or_else(InputFlags::empty);

                let static_channel_chunk_size = capability_sets
                    .iter()
                    .find_map(|c| match c {
                        CapabilitySet::VirtualChannel(channel) => channel
                            .chunk_size
                            .and_then(|chunk_size| usize::try_from(chunk_size).ok()),
                        _ => None,
                    })
                    .filter(|chunk_size| {
                        (ironrdp_svc::CHANNEL_CHUNK_LENGTH..=ironrdp_svc::MAX_CHANNEL_CHUNK_LENGTH).contains(chunk_size)
                    })
                    .unwrap_or(ironrdp_svc::CHANNEL_CHUNK_LENGTH);

                // At this point we have already sent a requested desktop size to the server -- either as a part of the
                // [`TS_UD_CS_CORE`] (on initial connection) or the [`DISPLAYCONTROL_MONITOR_LAYOUT`] (on resize event).
                //
                // The server is therefore responding with a desktop size here, which will be close to the requested size but
                // may be slightly different due to server-side constraints. We should use this negotiated size for the rest of
                // the session.
                //
                // [TS_UD_CS_CORE]: https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-rdpbcgr/00f1da4a-ee9c-421a-852f-c19f92343d73
                // [DISPLAYCONTROL_MONITOR_LAYOUT]: https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-rdpedisp/ea2de591-9203-42cd-9908-be7a55237d1c
                let desktop_size = capability_sets
                    .iter()
                    .find_map(|c| match c {
                        CapabilitySet::Bitmap(b) => Some(DesktopSize {
                            width: b.desktop_width,
                            height: b.desktop_height,
                        }),
                        _ => None,
                    })
                    .unwrap_or(DesktopSize {
                        width: self.config.desktop_size.width,
                        height: self.config.desktop_size.height,
                    });

                let share_id = share_control_ctx.share_id;

                let client_confirm_active = rdp::headers::ShareControlPdu::ClientConfirmActive(
                    create_client_confirm_active(&self.config, capability_sets, desktop_size, window_list)?,
                );

                debug!(message = ?client_confirm_active, "Send");

                let written = rdp::headers::encode_share_control(
                    self.user_channel_id,
                    self.io_channel_id,
                    share_id,
                    client_confirm_active,
                    output,
                )
                .map_err(ConnectorError::encode)?;

                (
                    Written::from_size(written)?,
                    ConnectionActivationState::ConnectionFinalization {
                        desktop_size,
                        share_id,
                        input_flags,
                        static_channel_chunk_size,
                        refresh_rect_support,
                        suppress_output_support,
                        window_support_level,
                        connection_finalization: ConnectionFinalizationSequence::new(
                            self.io_channel_id,
                            self.user_channel_id,
                            share_id,
                        ),
                    },
                )
            }
            ConnectionActivationState::ConnectionFinalization {
                desktop_size,
                share_id,
                input_flags,
                static_channel_chunk_size,
                refresh_rect_support,
                suppress_output_support,
                window_support_level,
                mut connection_finalization,
            } => {
                debug!("Connection Finalization");

                // TESSERA PATCH — diagnostic. `ConnectionFinalizationSequence`
                // is a distinct vendored struct (`connection_finalization.rs`)
                // that does its own `decode_send_data_indication` /
                // `decode_share_data` and still maps a decode failure through
                // a plain `ConnectorError::decode` with no classification —
                // the exact bare "decode error" this diagnostic exists to
                // eliminate, and reachable here: `virtual_channel_id_to_defer`
                // and `declined_defer_reason` both already cover the
                // `ConnectionFinalization` state, so a virtual-channel PDU
                // arriving during THIS phase of a Deactivation-Reactivation
                // Sequence is deferred exactly like one arriving during
                // `CapabilitiesExchange` and never reaches this call at all —
                // what lands here on the error path is genuinely undeferred
                // wire garbage or a decode this sequence cannot classify,
                // worth the same context.
                //
                // Scoped to `Decode`-kind failures only: `step` also returns
                // `Encode` errors on its send-side states, which have nothing
                // to do with an undeferred incoming PDU and would make this
                // diagnostic misattribute an encode bug as "not deferring a
                // PDU received".
                let written = connection_finalization
                    .step(input, received_at, output)
                    .inspect_err(|e| {
                        if matches!(e.kind(), crate::ConnectorErrorKind::Decode(_)) {
                            if let Some(reason) = self.classify_declined_defer(input) {
                                let prefix_len = input.len().min(16);
                                warn!(
                                    ?reason,
                                    len = input.len(),
                                    prefix = %hex_prefix(&input[..prefix_len]),
                                    "Not deferring a PDU received during connection finalization"
                                );
                            }
                        }
                    })?;

                let next_state = if !connection_finalization.state.is_terminal() {
                    ConnectionActivationState::ConnectionFinalization {
                        desktop_size,
                        share_id,
                        input_flags,
                        static_channel_chunk_size,
                        refresh_rect_support,
                        suppress_output_support,
                        window_support_level,
                        connection_finalization,
                    }
                } else {
                    self.monitor_layout = connection_finalization.monitor_layout;
                    ConnectionActivationState::Finalized {
                        desktop_size,
                        share_id,
                        input_flags,
                        static_channel_chunk_size,
                        enable_server_pointer: self.config.enable_server_pointer,
                        pointer_software_rendering: self.config.pointer_software_rendering,
                        refresh_rect_support,
                        suppress_output_support,
                        window_support_level,
                    }
                };

                (written, next_state)
            }
        };

        self.state = next_state;

        Ok(written)
    }
}

#[derive(Default, Debug, Clone)]
pub enum ConnectionActivationState {
    #[default]
    Consumed,
    CapabilitiesExchange {
        /// TESSERA PATCH — count of RTT auto-detect probes answered while
        /// waiting in this state; see `step`'s `CapabilitiesExchange` arm and
        /// `license_exchange::log_skipped_non_license_pdu`.
        probe_count: u32,
    },
    ConnectionFinalization {
        desktop_size: DesktopSize,
        share_id: u32,
        /// The server's Input capability flags from the Server Demand Active PDU.
        input_flags: InputFlags,
        /// The validated `VCChunkSize` from the server Virtual Channel Capability Set.
        static_channel_chunk_size: usize,
        refresh_rect_support: bool,
        suppress_output_support: bool,
        /// The server-negotiated Window List support level.
        window_support_level: Option<WindowSupportLevel>,
        connection_finalization: ConnectionFinalizationSequence,
    },
    Finalized {
        desktop_size: DesktopSize,
        share_id: u32,
        /// The server's Input capability flags from the Server Demand Active PDU.
        ///
        /// Per [MS-RDPBCGR] 2.2.8.1.2, fast-path input events may only be sent when
        /// `INPUT_FLAG_FASTPATH_INPUT` or `INPUT_FLAG_FASTPATH_INPUT2` is present.
        ///
        /// [MS-RDPBCGR]: https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-rdpbcgr/b8e7c588-51cb-455b-bb73-92d480903133
        input_flags: InputFlags,
        /// The validated `VCChunkSize` from the server Virtual Channel Capability Set.
        static_channel_chunk_size: usize,
        enable_server_pointer: bool,
        pointer_software_rendering: bool,
        /// Whether the server permits client Refresh Rect PDUs for visual recovery.
        refresh_rect_support: bool,
        /// Whether the server permits Suppress Output PDUs for visual recovery.
        suppress_output_support: bool,
        /// The server-negotiated Window List support level.
        window_support_level: Option<WindowSupportLevel>,
    },
}

impl State for ConnectionActivationState {
    fn name(&self) -> &'static str {
        match self {
            ConnectionActivationState::Consumed => "Consumed",
            ConnectionActivationState::CapabilitiesExchange { .. } => "CapabilitiesExchange",
            ConnectionActivationState::ConnectionFinalization { .. } => "ConnectionFinalization",
            ConnectionActivationState::Finalized { .. } => "Finalized",
        }
    }

    fn is_terminal(&self) -> bool {
        matches!(self, ConnectionActivationState::Finalized { .. })
    }

    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

const DEFAULT_POINTER_CACHE_SIZE: u16 = 32;

fn server_window_list(capability_sets: &[CapabilitySet]) -> Option<WindowList> {
    capability_sets.iter().find_map(|capability_set| {
        if let CapabilitySet::WindowList(window_list) = capability_set {
            Some(window_list.clone())
        } else {
            None
        }
    })
}

fn negotiated_window_support_level(window_list: Option<&WindowList>) -> Option<WindowSupportLevel> {
    window_list.and_then(|window_list| {
        (window_list.support_level != WindowSupportLevel::NotSupported).then_some(window_list.support_level)
    })
}

fn remote_app_rail_capability(
    remote_application_mode: bool,
    rail_support_level: RailSupportLevel,
    server_capability_sets: &[CapabilitySet],
    window_list: Option<&WindowList>,
) -> ConnectorResult<Option<Rail>> {
    if !remote_application_mode {
        return Ok(None);
    }
    if !rail_support_level.contains(RailSupportLevel::SUPPORTED) {
        return Err(reason_err!(
            "Capabilities Exchange",
            "client RemoteApp configuration does not support remote programs"
        ));
    }

    let rail_supported = server_capability_sets.iter().any(|capability_set| {
        matches!(
            capability_set,
            CapabilitySet::Rail(rail) if rail.support_level.contains(RailSupportLevel::SUPPORTED)
        )
    });
    let window_list_supported =
        window_list.is_some_and(|window_list| window_list.support_level != WindowSupportLevel::NotSupported);
    if !rail_supported || !window_list_supported {
        return Err(reason_err!(
            "Capabilities Exchange",
            "server does not support required RemoteApp capabilities"
        ));
    }

    Ok(Some(Rail {
        support_level: rail_support_level,
    }))
}

fn create_client_confirm_active(
    config: &Config,
    mut server_capability_sets: Vec<CapabilitySet>,
    desktop_size: DesktopSize,
    window_list: Option<WindowList>,
) -> ConnectorResult<rdp::capability_sets::ClientConfirmActive> {
    use ironrdp_pdu::rdp::capability_sets::{
        BITMAP_CACHE_ENTRIES_NUM, Bitmap, BitmapCache, BitmapDrawingFlags, Brush, CacheDefinition, CacheEntry,
        ClientConfirmActive, CmdFlags, DemandActive, FrameAcknowledge, GLYPH_CACHE_NUM, General, GeneralExtraFlags,
        GlyphCache, GlyphSupportLevel, Input, LargePointer, LargePointerSupportFlags, MultifragmentUpdate,
        OffscreenBitmapCache, Order, OrderFlags, OrderSupportExFlags, Pointer, SERVER_CHANNEL_ID, Sound, SoundFlags,
        SupportLevel, SurfaceCommands, VirtualChannel, VirtualChannelFlags, client_codecs_capabilities,
    };

    let remote_app_rail_capability = remote_app_rail_capability(
        config.remote_application_mode,
        config.rail_support_level,
        &server_capability_sets,
        window_list.as_ref(),
    )?;

    server_capability_sets.retain(|capability_set| matches!(capability_set, CapabilitySet::MultiFragmentUpdate(_)));

    let lossy_bitmap_compression = config
        .bitmap
        .as_ref()
        .map(|bitmap| bitmap.lossy_compression)
        .unwrap_or(false);
    let pref_bits_per_pix = requested_bitmap_color_depth(config.bitmap.as_ref())?;

    let drawing_flags = if lossy_bitmap_compression {
        BitmapDrawingFlags::ALLOW_SKIP_ALPHA
            | BitmapDrawingFlags::ALLOW_DYNAMIC_COLOR_FIDELITY
            | BitmapDrawingFlags::ALLOW_COLOR_SUBSAMPLING
    } else {
        BitmapDrawingFlags::ALLOW_SKIP_ALPHA
    };

    server_capability_sets.extend_from_slice(&[
        CapabilitySet::General(General {
            major_platform_type: config.platform,
            extra_flags: GeneralExtraFlags::FASTPATH_OUTPUT_SUPPORTED | GeneralExtraFlags::NO_BITMAP_COMPRESSION_HDR,
            ..Default::default()
        }),
        CapabilitySet::Bitmap(Bitmap {
            pref_bits_per_pix,
            desktop_width: desktop_size.width,
            desktop_height: desktop_size.height,
            // This is required to be true in order for the Microsoft::Windows::RDS::DisplayControl DVC to work.
            desktop_resize_flag: true,
            drawing_flags,
        }),
        CapabilitySet::Order(Order::new(
            OrderFlags::NEGOTIATE_ORDER_SUPPORT | OrderFlags::ZERO_BOUNDS_DELTAS_SUPPORT,
            OrderSupportExFlags::empty(),
            0,
            0,
        )),
        CapabilitySet::BitmapCache(BitmapCache {
            caches: [CacheEntry {
                entries: 0,
                max_cell_size: 0,
            }; BITMAP_CACHE_ENTRIES_NUM],
        }),
        CapabilitySet::Input(Input {
            input_flags: InputFlags::all(),
            keyboard_layout: 0,
            keyboard_type: Some(config.keyboard_type),
            keyboard_subtype: config.keyboard_subtype,
            keyboard_function_key: config.keyboard_functional_keys_count,
            keyboard_ime_filename: config.ime_file_name.clone(),
        }),
        CapabilitySet::Pointer(Pointer {
            // Pointer cache should be set to non-zero value to enable client-side pointer rendering.
            color_pointer_cache_size: DEFAULT_POINTER_CACHE_SIZE,
            pointer_cache_size: DEFAULT_POINTER_CACHE_SIZE,
        }),
        CapabilitySet::Brush(Brush {
            support_level: SupportLevel::Default,
        }),
        CapabilitySet::GlyphCache(GlyphCache {
            glyph_cache: [CacheDefinition {
                entries: 0,
                max_cell_size: 0,
            }; GLYPH_CACHE_NUM],
            frag_cache: CacheDefinition {
                entries: 0,
                max_cell_size: 0,
            },
            glyph_support_level: GlyphSupportLevel::None,
        }),
        CapabilitySet::OffscreenBitmapCache(OffscreenBitmapCache {
            is_supported: false,
            cache_size: 0,
            cache_entries: 0,
        }),
        CapabilitySet::VirtualChannel(VirtualChannel {
            flags: VirtualChannelFlags::NO_COMPRESSION,
            chunk_size: Some(0), // ignored
        }),
        CapabilitySet::Sound(Sound {
            flags: SoundFlags::empty(),
        }),
        CapabilitySet::LargePointer(LargePointer {
            // Setting `LargePointerSupportFlags::UP_TO_384X384_PIXELS` allows server to send
            // `TS_FP_LARGEPOINTERATTRIBUTE` update messages, which are required for client-side
            // rendering of pointers bigger than 96x96 pixels.
            // `LargePointerSupportFlags::UP_TO_96X96_PIXELS` is needed for proper cursor behavior
            // in Windows 2019 and older
            flags: LargePointerSupportFlags::UP_TO_96X96_PIXELS | LargePointerSupportFlags::UP_TO_384X384_PIXELS,
        }),
        CapabilitySet::SurfaceCommands(SurfaceCommands {
            flags: CmdFlags::SET_SURFACE_BITS | CmdFlags::STREAM_SURFACE_BITS | CmdFlags::FRAME_MARKER,
        }),
        CapabilitySet::BitmapCodecs(match config.bitmap.as_ref().map(|b| b.codecs.clone()) {
            Some(codecs) => codecs,
            None => client_codecs_capabilities(&[]).expect("can't panic for &[]"),
        }),
        CapabilitySet::FrameAcknowledge(FrameAcknowledge {
            // FIXME(#447): Revert this to 2 per FreeRDP.
            // This is a temporary hack to fix a resize bug, see:
            // https://github.com/Devolutions/IronRDP/issues/447
            max_unacknowledged_frame_count: 20,
        }),
    ]);

    if !server_capability_sets
        .iter()
        .any(|c| matches!(&c, CapabilitySet::MultiFragmentUpdate(_)))
    {
        server_capability_sets.push(CapabilitySet::MultiFragmentUpdate(MultifragmentUpdate {
            max_request_size: 8 * 1024 * 1024, // 8 MB
        }));
    }
    if let Some(rail) = remote_app_rail_capability {
        server_capability_sets.push(CapabilitySet::Rail(rail));
    }
    if let Some(window_list) = window_list {
        server_capability_sets.push(CapabilitySet::WindowList(window_list));
    }

    Ok(ClientConfirmActive {
        originator_id: SERVER_CHANNEL_ID,
        pdu: DemandActive {
            source_descriptor: "IRONRDP".to_owned(),
            capability_sets: server_capability_sets,
        },
    })
}

/// Returns the color depth requested by the client configuration for the Bitmap Capability Set.
///
/// [MS-RDPBCGR] requires `preferredBitsPerPixel` to match the requested Client Core Data color
/// depth. Keeping these values aligned is particularly important for non-32-bpp clients: a 32-bpp
/// Bitmap Capability Set permits the server to select RDP 6.0 bitmap compression instead.
///
/// [MS-RDPBCGR]: https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-rdpbcgr/49e7bcb9-a8d7-46f5-987e-46c63c44b2c4
fn requested_bitmap_color_depth(bitmap: Option<&crate::BitmapConfig>) -> ConnectorResult<u16> {
    match bitmap.map_or(32, |bitmap| bitmap.color_depth) {
        15 => Ok(15),
        16 => Ok(16),
        24 => Ok(24),
        32 => Ok(32),
        color_depth => Err(reason_err!(
            "create client confirm active",
            "unsupported bitmap color depth: {color_depth}"
        )),
    }
}

/// TESSERA PATCH — lowercase hex of a short byte prefix, for diagnostics.
/// Hand-rolled rather than pulling in `hex`: this is a vendored fork and a
/// new dependency here would have to be justified against upstream on every
/// rebase.
fn hex_prefix(bytes: &[u8]) -> String {
    use core::fmt::Write as _;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

#[cfg(test)]
mod tests {
    use ironrdp_pdu::rdp::capability_sets::{CapabilitySet, Rail, RailSupportLevel, WindowList, WindowSupportLevel};

    use super::{
        negotiated_window_support_level, remote_app_rail_capability, requested_bitmap_color_depth, server_window_list,
    };
    use crate::BitmapConfig;

    #[test]
    fn bitmap_capability_uses_requested_color_depth() {
        assert_eq!(requested_bitmap_color_depth(None).unwrap(), 32);
        for expected_color_depth in [15, 16, 24, 32] {
            let bitmap = BitmapConfig {
                color_depth: u32::from(expected_color_depth),
                lossy_compression: false,
                codecs: ironrdp_pdu::rdp::capability_sets::BitmapCodecs(Vec::new()),
            };

            assert_eq!(
                requested_bitmap_color_depth(Some(&bitmap)).unwrap(),
                expected_color_depth
            );
        }
    }

    #[test]
    fn window_list_capability_preserves_supported_level() {
        let window_list = WindowList {
            support_level: WindowSupportLevel::SupportedEx,
            num_icon_caches: 3,
            num_icon_cache_entries: 12,
        };
        let capabilities = vec![CapabilitySet::WindowList(window_list.clone())];

        assert_eq!(server_window_list(&capabilities), Some(window_list));
        assert_eq!(
            negotiated_window_support_level(server_window_list(&capabilities).as_ref()),
            Some(WindowSupportLevel::SupportedEx)
        );
        assert_eq!(negotiated_window_support_level(None), None);
        assert_eq!(
            negotiated_window_support_level(Some(&WindowList {
                support_level: WindowSupportLevel::NotSupported,
                num_icon_caches: 0,
                num_icon_cache_entries: 0,
            })),
            None
        );
    }

    #[test]
    fn remote_app_capabilities_require_server_rail_and_window_list() {
        let rail_support_level = RailSupportLevel::SUPPORTED;
        let window_list = WindowList {
            support_level: WindowSupportLevel::SupportedEx,
            num_icon_caches: 3,
            num_icon_cache_entries: 12,
        };
        let rail = CapabilitySet::Rail(Rail {
            support_level: RailSupportLevel::SUPPORTED,
        });

        assert_eq!(
            remote_app_rail_capability(
                true,
                rail_support_level,
                core::slice::from_ref(&rail),
                Some(&window_list)
            )
            .unwrap(),
            Some(Rail {
                support_level: rail_support_level,
            })
        );
        assert!(remote_app_rail_capability(true, rail_support_level, &[], Some(&window_list)).is_err());
        assert!(remote_app_rail_capability(true, rail_support_level, core::slice::from_ref(&rail), None).is_err());
    }

    #[test]
    fn remote_app_capabilities_require_client_rail_support() {
        let window_list = WindowList {
            support_level: WindowSupportLevel::Supported,
            num_icon_caches: 0,
            num_icon_cache_entries: 0,
        };
        let rail = CapabilitySet::Rail(Rail {
            support_level: RailSupportLevel::SUPPORTED,
        });

        assert!(
            remote_app_rail_capability(
                true,
                RailSupportLevel::empty(),
                core::slice::from_ref(&rail),
                Some(&window_list)
            )
            .is_err()
        );
    }
}

// TESSERA PATCH — pins the core guard-2 behaviour: a virtual-channel PDU
// arriving during `CapabilitiesExchange` must be deferred, not fed to
// `decode_share_control`, which would kill a Deactivation-Reactivation
// Sequence outright. See `rdp/ironrdp-fork.md`.
#[cfg(test)]
mod tessera_guard2_tests {
    use ironrdp_pdu::rdp::capability_sets::MajorPlatformType;

    use super::*;
    use crate::Credentials;

    fn sample_config() -> Config {
        Config {
            desktop_size: DesktopSize {
                width: 1024,
                height: 768,
            },
            monitor_layout: None,
            desktop_scale_factor: 0,
            enable_tls: true,
            enable_credssp: false,
            enable_standard_rdp_security: false,
            credentials: Credentials::UsernamePassword {
                username: "test".into(),
                password: "test".into(),
            },
            domain: None,
            client_build: 0,
            client_name: "test".into(),
            keyboard_type: ironrdp_pdu::gcc::KeyboardType::IBM_ENHANCED,
            keyboard_subtype: 0,
            keyboard_layout: 0,
            keyboard_functional_keys_count: 12,
            connection_type: ironrdp_pdu::gcc::ConnectionType::Lan,
            ime_file_name: String::new(),
            bitmap: None,
            dig_product_id: String::new(),
            client_dir: String::new(),
            platform: MajorPlatformType::UNIX,
            hardware_id: None,
            request_data: None,
            autologon: false,
            enable_audio_playback: false,
            enable_audio_capture: false,
            enable_graphics_pipeline: false,
            license_cache: None,
            compression_type: None,
            enable_server_pointer: false,
            pointer_software_rendering: false,
            multitransport_flags: None,
            performance_flags: Default::default(),
            timezone_info: Default::default(),
            alternate_shell: String::new(),
            work_dir: String::new(),
            remote_application_mode: false,
            rail_support_level: RailSupportLevel::SUPPORTED,
        }
    }

    /// A PDU on a static virtual channel is not a Share Control PDU. Feeding it
    /// to `decode_share_control` kills the reactivation; deferring it does not.
    #[test]
    fn a_virtual_channel_pdu_during_capabilities_exchange_is_deferred() {
        const USER_CHANNEL_ID: u16 = 1002;
        const IO_CHANNEL_ID: u16 = 1003;
        const CLIPRDR_CHANNEL_ID: u16 = 1004;

        let mut sequence =
            ConnectionActivationSequence::new(sample_config(), IO_CHANNEL_ID, USER_CHANNEL_ID, Some(1005));

        let indication = ironrdp_pdu::mcs::McsMessage::SendDataIndication(ironrdp_pdu::mcs::SendDataIndication {
            initiator_id: USER_CHANNEL_ID,
            channel_id: CLIPRDR_CHANNEL_ID,
            user_data: std::borrow::Cow::Borrowed(&[0u8; 16]),
        });
        let input = ironrdp_core::encode_vec(&ironrdp_pdu::x224::X224(indication)).unwrap();

        let mut output = ironrdp_core::WriteBuf::new();
        let written = sequence.step(&input, None, &mut output).unwrap();

        assert!(matches!(written, Written::Nothing), "a deferred PDU writes nothing");
        assert!(
            matches!(sequence.state, ConnectionActivationState::CapabilitiesExchange { .. }),
            "the sequence stays in CapabilitiesExchange rather than advancing or erroring"
        );
    }
}
