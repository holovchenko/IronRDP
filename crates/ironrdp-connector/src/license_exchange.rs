use core::fmt::Debug;
use core::panic::RefUnwindSafe;
use core::{fmt, mem};
use std::str;
use std::sync::Arc;

use ironrdp_core::WriteBuf;
use ironrdp_pdu::PduHint;
use ironrdp_pdu::rdp::server_license::{self, LicenseInformation, LicensePdu, ServerLicenseError};
use rand::RngCore as _;
use tracing::{debug, error, info, trace, warn};

use super::{ConnectorError, ConnectorErrorExt as _, custom_err, general_err};
use crate::{
    ConnectorResult, ConnectorResultExt as _, MonotonicInstant, Sequence, State, Written, encode_send_data_request,
};

// TESSERA PATCH (vendored fork — see rdp/ironrdp-fork.md for the full story).
//
// A Windows RDS host here interleaves an Auto-Detect Request (an RTT measure)
// into the licensing exchange. Upstream feeds whatever arrives straight into
// `LicensePdu::decode`, which rejects it on `securityHeaderFlags` and kills the
// entire connection with a bare "decode error". mstsc talks to that same host
// without complaint.
//
// The observed payload was `[0, 16, 0, 0, 6, 0, 0, 0, 20, 0]`: flags `0x1000`
// (SEC_AUTODETECT_REQ, not SEC_LICENSE_PKT), headerLength 6, headerTypeId 0,
// sequenceNumber 0, requestType `0x0014` (RDP_RTT_REQUEST).
//
// So: check the security header before decoding, and skip anything that is not
// a licensing packet WITHOUT leaving the current state — the real licensing PDU
// is still to come, and treating the intruder as the answer would strand the
// exchange half-finished. Not answering the RTT probe is allowed; auto-detect
// is advisory and the server proceeds regardless.

/// `SEC_LICENSE_PKT` in `BasicSecurityHeader.flags` (MS-RDPBCGR 2.2.8.1.1.2.1).
const SEC_LICENSE_PKT: u16 = 0x0080;

/// Whether this MCS user data is a licensing packet at all, judged by the
/// first LE `u16` of its `BasicSecurityHeader`.
fn is_license_packet(user_data: &[u8]) -> bool {
    user_data.len() >= 2 && u16::from_le_bytes([user_data[0], user_data[1]]) & SEC_LICENSE_PKT != 0
}

/// Skips past this count in a single state without a matching licensing PDU,
/// `log_skipped_non_license_pdu` escalates from `debug!` to `warn!` — the
/// deadline in `tessera-rdp-worker` (`CONNECT_PHASE_TIMEOUT`) is what bounds
/// how long that can go on; this is purely a diagnostic trail so a real
/// misbehaving-server hang is visible in a release run, not silent.
const SKIP_WARN_THRESHOLD: u32 = 3;

fn log_skipped_non_license_pdu(ctx: &ironrdp_pdu::mcs::SendDataIndicationCtx<'_>, state: &str, skip_count: u32) {
    let flags = if ctx.user_data.len() >= 2 {
        u16::from_le_bytes([ctx.user_data[0], ctx.user_data[1]])
    } else {
        0
    };
    if skip_count > SKIP_WARN_THRESHOLD {
        warn!(
            state,
            skip_count,
            security_header_flags = format_args!("{flags:#06x}"),
            user_data_len = ctx.user_data.len(),
            user_data = ?ctx.user_data,
            "repeatedly skipping non-licensing PDUs received mid-licensing; still staying in this state"
        );
    } else {
        debug!(
            state,
            skip_count,
            security_header_flags = format_args!("{flags:#06x}"),
            user_data_len = ctx.user_data.len(),
            user_data = ?ctx.user_data,
            "skipping a non-licensing PDU received mid-licensing; staying in this state"
        );
    }
}

#[derive(Default, Debug)]
#[non_exhaustive]
pub enum LicenseExchangeState {
    #[default]
    Consumed,

    NewLicenseRequest {
        /// TESSERA PATCH — count of non-licensing PDUs skipped while waiting
        /// in this state; see `log_skipped_non_license_pdu`.
        skip_count: u32,
    },
    PlatformChallenge {
        encryption_data: server_license::LicenseEncryptionData,
        /// TESSERA PATCH — see `NewLicenseRequest::skip_count`.
        skip_count: u32,
    },
    UpgradeLicense {
        encryption_data: server_license::LicenseEncryptionData,
        /// TESSERA PATCH — see `NewLicenseRequest::skip_count`.
        skip_count: u32,
    },
    LicenseExchanged,
}

impl State for LicenseExchangeState {
    fn name(&self) -> &'static str {
        match self {
            Self::Consumed => "Consumed",
            Self::NewLicenseRequest { .. } => "NewLicenseRequest",
            Self::PlatformChallenge { .. } => "PlatformChallenge",
            Self::UpgradeLicense { .. } => "UpgradeLicense",
            Self::LicenseExchanged => "LicenseExchanged",
        }
    }

    fn is_terminal(&self) -> bool {
        matches!(self, Self::LicenseExchanged)
    }

    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

/// Client licensing sequence
///
/// Implements the state machine described in MS-RDPELE, section [3.1.5.3.1] Client State Transition.
///
/// [3.1.5.3.1]: https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-rdpele/8f9b860a-3687-401d-b3bc-7e9f5d4f7528
#[derive(Debug)]
pub struct LicenseExchangeSequence {
    pub state: LicenseExchangeState,
    pub io_channel_id: u16,
    pub username: String,
    pub domain: Option<String>,
    pub hardware_id: [u32; 4],
    pub license_cache: Arc<dyn LicenseCache>,
}

// Use RefUnwindSafe so that types that embed LicenseCache remain UnwindSafe
pub trait LicenseCache: Sync + Send + Debug + RefUnwindSafe {
    fn get_license(&self, license_info: LicenseInformation) -> ConnectorResult<Option<Vec<u8>>>;
    fn store_license(&self, license_info: LicenseInformation) -> ConnectorResult<()>;
}

#[derive(Debug)]
pub(crate) struct NoopLicenseCache;

impl LicenseCache for NoopLicenseCache {
    fn get_license(&self, _license_info: LicenseInformation) -> ConnectorResult<Option<Vec<u8>>> {
        Ok(None)
    }

    fn store_license(&self, _license_info: LicenseInformation) -> ConnectorResult<()> {
        Ok(())
    }
}

impl LicenseExchangeSequence {
    pub fn new(
        io_channel_id: u16,
        username: String,
        domain: Option<String>,
        hardware_id: [u32; 4],
        license_cache: Arc<dyn LicenseCache>,
    ) -> Self {
        Self {
            state: LicenseExchangeState::NewLicenseRequest { skip_count: 0 },
            io_channel_id,
            username,
            domain,
            hardware_id,
            license_cache,
        }
    }
}

impl Sequence for LicenseExchangeSequence {
    fn next_pdu_hint(&self) -> Option<&dyn PduHint> {
        match self.state {
            LicenseExchangeState::Consumed => None,
            LicenseExchangeState::NewLicenseRequest { .. } => Some(&ironrdp_pdu::X224_HINT),
            LicenseExchangeState::PlatformChallenge { .. } => Some(&ironrdp_pdu::X224_HINT),
            LicenseExchangeState::UpgradeLicense { .. } => Some(&ironrdp_pdu::X224_HINT),
            LicenseExchangeState::LicenseExchanged => None,
        }
    }

    fn state(&self) -> &dyn State {
        &self.state
    }

    fn step(
        &mut self,
        input: &[u8],
        _received_at: Option<MonotonicInstant>,
        output: &mut WriteBuf,
    ) -> ConnectorResult<Written> {
        let (written, next_state) = match mem::take(&mut self.state) {
            LicenseExchangeState::Consumed => {
                return Err(general_err!(
                    "license exchange sequence state is consumed (this is a bug)",
                ));
            }

            LicenseExchangeState::NewLicenseRequest { skip_count } => {
                let send_data_indication_ctx =
                    ironrdp_pdu::mcs::decode_send_data_indication(input).map_err(ConnectorError::decode)?;
                // TESSERA PATCH — see rdp/ironrdp-fork.md.
                if !is_license_packet(send_data_indication_ctx.user_data) {
                    let skip_count = skip_count + 1;
                    log_skipped_non_license_pdu(&send_data_indication_ctx, "NewLicenseRequest", skip_count);
                    self.state = LicenseExchangeState::NewLicenseRequest { skip_count };
                    return Ok(Written::Nothing);
                }

                let license_pdu = send_data_indication_ctx
                    .decode_user_data::<LicensePdu>()
                    .map_err(ConnectorError::decode)
                    .with_context("decode during LicenseExchangeState::NewLicenseRequest")?;

                match license_pdu {
                    LicensePdu::ServerLicenseRequest(license_request) => {
                        let mut rng = rand::rng();
                        let mut client_random = [0u8; server_license::RANDOM_NUMBER_SIZE];
                        rng.fill_bytes(&mut client_random);

                        let mut premaster_secret = [0u8; server_license::PREMASTER_SECRET_SIZE];
                        rng.fill_bytes(&mut premaster_secret);

                        let license_info = license_request
                            .scope_list
                            .iter()
                            .filter_map(|scope| {
                                self.license_cache
                                    .get_license(LicenseInformation {
                                        version: license_request.product_info.version,
                                        scope: scope.0.clone(),
                                        company_name: license_request.product_info.company_name.clone(),
                                        product_id: license_request.product_info.product_id.clone(),
                                        license_info: vec![],
                                    })
                                    .transpose()
                            })
                            .next()
                            .transpose()?;

                        if let Some(info) = license_info {
                            match server_license::ClientLicenseInfo::from_server_license_request(
                                &license_request,
                                &client_random,
                                &premaster_secret,
                                self.hardware_id,
                                info,
                            ) {
                                Ok((client_license_info, encryption_data)) => {
                                    trace!(?encryption_data, "Successfully generated Client License Info");
                                    trace!(message = ?client_license_info, "Send");

                                    let written = encode_send_data_request::<LicensePdu>(
                                        send_data_indication_ctx.initiator_id,
                                        send_data_indication_ctx.channel_id,
                                        &client_license_info.into(),
                                        output,
                                    )?;

                                    trace!(?written, "Written ClientLicenseInfo");

                                    (
                                        Written::from_size(written)?,
                                        LicenseExchangeState::PlatformChallenge {
                                            encryption_data,
                                            skip_count: 0,
                                        },
                                    )
                                }
                                Err(err) => {
                                    return Err(custom_err!("ClientNewLicenseRequest", err));
                                }
                            }
                        } else {
                            let hwid = self.hardware_id;
                            match server_license::ClientNewLicenseRequest::from_server_license_request(
                                &license_request,
                                &client_random,
                                &premaster_secret,
                                &self.username,
                                &format!("{:X}-{:X}-{:X}-{:X}", hwid[0], hwid[1], hwid[2], hwid[3]),
                            ) {
                                Ok((new_license_request, encryption_data)) => {
                                    trace!(?encryption_data, "Successfully generated Client New License Request");
                                    trace!(message = ?new_license_request, "Send");

                                    let written = encode_send_data_request::<LicensePdu>(
                                        send_data_indication_ctx.initiator_id,
                                        send_data_indication_ctx.channel_id,
                                        &new_license_request.into(),
                                        output,
                                    )?;

                                    (
                                        Written::from_size(written)?,
                                        LicenseExchangeState::PlatformChallenge {
                                            encryption_data,
                                            skip_count: 0,
                                        },
                                    )
                                }
                                Err(error) => {
                                    if let ServerLicenseError::InvalidX509Certificate {
                                        source: error,
                                        cert_der,
                                    } = &error
                                    {
                                        struct BytesHexFormatter<'a>(&'a [u8]);

                                        impl fmt::Display for BytesHexFormatter<'_> {
                                            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                                                write!(f, "0x")?;
                                                self.0.iter().try_for_each(|byte| write!(f, "{byte:02X}"))
                                            }
                                        }

                                        error!(
                                            %error,
                                            cert_der = %BytesHexFormatter(cert_der),
                                            "Unsupported or invalid X509 certificate received during license exchange step"
                                        );
                                    }

                                    return Err(custom_err!("ClientNewLicenseRequest", error));
                                }
                            }
                        }
                    }
                    LicensePdu::LicensingErrorMessage(error_message) => {
                        if error_message.error_code != server_license::LicenseErrorCode::StatusValidClient {
                            return Err(custom_err!(
                                "LicensingErrorMessage",
                                ServerLicenseError::from(error_message)
                            ));
                        }
                        info!("Server did not initiate license exchange");
                        (Written::Nothing, LicenseExchangeState::LicenseExchanged)
                    }
                    _ => {
                        return Err(general_err!(
                            "unexpected PDU received during LicenseExchangeState::NewLicenseRequest"
                        ));
                    }
                }
            }

            LicenseExchangeState::PlatformChallenge {
                encryption_data,
                skip_count,
            } => {
                let send_data_indication_ctx =
                    ironrdp_pdu::mcs::decode_send_data_indication(input).map_err(ConnectorError::decode)?;

                // TESSERA PATCH — see rdp/ironrdp-fork.md.
                if !is_license_packet(send_data_indication_ctx.user_data) {
                    let skip_count = skip_count + 1;
                    log_skipped_non_license_pdu(&send_data_indication_ctx, "PlatformChallenge", skip_count);
                    self.state = LicenseExchangeState::PlatformChallenge {
                        encryption_data,
                        skip_count,
                    };
                    return Ok(Written::Nothing);
                }

                let license_pdu = send_data_indication_ctx
                    .decode_user_data::<LicensePdu>()
                    .map_err(ConnectorError::decode)
                    .with_context("decode during LicenseExchangeState::PlatformChallenge")?;

                match license_pdu {
                    LicensePdu::ServerPlatformChallenge(challenge) => {
                        debug!(message = ?challenge, "Received");

                        let challenge_response =
                            server_license::ClientPlatformChallengeResponse::from_server_platform_challenge(
                                &challenge,
                                self.hardware_id,
                                &encryption_data,
                            )
                            .map_err(|e| custom_err!("ClientPlatformChallengeResponse", e))?;

                        debug!(message = ?challenge_response, "Send");

                        let written = encode_send_data_request::<LicensePdu>(
                            send_data_indication_ctx.initiator_id,
                            send_data_indication_ctx.channel_id,
                            &challenge_response.into(),
                            output,
                        )?;

                        (
                            Written::from_size(written)?,
                            LicenseExchangeState::UpgradeLicense {
                                encryption_data,
                                skip_count: 0,
                            },
                        )
                    }
                    LicensePdu::LicensingErrorMessage(error_message) => {
                        if error_message.error_code != server_license::LicenseErrorCode::StatusValidClient {
                            return Err(custom_err!(
                                "LicensingErrorMessage",
                                ServerLicenseError::from(error_message)
                            ));
                        }
                        debug!(message = ?error_message, "Received");
                        info!("Client licensing completed");
                        (Written::Nothing, LicenseExchangeState::LicenseExchanged)
                    }
                    _ => {
                        return Err(general_err!(
                            "unexpected PDU received during LicenseExchangeState::PlatformChallenge"
                        ));
                    }
                }
            }

            LicenseExchangeState::UpgradeLicense {
                encryption_data,
                skip_count,
            } => {
                let send_data_indication_ctx =
                    ironrdp_pdu::mcs::decode_send_data_indication(input).map_err(ConnectorError::decode)?;

                // TESSERA PATCH — see rdp/ironrdp-fork.md.
                if !is_license_packet(send_data_indication_ctx.user_data) {
                    let skip_count = skip_count + 1;
                    log_skipped_non_license_pdu(&send_data_indication_ctx, "UpgradeLicense", skip_count);
                    self.state = LicenseExchangeState::UpgradeLicense {
                        encryption_data,
                        skip_count,
                    };
                    return Ok(Written::Nothing);
                }

                let license_pdu = send_data_indication_ctx
                    .decode_user_data::<LicensePdu>()
                    .map_err(ConnectorError::decode)
                    .with_context("decode during SERVER_NEW_LICENSE/LicenseExchangeState::UpgradeLicense")?;

                match license_pdu {
                    LicensePdu::ServerUpgradeLicense(upgrade_license) => {
                        debug!(message = ?upgrade_license, "Received");

                        upgrade_license
                            .verify_server_license(&encryption_data)
                            .map_err(|e| custom_err!("license verification", e))?;

                        debug!("License verified with success");

                        let license_info = upgrade_license
                            .new_license_info(&encryption_data)
                            .map_err(ConnectorError::decode)?;

                        self.license_cache.store_license(license_info)?
                    }
                    LicensePdu::LicensingErrorMessage(error_message) => {
                        if error_message.error_code != server_license::LicenseErrorCode::StatusValidClient {
                            return Err(custom_err!(
                                "LicensingErrorMessage",
                                ServerLicenseError::from(error_message)
                            ));
                        }

                        debug!(message = ?error_message, "Received");
                        info!("Client licensing completed");
                    }
                    _ => {
                        return Err(general_err!(
                            "unexpected PDU received during LicenseExchangeState::UpgradeLicense"
                        ));
                    }
                }

                (Written::Nothing, LicenseExchangeState::LicenseExchanged)
            }

            LicenseExchangeState::LicenseExchanged => return Err(general_err!("license already exchanged")),
        };

        self.state = next_state;

        Ok(written)
    }
}

// TESSERA PATCH — tests for the skip-and-stay guard above (rdp/ironrdp-fork.md).
#[cfg(test)]
mod tests {
    use super::*;

    /// The exact bytes captured from the incident (rdp/ironrdp-fork.md):
    /// `BasicSecurityHeader.flags = 0x1000` (SEC_AUTODETECT_REQ), not a
    /// licensing packet at all.
    const CAPTURED_AUTODETECT_PROBE: [u8; 10] = [0, 16, 0, 0, 6, 0, 0, 0, 20, 0];

    #[test]
    fn captured_autodetect_probe_is_rejected() {
        assert!(!is_license_packet(&CAPTURED_AUTODETECT_PROBE));
    }

    #[test]
    fn sec_license_pkt_alone_is_accepted() {
        // flags = 0x0080, little-endian.
        let payload = [0x80, 0x00];
        assert!(is_license_packet(&payload));
    }

    #[test]
    fn sec_license_pkt_with_other_bits_is_accepted() {
        // flags = 0x1080 (SEC_AUTODETECT_REQ | SEC_LICENSE_PKT), little-endian.
        let payload = [0x80, 0x10];
        assert!(is_license_packet(&payload));
    }

    #[test]
    fn empty_payload_is_rejected_without_panicking() {
        assert!(!is_license_packet(&[]));
    }

    #[test]
    fn one_byte_payload_is_rejected_without_panicking() {
        assert!(!is_license_packet(&[0x80]));
    }

    /// Drives `LicenseExchangeSequence` through repeated auto-detect probes
    /// (the captured bytes, wrapped in a real Send Data Indication) and
    /// checks the skip counter climbs and the state never advances. This
    /// does not observe the `debug!`/`warn!` log-level escalation directly —
    /// no lightweight tracing harness is wired into this crate — but it
    /// covers the counting logic `log_skipped_non_license_pdu`'s threshold
    /// check depends on.
    #[test]
    fn skip_counter_climbs_and_state_stays_put_across_repeated_probes() {
        let mut sequence =
            LicenseExchangeSequence::new(1001, "tessera".to_owned(), None, [0; 4], Arc::new(NoopLicenseCache));

        // initiator_id is PER-encoded relative to MCS's BASE_CHANNEL_ID (1001);
        // anything lower underflows the encoder.
        let probe_pdu = encode_send_data_indication(1001, 1001, &CAPTURED_AUTODETECT_PROBE);
        let mut output = WriteBuf::new();

        for expected_skip_count in 1..=(SKIP_WARN_THRESHOLD + 2) {
            let written = sequence
                .step(&probe_pdu, None, &mut output)
                .expect("skip-and-stay must not error");
            assert!(matches!(written, Written::Nothing));

            assert!(
                matches!(&sequence.state, LicenseExchangeState::NewLicenseRequest { .. }),
                "expected to stay in NewLicenseRequest, got {:?}",
                sequence.state
            );
            if let LicenseExchangeState::NewLicenseRequest { skip_count } = &sequence.state {
                assert_eq!(*skip_count, expected_skip_count);
            }
        }
    }

    /// Builds a Send Data Indication PDU carrying `user_data` verbatim,
    /// matching the wire shape `ironrdp_pdu::mcs::decode_send_data_indication`
    /// expects — i.e. what a server actually sends, as opposed to
    /// `encode_send_data_request` (client -> server) used by production code
    /// above.
    fn encode_send_data_indication(initiator_id: u16, channel_id: u16, user_data: &[u8]) -> Vec<u8> {
        let pdu = ironrdp_pdu::mcs::SendDataIndication {
            initiator_id,
            channel_id,
            user_data: std::borrow::Cow::Borrowed(user_data),
        };
        let mut buf = WriteBuf::new();
        ironrdp_core::encode_buf(&ironrdp_pdu::x224::X224(pdu), &mut buf)
            .expect("encoding a Send Data Indication for the test fixture must not fail");
        buf.into_inner()
    }
}
