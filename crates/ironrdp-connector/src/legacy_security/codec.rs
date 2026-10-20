use core::fmt;
use std::sync::{Arc, Mutex, PoisonError};

use ironrdp_pdu::rdp::headers::BasicSecurityHeaderFlags;

use super::{Cipher, SessionKeys, mac, salted_mac};
use crate::{ConnectorError, ConnectorErrorExt as _, ConnectorResult};

const TPKT_VERSION: u8 = 3;
const X224_DATA_HEADER: [u8; 3] = [0x02, 0xF0, 0x80];
const MCS_SEND_DATA_REQUEST: u8 = 25;
const MCS_SEND_DATA_INDICATION: u8 = 26;
/// TPKT header (4), X.224 data header (3), MCS choice (1), initiator (2), channel (2), priority (1).
const SLOW_PATH_PREFIX_SIZE: usize = 13;
const SECURITY_HEADER_SIZE: usize = 4;
const SIGNATURE_SIZE: usize = 8;
const FAST_PATH_SECURE_CHECKSUM: u8 = 0x40;
const FAST_PATH_ENCRYPTED: u8 = 0x80;
const MAX_PER_LENGTH: usize = 0x7FFF;

const CRYPTO_FLAGS: BasicSecurityHeaderFlags = BasicSecurityHeaderFlags::ENCRYPT
    .union(BasicSecurityHeaderFlags::SECURE_CHECKSUM)
    .union(BasicSecurityHeaderFlags::RESET_SEQNO)
    .union(BasicSecurityHeaderFlags::IGNORE_SEQNO);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    SecurityExchange,
    ClientInfo,
    Licensing,
    Active,
}

impl fmt::Display for Phase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::SecurityExchange => "SecurityExchange",
            Self::ClientInfo => "ClientInfo",
            Self::Licensing => "Licensing",
            Self::Active => "Active",
        };
        f.write_str(name)
    }
}

/// Encrypts and decrypts whole TPKT and fast-path frames with Standard RDP Security (RC4),
/// MS-RDPBCGR 5.3, so that the layers above see the stream they would see without encryption.
pub struct LegacySecurityCodec {
    mac_key: Vec<u8>,
    encrypt: Cipher,
    decrypt: Cipher,
    // Total packets decrypted: `Cipher::use_count` resets at each key update, the salted MAC does not.
    decrypt_count: u32,
    io_channel_id: u16,
    message_channel_id: Option<u16>,
    phase: Phase,
    license_encrypt: bool,
}

impl fmt::Debug for LegacySecurityCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LegacySecurityCodec")
            .field("phase", &self.phase)
            .finish_non_exhaustive()
    }
}

impl LegacySecurityCodec {
    pub fn new(keys: SessionKeys, io_channel_id: u16, message_channel_id: Option<u16>) -> Self {
        Self {
            mac_key: keys.mac_key,
            encrypt: keys.encrypt,
            decrypt: keys.decrypt,
            decrypt_count: 0,
            io_channel_id,
            message_channel_id,
            phase: Phase::SecurityExchange,
            license_encrypt: false,
        }
    }

    /// Turns one complete frame received from the server into the frame the connection layers expect.
    pub fn decrypt_inbound(&mut self, frame: &[u8]) -> ConnectorResult<Vec<u8>> {
        match frame_kind(frame)? {
            FrameKind::FastPath => self.decrypt_fast_path(frame),
            FrameKind::Tpkt => self.decrypt_slow_path(frame),
        }
    }

    /// Turns one complete frame produced by the connection layers into the frame to send to the server.
    pub fn encrypt_outbound(&mut self, frame: &[u8]) -> ConnectorResult<Vec<u8>> {
        match frame_kind(frame)? {
            FrameKind::FastPath => self.encrypt_fast_path(frame),
            FrameKind::Tpkt => self.encrypt_slow_path(frame),
        }
    }

    fn decrypt_slow_path(&mut self, frame: &[u8]) -> ConnectorResult<Vec<u8>> {
        let Some(slow) = SlowPath::parse(frame, MCS_SEND_DATA_INDICATION)? else {
            return Ok(frame.to_vec());
        };

        let (flags, body) = split_security_header(slow.user_data)?;

        let plaintext = if flags.contains(BasicSecurityHeaderFlags::ENCRYPT) {
            let (signature, ciphertext) = body
                .split_at_checked(SIGNATURE_SIZE)
                .ok_or_else(|| ConnectorError::reason("legacy security", "encrypted PDU is too short"))?;
            self.open(
                signature,
                ciphertext,
                flags.contains(BasicSecurityHeaderFlags::SECURE_CHECKSUM),
            )?
        } else {
            body.to_vec()
        };

        if slow.channel_id == self.io_channel_id
            && self.phase == Phase::Licensing
            && !flags.contains(BasicSecurityHeaderFlags::LICENSE_PKT)
        {
            self.phase = Phase::Active;
        }

        if flags.contains(BasicSecurityHeaderFlags::LICENSE_PKT | BasicSecurityHeaderFlags::LICENSE_ENCRYPT_CS) {
            self.license_encrypt = true;
        }

        let remaining = flags.difference(CRYPTO_FLAGS);
        let mut user_data = Vec::with_capacity(SECURITY_HEADER_SIZE + plaintext.len());
        if !remaining.is_empty() {
            user_data.extend_from_slice(&security_header(remaining));
        }
        user_data.extend_from_slice(&plaintext);

        slow.rebuild(&user_data)
    }

    fn encrypt_slow_path(&mut self, frame: &[u8]) -> ConnectorResult<Vec<u8>> {
        let Some(slow) = SlowPath::parse(frame, MCS_SEND_DATA_REQUEST)? else {
            return Ok(frame.to_vec());
        };

        let peeked = peek_security_header(slow.user_data);
        let (has_header, protected, next_phase) = if Some(slow.channel_id) == self.message_channel_id {
            (true, true, self.phase)
        } else if slow.channel_id == self.io_channel_id {
            match self.phase {
                Phase::SecurityExchange => {
                    self.require(peeked, BasicSecurityHeaderFlags::EXCHANGE_PKT)?;
                    self.phase = Phase::ClientInfo;
                    return Ok(frame.to_vec());
                }
                Phase::ClientInfo => {
                    self.require(peeked, BasicSecurityHeaderFlags::INFO_PKT)?;
                    (true, true, Phase::Licensing)
                }
                Phase::Licensing => {
                    self.require(peeked, BasicSecurityHeaderFlags::LICENSE_PKT)?;
                    (true, self.license_encrypt, Phase::Licensing)
                }
                Phase::Active => (false, true, Phase::Active),
            }
        } else {
            (false, true, self.phase)
        };

        self.phase = next_phase;

        if !protected {
            return Ok(frame.to_vec());
        }

        let (flags, data) = if has_header {
            let peeked = peeked.ok_or_else(|| self.unexpected_outbound())?;
            (peeked, slow.user_data.get(SECURITY_HEADER_SIZE..).unwrap_or_default())
        } else {
            (BasicSecurityHeaderFlags::empty(), slow.user_data)
        };

        let mut user_data = Vec::with_capacity(SECURITY_HEADER_SIZE + SIGNATURE_SIZE + data.len());
        user_data.extend_from_slice(&security_header(flags | BasicSecurityHeaderFlags::ENCRYPT));
        self.seal(data, &mut user_data);

        slow.rebuild(&user_data)
    }

    fn decrypt_fast_path(&mut self, frame: &[u8]) -> ConnectorResult<Vec<u8>> {
        let fast = FastPath::parse(frame)?;

        if fast.header & FAST_PATH_ENCRYPTED == 0 {
            return Ok(frame.to_vec());
        }

        let (signature, ciphertext) = fast
            .payload
            .split_at_checked(SIGNATURE_SIZE)
            .ok_or_else(|| ConnectorError::reason("legacy security", "encrypted fast-path PDU is too short"))?;
        let plaintext = self.open(signature, ciphertext, fast.header & FAST_PATH_SECURE_CHECKSUM != 0)?;

        Ok(build_fast_path(
            fast.header & !(FAST_PATH_ENCRYPTED | FAST_PATH_SECURE_CHECKSUM),
            &plaintext,
        ))
    }

    fn encrypt_fast_path(&mut self, frame: &[u8]) -> ConnectorResult<Vec<u8>> {
        let fast = FastPath::parse(frame)?;

        let mut payload = Vec::with_capacity(SIGNATURE_SIZE + fast.payload.len());
        self.seal(fast.payload, &mut payload);

        Ok(build_fast_path(fast.header | FAST_PATH_ENCRYPTED, &payload))
    }

    /// Decrypts `ciphertext` and checks it against `signature`.
    fn open(&mut self, signature: &[u8], ciphertext: &[u8], salted: bool) -> ConnectorResult<Vec<u8>> {
        let mut plaintext = ciphertext.to_vec();
        self.decrypt.process(&mut plaintext);

        let expected = if salted {
            salted_mac(&self.mac_key, &plaintext, self.decrypt_count)
        } else {
            mac(&self.mac_key, &plaintext)
        };
        self.decrypt_count = self.decrypt_count.wrapping_add(1);

        if expected.as_slice() != signature {
            return Err(ConnectorError::reason(
                "legacy security",
                "legacy security MAC mismatch",
            ));
        }

        Ok(plaintext)
    }

    /// Appends the signature of `data` followed by its ciphertext to `out`.
    fn seal(&mut self, data: &[u8], out: &mut Vec<u8>) {
        out.extend_from_slice(&mac(&self.mac_key, data));
        let start = out.len();
        out.extend_from_slice(data);
        if let Some(body) = out.get_mut(start..) {
            self.encrypt.process(body);
        }
    }

    fn require(
        &self,
        header: Option<BasicSecurityHeaderFlags>,
        expected: BasicSecurityHeaderFlags,
    ) -> ConnectorResult<()> {
        match header {
            Some(flags) if flags.contains(expected) => Ok(()),
            _ => Err(self.unexpected_outbound()),
        }
    }

    fn unexpected_outbound(&self) -> ConnectorError {
        ConnectorError::reason(
            "legacy security",
            format!("legacy security: unexpected outbound PDU in phase {}", self.phase),
        )
    }
}

enum FrameKind {
    Tpkt,
    FastPath,
}

fn frame_kind(frame: &[u8]) -> ConnectorResult<FrameKind> {
    match frame.first() {
        Some(&TPKT_VERSION) => Ok(FrameKind::Tpkt),
        Some(first) if first & 0x03 == 0 => Ok(FrameKind::FastPath),
        _ => Err(ConnectorError::reason("legacy security", "unrecognized frame")),
    }
}

fn security_header(flags: BasicSecurityHeaderFlags) -> [u8; SECURITY_HEADER_SIZE] {
    let [lo, hi] = flags.bits().to_le_bytes();
    [lo, hi, 0, 0]
}

fn peek_security_header(user_data: &[u8]) -> Option<BasicSecurityHeaderFlags> {
    let bytes = user_data.get(..2)?;
    Some(BasicSecurityHeaderFlags::from_bits_retain(u16::from_le_bytes([
        *bytes.first()?,
        *bytes.get(1)?,
    ])))
}

fn split_security_header(user_data: &[u8]) -> ConnectorResult<(BasicSecurityHeaderFlags, &[u8])> {
    peek_security_header(user_data)
        .zip(user_data.get(SECURITY_HEADER_SIZE..))
        .ok_or_else(|| ConnectorError::reason("legacy security", "missing basic security header"))
}

/// A TPKT-wrapped MCS Send Data Request or Indication, split around its user data.
struct SlowPath<'a> {
    prefix: &'a [u8],
    channel_id: u16,
    user_data: &'a [u8],
}

impl<'a> SlowPath<'a> {
    /// Returns `None` for frames that are not an MCS PDU of the `expected` type.
    fn parse(frame: &'a [u8], expected: u8) -> ConnectorResult<Option<Self>> {
        let malformed = || ConnectorError::reason("legacy security", "malformed slow-path frame");

        let Some(prefix) = frame.get(..SLOW_PATH_PREFIX_SIZE) else {
            return Ok(None);
        };
        if prefix.get(4..7) != Some(&X224_DATA_HEADER) || prefix.get(7).map(|b| b >> 2) != Some(expected) {
            return Ok(None);
        }

        let channel_id = u16::from_be_bytes([
            *prefix.get(10).ok_or_else(malformed)?,
            *prefix.get(11).ok_or_else(malformed)?,
        ]);

        let length_bytes = frame.get(SLOW_PATH_PREFIX_SIZE..).ok_or_else(malformed)?;
        let (length, length_size) = match length_bytes {
            [a, b, ..] if a & 0x80 != 0 => ((usize::from(a & 0x7F) << 8) | usize::from(*b), 2),
            [a, ..] => (usize::from(*a), 1),
            [] => return Err(malformed()),
        };
        let user_data = length_bytes
            .get(length_size..)
            .filter(|rest| rest.len() == length)
            .ok_or_else(malformed)?;

        Ok(Some(Self {
            prefix,
            channel_id,
            user_data,
        }))
    }

    fn rebuild(&self, user_data: &[u8]) -> ConnectorResult<Vec<u8>> {
        if user_data.len() > MAX_PER_LENGTH {
            return Err(ConnectorError::reason("legacy security", "PDU is too large"));
        }
        let length = u16::try_from(user_data.len())
            .map_err(|_| ConnectorError::reason("legacy security", "PDU is too large"))?;

        let mut out = Vec::with_capacity(SLOW_PATH_PREFIX_SIZE + 2 + user_data.len());
        out.extend_from_slice(self.prefix);
        if length > 0x7F {
            out.extend_from_slice(&(length | 0x8000).to_be_bytes());
        } else {
            out.extend_from_slice(&length.to_be_bytes()[1..]);
        }
        out.extend_from_slice(user_data);

        let total =
            u16::try_from(out.len()).map_err(|_| ConnectorError::reason("legacy security", "PDU is too large"))?;
        if let Some(tpkt_length) = out.get_mut(2..4) {
            tpkt_length.copy_from_slice(&total.to_be_bytes());
        }

        Ok(out)
    }
}

struct FastPath<'a> {
    header: u8,
    payload: &'a [u8],
}

impl<'a> FastPath<'a> {
    fn parse(frame: &'a [u8]) -> ConnectorResult<Self> {
        let malformed = || ConnectorError::reason("legacy security", "malformed fast-path frame");

        let (&header, rest) = frame.split_first().ok_or_else(malformed)?;
        let (length, length_size) = match rest {
            [a, b, ..] if a & 0x80 != 0 => ((usize::from(a & 0x7F) << 8) | usize::from(*b), 2),
            [a, ..] => (usize::from(*a), 1),
            [] => return Err(malformed()),
        };
        if length != frame.len() {
            return Err(malformed());
        }

        Ok(Self {
            header,
            payload: rest.get(length_size..).ok_or_else(malformed)?,
        })
    }
}

fn build_fast_path(header: u8, payload: &[u8]) -> Vec<u8> {
    // The length field counts the header byte and itself; it takes the short form whenever that fits in 7 bits.
    let short_total = 2 + payload.len();
    let mut out = Vec::with_capacity(3 + payload.len());
    out.push(header);
    if short_total <= 0x7F {
        out.extend_from_slice(&u16::try_from(short_total).unwrap_or_default().to_be_bytes()[1..]);
    } else {
        out.extend_from_slice(&(u16::try_from(short_total + 1).unwrap_or(0x7FFF) | 0x8000).to_be_bytes());
    }
    out.extend_from_slice(payload);
    out
}

/// Frame codec shared between the connector and the transport stream; passes frames through until armed.
#[derive(Clone, Default)]
pub struct LegacySecurityHandle {
    inner: Arc<Mutex<Option<LegacySecurityCodec>>>,
}

impl fmt::Debug for LegacySecurityHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LegacySecurityHandle")
            .field("armed", &self.is_armed())
            .finish()
    }
}

impl LegacySecurityHandle {
    pub fn arm(&self, codec: LegacySecurityCodec) {
        *self.inner.lock().unwrap_or_else(PoisonError::into_inner) = Some(codec);
    }

    pub fn is_armed(&self) -> bool {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner).is_some()
    }

    pub fn decrypt_inbound(&self, frame: &[u8]) -> ConnectorResult<Vec<u8>> {
        match self.inner.lock().unwrap_or_else(PoisonError::into_inner).as_mut() {
            Some(codec) => codec.decrypt_inbound(frame),
            None => Ok(frame.to_vec()),
        }
    }

    pub fn encrypt_outbound(&self, frame: &[u8]) -> ConnectorResult<Vec<u8>> {
        match self.inner.lock().unwrap_or_else(PoisonError::into_inner).as_mut() {
            Some(codec) => codec.encrypt_outbound(frame),
            None => Ok(frame.to_vec()),
        }
    }
}

#[cfg(test)]
mod tests {
    use ironrdp_core::encode_vec;
    use ironrdp_pdu::gcc::EncryptionMethod;
    use ironrdp_pdu::mcs::{SendDataIndication, SendDataRequest};
    use ironrdp_pdu::x224::X224;

    use super::*;

    const SERVER_RANDOM: &str = "1011772030610a12e434a11ef2c39f317da45f01893496e0ff1108697f1ac3d2";
    const CLIENT_RANDOM: &str = "0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20";
    const PLAINTEXT: &str = "09040904b3430000a0a1a2a3a4a5a6a7a8a9aaabacadaeafb0b1b2b3b4b5b6b7";
    const MAC_128: &str = "c9cdd9bd528b5039";
    const CIPHERTEXT_128: &str = "4746570038d531ae39d4e63c9fc6b6cfcf55293b658dbef321d2a13d950a7691";

    const IO: u16 = 1003;
    const MSG: u16 = 1004;
    const STATIC_CHANNEL: u16 = 1005;

    fn unhex(s: &str) -> Vec<u8> {
        s.as_bytes()
            .chunks(2)
            .map(|pair| u8::from_str_radix(core::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect()
    }

    fn random(s: &str) -> [u8; 32] {
        unhex(s).try_into().unwrap()
    }

    fn keys() -> SessionKeys {
        SessionKeys::derive(
            &random(CLIENT_RANDOM),
            &random(SERVER_RANDOM),
            EncryptionMethod::BIT_128,
        )
        .unwrap()
    }

    /// Server-side view of the same keys: encrypt and decrypt swapped.
    fn server_keys() -> SessionKeys {
        let SessionKeys {
            mac_key,
            encrypt,
            decrypt,
        } = keys();
        SessionKeys {
            mac_key,
            encrypt: decrypt,
            decrypt: encrypt,
        }
    }

    fn client() -> LegacySecurityCodec {
        LegacySecurityCodec::new(keys(), IO, Some(MSG))
    }

    fn header(flags: BasicSecurityHeaderFlags) -> Vec<u8> {
        security_header(flags).to_vec()
    }

    fn request(channel_id: u16, user_data: &[u8]) -> Vec<u8> {
        encode_vec(&X224(SendDataRequest {
            initiator_id: 1002,
            channel_id,
            user_data: user_data.into(),
        }))
        .unwrap()
    }

    fn indication(channel_id: u16, user_data: &[u8]) -> Vec<u8> {
        encode_vec(&X224(SendDataIndication {
            initiator_id: 1002,
            channel_id,
            user_data: user_data.into(),
        }))
        .unwrap()
    }

    fn user_data_of(frame: &[u8]) -> Vec<u8> {
        SlowPath::parse(frame, MCS_SEND_DATA_REQUEST)
            .unwrap()
            .or_else(|| SlowPath::parse(frame, MCS_SEND_DATA_INDICATION).unwrap())
            .unwrap()
            .user_data
            .to_vec()
    }

    /// What a server does to `data`: MAC (optionally salted with `count`) then RC4.
    fn server_seal(keys: &mut SessionKeys, data: &[u8], salted_count: Option<u32>) -> Vec<u8> {
        let signature = match salted_count {
            Some(count) => salted_mac(&keys.mac_key, data, count),
            None => mac(&keys.mac_key, data),
        };
        let mut body = data.to_vec();
        keys.encrypt.process(&mut body);
        [signature.as_slice(), &body].concat()
    }

    /// What a server does to a client frame body: decrypt and verify.
    fn server_open(keys: &mut SessionKeys, sealed: &[u8]) -> Vec<u8> {
        let (signature, ciphertext) = sealed.split_at(SIGNATURE_SIZE);
        let mut plaintext = ciphertext.to_vec();
        keys.decrypt.process(&mut plaintext);
        assert_eq!(mac(&keys.mac_key, &plaintext), signature);
        plaintext
    }

    fn encrypted_indication(
        keys: &mut SessionKeys,
        channel_id: u16,
        flags: BasicSecurityHeaderFlags,
        data: &[u8],
    ) -> Vec<u8> {
        let user_data = [
            header(flags | BasicSecurityHeaderFlags::ENCRYPT),
            server_seal(keys, data, None),
        ]
        .concat();
        indication(channel_id, &user_data)
    }

    fn advance_to_active(codec: &mut LegacySecurityCodec) {
        codec
            .encrypt_outbound(&request(
                IO,
                &[header(BasicSecurityHeaderFlags::EXCHANGE_PKT), vec![1, 2, 3]].concat(),
            ))
            .unwrap();
        codec
            .encrypt_outbound(&request(
                IO,
                &[header(BasicSecurityHeaderFlags::INFO_PKT), vec![1]].concat(),
            ))
            .unwrap();
        codec
            .decrypt_inbound(&indication(
                IO,
                &[header(BasicSecurityHeaderFlags::LICENSE_PKT), vec![7]].concat(),
            ))
            .unwrap();
        codec.decrypt_inbound(&indication(IO, &[0, 0, 0, 0, 9])).unwrap();
        assert_eq!(codec.phase, Phase::Active);
    }

    #[test]
    fn client_info_encrypts_to_reference_bytes() {
        let mut codec = client();
        let exchange = request(
            IO,
            &[header(BasicSecurityHeaderFlags::EXCHANGE_PKT), vec![0xAA; 4]].concat(),
        );
        assert_eq!(codec.encrypt_outbound(&exchange).unwrap(), exchange);

        let info = [header(BasicSecurityHeaderFlags::INFO_PKT), unhex(PLAINTEXT)].concat();
        let out = codec.encrypt_outbound(&request(IO, &info)).unwrap();

        let expected = [
            header(BasicSecurityHeaderFlags::INFO_PKT | BasicSecurityHeaderFlags::ENCRYPT),
            unhex(MAC_128),
            unhex(CIPHERTEXT_128),
        ]
        .concat();
        assert_eq!(user_data_of(&out), expected);
        assert_eq!(usize::from(u16::from_be_bytes([out[2], out[3]])), out.len());
    }

    #[test]
    fn security_exchange_passes_through_and_advances_phase() {
        let mut codec = client();
        let exchange = request(
            IO,
            &[header(BasicSecurityHeaderFlags::EXCHANGE_PKT), vec![5; 70]].concat(),
        );

        assert_eq!(codec.encrypt_outbound(&exchange).unwrap(), exchange);
        assert_eq!(codec.phase, Phase::ClientInfo);
    }

    #[test]
    fn inbound_slow_path_strips_header_when_only_crypto_flags() {
        let mut codec = client();
        let mut server = server_keys();
        let payload = vec![0x42; 200];
        let frame = encrypted_indication(&mut server, IO, BasicSecurityHeaderFlags::empty(), &payload);

        let out = codec.decrypt_inbound(&frame).unwrap();

        assert_eq!(out, indication(IO, &payload));
    }

    #[test]
    fn inbound_license_pdu_keeps_license_header() {
        let mut codec = client();
        let mut server = server_keys();
        let frame = encrypted_indication(&mut server, IO, BasicSecurityHeaderFlags::LICENSE_PKT, &[1, 2, 3]);

        let out = codec.decrypt_inbound(&frame).unwrap();

        assert_eq!(
            out,
            indication(
                IO,
                &[header(BasicSecurityHeaderFlags::LICENSE_PKT), vec![1, 2, 3]].concat()
            )
        );
    }

    #[test]
    fn inbound_autodetect_on_message_channel_keeps_header() {
        let mut codec = client();
        let mut server = server_keys();
        let autodetect = BasicSecurityHeaderFlags::from_bits_retain(0x1000);
        let frame = encrypted_indication(&mut server, MSG, autodetect, &[9, 9]);

        let out = codec.decrypt_inbound(&frame).unwrap();

        assert_eq!(out, indication(MSG, &[header(autodetect), vec![9, 9]].concat()));
    }

    #[test]
    fn outbound_on_message_channel_is_headed_in_active_phase() {
        let mut codec = client();
        advance_to_active(&mut codec);
        let mut server = server_keys();
        // The licensing walk above used the encrypt cipher once (Client Info), mirror that.
        server_open_info(&mut server);
        let autodetect = BasicSecurityHeaderFlags::from_bits_retain(0x2000);

        let out = codec
            .encrypt_outbound(&request(MSG, &[header(autodetect), vec![4, 5, 6]].concat()))
            .unwrap();

        let user_data = user_data_of(&out);
        assert_eq!(user_data[..4], header(autodetect | BasicSecurityHeaderFlags::ENCRYPT));
        assert_eq!(server_open(&mut server, &user_data[4..]), vec![4, 5, 6]);
    }

    /// Consumes the Client Info packet sent by `advance_to_active` from the server's decrypt stream.
    fn server_open_info(server: &mut SessionKeys) {
        let mut probe = client();
        probe
            .encrypt_outbound(&request(
                IO,
                &[header(BasicSecurityHeaderFlags::EXCHANGE_PKT), vec![1, 2, 3]].concat(),
            ))
            .unwrap();
        let sent = probe
            .encrypt_outbound(&request(
                IO,
                &[header(BasicSecurityHeaderFlags::INFO_PKT), vec![1]].concat(),
            ))
            .unwrap();
        assert_eq!(server_open(server, &user_data_of(&sent)[4..]), vec![1]);
    }

    #[test]
    fn license_reply_encrypted_only_with_license_encrypt_cs() {
        let reply = request(
            IO,
            &[header(BasicSecurityHeaderFlags::LICENSE_PKT), vec![8; 10]].concat(),
        );

        let mut without = client();
        without.phase = Phase::Licensing;
        let mut server = server_keys();
        let challenge = encrypted_indication(&mut server, IO, BasicSecurityHeaderFlags::LICENSE_PKT, &[1]);
        without.decrypt_inbound(&challenge).unwrap();
        assert_eq!(without.encrypt_outbound(&reply).unwrap(), reply);

        let mut with = client();
        with.phase = Phase::Licensing;
        let mut server = server_keys();
        let challenge = encrypted_indication(
            &mut server,
            IO,
            BasicSecurityHeaderFlags::LICENSE_PKT | BasicSecurityHeaderFlags::LICENSE_ENCRYPT_CS,
            &[1],
        );
        with.decrypt_inbound(&challenge).unwrap();
        let out = with.encrypt_outbound(&reply).unwrap();
        let user_data = user_data_of(&out);
        assert_eq!(
            user_data[..4],
            header(BasicSecurityHeaderFlags::LICENSE_PKT | BasicSecurityHeaderFlags::ENCRYPT)
        );
        assert_eq!(server_open(&mut server, &user_data[4..]), vec![8; 10]);
    }

    #[test]
    fn fast_path_length_shrinks_to_one_byte() {
        for (payload_len, expected_len_bytes) in [(0x7Ausize, 1usize), (0x7D, 1), (0x7E, 2), (0x7F, 2)] {
            let mut codec = client();
            let mut server = server_keys();
            let payload = vec![0x33; payload_len];
            let sealed = server_seal(&mut server, &payload, None);
            let mut frame = vec![FAST_PATH_ENCRYPTED, 0x80];
            // 2-byte length form regardless of size, as a server may send it
            let total = 3 + sealed.len();
            frame[1] = 0x80 | u8::try_from(total >> 8).unwrap();
            frame.push(u8::try_from(total & 0xFF).unwrap());
            frame.extend_from_slice(&sealed);

            let out = codec.decrypt_inbound(&frame).unwrap();

            let total_out = 2 + payload_len;
            if expected_len_bytes == 1 {
                assert_eq!(out[0], 0);
                assert_eq!(usize::from(out[1]), total_out);
                assert_eq!(out[2..], payload[..]);
            } else {
                assert_eq!(out[1], 0x80 | u8::try_from((total_out + 1) >> 8).unwrap());
                assert_eq!(usize::from(out[2]), (total_out + 1) & 0xFF);
                assert_eq!(out[3..], payload[..]);
            }
            assert_eq!(FastPath::parse(&out).unwrap().payload, payload);
        }
    }

    #[test]
    fn fast_path_inbound_salted_checksum() {
        let mut codec = client();
        let mut server = server_keys();
        for count in 0..3 {
            let payload = vec![u8::try_from(count).unwrap(); 20];
            let sealed = server_seal(&mut server, &payload, Some(count));
            let frame = [
                vec![
                    FAST_PATH_ENCRYPTED | FAST_PATH_SECURE_CHECKSUM,
                    u8::try_from(2 + sealed.len()).unwrap(),
                ],
                sealed,
            ]
            .concat();

            assert_eq!(
                codec.decrypt_inbound(&frame).unwrap(),
                [vec![0, u8::try_from(2 + payload.len()).unwrap()], payload].concat()
            );
        }
    }

    #[test]
    fn fast_path_input_round_trip() {
        let mut codec = client();
        let mut server = server_keys();
        for payload_len in [5usize, 0x75, 0x90] {
            let payload = vec![0x11; payload_len];
            let built = build_fast_path(0x04, &payload);

            let out = codec.encrypt_outbound(&built).unwrap();

            let parsed = FastPath::parse(&out).unwrap();
            assert_eq!(parsed.header, 0x04 | FAST_PATH_ENCRYPTED);
            assert_eq!(server_open(&mut server, parsed.payload), payload);
        }
    }

    #[test]
    fn mac_mismatch_is_an_error() {
        let mut codec = client();
        let mut server = server_keys();
        let mut frame = encrypted_indication(&mut server, IO, BasicSecurityHeaderFlags::empty(), &[1, 2, 3, 4]);
        let last = frame.len() - 1;
        frame[last] ^= 0x01;

        let err = codec.decrypt_inbound(&frame).unwrap_err();

        assert!(format!("{err:?}").contains("legacy security MAC mismatch"), "{err:?}");
    }

    #[test]
    fn unexpected_outbound_in_client_info_phase_is_an_error() {
        let mut codec = client();
        codec.phase = Phase::ClientInfo;

        let err = codec.encrypt_outbound(&request(IO, &[0, 0, 0, 0, 1])).unwrap_err();

        assert!(
            format!("{err:?}").contains("legacy security: unexpected outbound PDU in phase ClientInfo"),
            "{err:?}"
        );
    }

    #[test]
    fn active_phase_outbound_prepends_header_and_static_channels_too() {
        let mut codec = client();
        advance_to_active(&mut codec);
        let mut server = server_keys();
        server_open_info(&mut server);

        for channel in [IO, STATIC_CHANNEL] {
            let out = codec.encrypt_outbound(&request(channel, &[1, 2, 3])).unwrap();
            let user_data = user_data_of(&out);
            assert_eq!(user_data[..4], header(BasicSecurityHeaderFlags::ENCRYPT));
            assert_eq!(server_open(&mut server, &user_data[4..]), vec![1, 2, 3]);
        }
    }

    #[test]
    fn handle_passes_through_until_armed() {
        let handle = LegacySecurityHandle::default();
        let frame = request(IO, &[0, 0, 0, 0, 1]);
        assert!(!handle.is_armed());
        assert_eq!(handle.encrypt_outbound(&frame).unwrap(), frame);
        assert_eq!(handle.decrypt_inbound(&frame).unwrap(), frame);

        handle.arm(client());

        assert!(handle.is_armed());
        assert!(format!("{handle:?}").contains("armed: true"));
        assert!(handle.encrypt_outbound(&frame).is_err());
    }
}
