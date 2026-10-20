//! Security Exchange PDU.
//!
//! Defined in [\[MS-RDPBCGR\] 2.2.1.10].
//!
//! [\[MS-RDPBCGR\] 2.2.1.10]: https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-rdpbcgr/b9b0a1e4-4d03-4cb1-9dd7-1d2e1b1f1d2c

use ironrdp_core::{
    Decode, DecodeResult, Encode, EncodeResult, ReadCursor, WriteCursor, cast_length, ensure_fixed_part_size,
    ensure_size,
};

use crate::rdp::headers::{BasicSecurityHeader, BasicSecurityHeaderFlags};

/// Security Exchange PDU: carries the client random encrypted with the server public key.
///
/// Defined in [\[MS-RDPBCGR\] 2.2.1.10].
///
/// [\[MS-RDPBCGR\] 2.2.1.10]: https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-rdpbcgr/b9b0a1e4-4d03-4cb1-9dd7-1d2e1b1f1d2c
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub struct SecurityExchangePdu {
    pub encrypted_client_random: Vec<u8>,
}

impl SecurityExchangePdu {
    const NAME: &'static str = "SecurityExchangePdu";

    const FIXED_PART_SIZE: usize = BasicSecurityHeader::FIXED_PART_SIZE + 4 /* length */;
}

impl Encode for SecurityExchangePdu {
    fn encode(&self, dst: &mut WriteCursor<'_>) -> EncodeResult<()> {
        ensure_size!(in: dst, size: self.size());

        BasicSecurityHeader {
            flags: BasicSecurityHeaderFlags::EXCHANGE_PKT,
        }
        .encode(dst)?;
        dst.write_u32(cast_length!("length", self.encrypted_client_random.len())?);
        dst.write_slice(&self.encrypted_client_random);

        Ok(())
    }

    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn size(&self) -> usize {
        Self::FIXED_PART_SIZE + self.encrypted_client_random.len()
    }
}

impl<'de> Decode<'de> for SecurityExchangePdu {
    fn decode(src: &mut ReadCursor<'de>) -> DecodeResult<Self> {
        ensure_fixed_part_size!(in: src);

        let _header = BasicSecurityHeader::decode(src)?;
        let length: usize = cast_length!("length", src.read_u32())?;

        ensure_size!(in: src, size: length);
        let encrypted_client_random = src.read_slice(length).to_vec();

        Ok(Self {
            encrypted_client_random,
        })
    }
}

#[cfg(test)]
mod tests {
    use ironrdp_core::{decode, encode_vec};

    use super::*;

    // Client random encrypted for the server public key, MS-RDPBCGR 4.1.9 (shared with the
    // legacy_security key tests in ironrdp-connector).
    const ENCRYPTED_CLIENT_RANDOM: &str = "affb041af5d13a0b2acead80dc0392d25e9de3b00d96d28ac76a08adee8d0c0dfd52da472aaa8967fb6958c97a7177120f9cb790942f8005692db56f48bcb0150000000000000000";

    fn unhex(s: &str) -> Vec<u8> {
        s.as_bytes()
            .chunks(2)
            .map(|pair| u8::from_str_radix(core::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect()
    }

    #[test]
    fn security_exchange_round_trips_spec_4_1_9() {
        let random = unhex(ENCRYPTED_CLIENT_RANDOM);
        let mut bytes = vec![0x01, 0x00, 0x00, 0x00];
        bytes.extend_from_slice(&u32::try_from(random.len()).unwrap().to_le_bytes());
        bytes.extend_from_slice(&random);

        let pdu = decode::<SecurityExchangePdu>(&bytes).unwrap();

        assert_eq!(pdu.encrypted_client_random, random);
        assert_eq!(encode_vec(&pdu).unwrap(), bytes);
    }

    #[test]
    fn security_exchange_rejects_truncated_random() {
        let bytes = [0x01, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0xAA];

        assert!(decode::<SecurityExchangePdu>(&bytes).is_err());
    }
}
