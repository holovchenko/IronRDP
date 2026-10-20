use std::io;

use der_parser::parse_der;
use num_bigint::BigUint;

/// Smallest accepted modulus. This is only a sanity floor: it does not make every message fit,
/// that is checked separately against the actual modulus (`m < n`).
const MIN_MODULUS_BITS: u64 = 256;

/// Largest accepted modulus: the key arrives from the server before authentication, and `modpow`
/// cost grows with the modulus size.
const MAX_MODULUS_BITS: u64 = 16384;

// TESSERA PATCH (see rdp/ironrdp-fork.md in Tessera): visibility widened from `pub(crate)` for the connector's Security Exchange.
pub fn encrypt_with_public_key(message: &[u8], public_key_der: &[u8]) -> io::Result<Vec<u8>> {
    let (_, der_object) = parse_der(public_key_der).map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unable to parse public key from DER: {err:?}"),
        )
    })?;

    let der_object_sequence = der_object.as_sequence().map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unable to extract a sequence from the DER object: {err:?}"),
        )
    })?;

    if der_object_sequence.len() != 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "DER object sequence is empty",
        ));
    }

    let n = der_object_sequence[0].as_slice().map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unable to extract a slice from public key modulus sequence: {err:?}"),
        )
    })?;

    let e = der_object_sequence[1].as_slice().map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unable to extract a slice from public key exponent sequence: {err:?}"),
        )
    })?;

    let n = BigUint::from_bytes_be(n);
    let e = BigUint::from_bytes_be(e);
    // TESSERA PATCH (see rdp/ironrdp-fork.md in Tessera): the key reaches here before authentication.
    // A zero modulus would panic in `modpow`, an oversized one is a CPU-exhaustion vector, and a
    // message that does not fit (`m >= n`) would be silently reduced, so the server would derive other keys.
    if n.bits() < MIN_MODULUS_BITS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("public key modulus is too short: {} bits", n.bits()),
        ));
    }
    if n.bits() > MAX_MODULUS_BITS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("public key modulus is too long: {} bits", n.bits()),
        ));
    }
    if e.bits() == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "public key exponent is zero",
        ));
    }
    let m = BigUint::from_bytes_le(message);
    if m >= n {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "message does not fit the public key modulus",
        ));
    }
    let c = m.modpow(&e, &n);

    // TESSERA PATCH (see rdp/ironrdp-fork.md in Tessera): keep the full modulus width,
    // `to_bytes_le` drops zero high-order bytes.
    let mut result = c.to_bytes_le();
    let width = usize::try_from(n.bits().div_ceil(8)).unwrap_or(usize::MAX);
    result.resize(width.saturating_add(8), 0u8);

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn der_integer(bytes: &[u8]) -> Vec<u8> {
        [&[0x02, u8::try_from(bytes.len()).unwrap()], bytes].concat()
    }

    fn public_key_der(modulus: &[u8], exponent: &[u8]) -> Vec<u8> {
        let body = [der_integer(modulus), der_integer(exponent)].concat();
        [&[0x30, u8::try_from(body.len()).unwrap()], body.as_slice()].concat()
    }

    #[test]
    fn zero_modulus_is_an_error_not_a_panic() {
        let der = public_key_der(&[0], &[0x01, 0x00, 0x01]);

        let err = encrypt_with_public_key(&[0x11; 32], &der).unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn modulus_shorter_than_256_bits_is_rejected() {
        let der = public_key_der(&[0xFF; 31], &[0x01, 0x00, 0x01]);

        let err = encrypt_with_public_key(&[0x11; 32], &der).unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn message_not_below_modulus_is_rejected() {
        // 256-bit modulus 0x80 00..00; the little-endian message 0xFF.. is above it.
        let modulus = [&[0x00u8, 0x80][..], &[0; 31]].concat();
        let der = public_key_der(&modulus, &[0x01, 0x00, 0x01]);

        let err = encrypt_with_public_key(&[0xFF; 32], &der).unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("does not fit"), "{err}");
    }

    #[test]
    fn message_equal_to_modulus_is_rejected() {
        let mut modulus = vec![0x00u8, 0x80];
        modulus.extend_from_slice(&[0; 31]);
        let mut message = vec![0u8; 31];
        message.push(0x80);
        let der = public_key_der(&modulus, &[0x03]);

        let err = encrypt_with_public_key(&message, &der).unwrap_err();

        assert!(err.to_string().contains("does not fit"), "{err}");
    }

    #[test]
    fn zero_exponent_is_rejected() {
        let modulus = [&[0x00u8][..], &[0xFF; 32]].concat();
        let der = public_key_der(&modulus, &[0x00]);

        let err = encrypt_with_public_key(&[0x11; 32], &der).unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("exponent"), "{err}");
    }

    #[test]
    fn oversized_modulus_is_rejected() {
        // 2049 bytes of 0xFF is 16392 bits; DER long-form length for the INTEGER.
        let modulus = [&[0x00u8][..], &[0xFF; 2049]].concat();
        let integer = [
            &[0x02, 0x82],
            &u16::try_from(modulus.len()).unwrap().to_be_bytes()[..],
            &modulus,
        ]
        .concat();
        let exponent = der_integer(&[0x01, 0x00, 0x01]);
        let body = [integer, exponent].concat();
        let der = [
            &[0x30, 0x82][..],
            &u16::try_from(body.len()).unwrap().to_be_bytes(),
            &body,
        ]
        .concat();

        let err = encrypt_with_public_key(&[0x11; 32], &der).unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("too long"), "{err}");
    }

    #[test]
    fn output_keeps_modulus_width_plus_eight_when_ciphertext_has_zero_high_byte() {
        // Exponent 1 makes the ciphertext equal to the message, so its high-order bytes are zero.
        let modulus = [&[0x00u8][..], &[0xFF; 32]].concat();
        let der = public_key_der(&modulus, &[0x01]);

        let out = encrypt_with_public_key(&[1, 2, 3], &der).unwrap();

        assert_eq!(out.len(), 32 + 8);
        assert_eq!(out[..3], [1, 2, 3]);
        assert!(out[3..].iter().all(|byte| *byte == 0));
    }
}
