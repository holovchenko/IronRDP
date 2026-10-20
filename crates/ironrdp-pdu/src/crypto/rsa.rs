use std::io;

use der_parser::parse_der;
use num_bigint::BigUint;

/// Smallest accepted modulus: a 32-byte client random must fit.
const MIN_MODULUS_BITS: u64 = 256;

// TESSERA PATCH: visibility widened from `pub(crate)` for the connector's Security Exchange.
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
    // TESSERA PATCH: a server-supplied modulus reaches here before authentication; zero would panic in `modpow`.
    if n.bits() < MIN_MODULUS_BITS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("public key modulus is too short: {} bits", n.bits()),
        ));
    }
    let m = BigUint::from_bytes_le(message);
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
