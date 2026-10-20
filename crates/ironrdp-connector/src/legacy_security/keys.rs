use core::fmt;

use ironrdp_pdu::crypto::rc4::Rc4;
use ironrdp_pdu::crypto::rsa::encrypt_with_public_key;
use ironrdp_pdu::gcc::EncryptionMethod;
use md5::Digest as _;
use sha1::Digest as _;

use crate::{ConnectorError, ConnectorErrorExt as _, ConnectorResult};

const KEY_UPDATE_INTERVAL: u32 = 4096;
const PAD1: [u8; 40] = [0x36; 40];
const PAD2: [u8; 48] = [0x5C; 48];
const SALT_40_BIT: [u8; 3] = [0xD1, 0x26, 0x9E];
const SALT_56_BIT: [u8; 1] = [0xD1];

/// Negotiated RC4 key strength (MS-RDPBCGR 5.3.5.1, Table 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KeyStrength {
    Bit40,
    Bit56,
    Bit128,
}

impl KeyStrength {
    fn from_method(method: EncryptionMethod) -> ConnectorResult<Self> {
        if method == EncryptionMethod::BIT_40 {
            Ok(Self::Bit40)
        } else if method == EncryptionMethod::BIT_56 {
            Ok(Self::Bit56)
        } else if method == EncryptionMethod::BIT_128 {
            Ok(Self::Bit128)
        } else if method == EncryptionMethod::FIPS {
            Err(ConnectorError::reason(
                "legacy security",
                "server requires FIPS encryption, which Tessera does not support",
            ))
        } else {
            Err(ConnectorError::reason(
                "legacy security",
                format!("unsupported encryption method: {method:?}"),
            ))
        }
    }

    /// Reduces a key to this strength: the first 64 bits with the leading bytes salted (a no-op for 128 bit).
    fn salt(self, key: &[u8]) -> Vec<u8> {
        let (salt, key_len): (&[u8], usize) = match self {
            Self::Bit40 => (&SALT_40_BIT, 8),
            Self::Bit56 => (&SALT_56_BIT, 8),
            Self::Bit128 => (&[], 16),
        };

        let mut salted = key[..key_len].to_vec();
        salted[..salt.len()].copy_from_slice(salt);
        salted
    }
}

/// Session keys for Standard RDP Security, named from the client's point of view.
pub struct SessionKeys {
    pub mac_key: Vec<u8>,
    pub encrypt: Cipher,
    pub decrypt: Cipher,
}

impl fmt::Debug for SessionKeys {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SessionKeys").finish_non_exhaustive()
    }
}

impl SessionKeys {
    /// Derives the initial session keys, MS-RDPBCGR 5.3.5.1 and 5.3.5.3.
    pub fn derive(
        client_random: &[u8; 32],
        server_random: &[u8; 32],
        method: EncryptionMethod,
    ) -> ConnectorResult<Self> {
        let strength = KeyStrength::from_method(method)?;

        let pre_master_secret = [&client_random[..24], &server_random[..24]].concat();
        let salted_hash = |secret: &[u8], label: &[u8]| -> [u8; 16] {
            let sha = sha1_of(&[label, secret, client_random, server_random]);
            md5_of(&[secret, &sha])
        };

        let master_secret: Vec<u8> = [&b"A"[..], b"BB", b"CCC"]
            .iter()
            .flat_map(|label| salted_hash(&pre_master_secret, label))
            .collect();
        let session_key_blob: Vec<u8> = [&b"X"[..], b"YY", b"ZZZ"]
            .iter()
            .flat_map(|label| salted_hash(&master_secret, label))
            .collect();

        let final_hash = |key: &[u8]| md5_of(&[key, client_random, server_random]);
        // The server encrypts with the second block and decrypts with the third, so the client does the reverse.
        let client_decrypt_key = final_hash(&session_key_blob[16..32]);
        let client_encrypt_key = final_hash(&session_key_blob[32..48]);

        Ok(Self {
            mac_key: strength.salt(&session_key_blob[..16]),
            encrypt: Cipher::new(strength, strength.salt(&client_encrypt_key)),
            decrypt: Cipher::new(strength, strength.salt(&client_decrypt_key)),
        })
    }
}

/// One direction of the RC4 stream, with the 5.3.7 key update every 4096 packets.
pub struct Cipher {
    strength: KeyStrength,
    initial_key: Vec<u8>,
    current_key: Vec<u8>,
    rc4: Rc4,
    use_count: u32,
}

impl fmt::Debug for Cipher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Cipher")
            .field("strength", &self.strength)
            .field("use_count", &self.use_count)
            .finish_non_exhaustive()
    }
}

impl Cipher {
    fn new(strength: KeyStrength, key: Vec<u8>) -> Self {
        Self {
            strength,
            rc4: Rc4::new(&key),
            initial_key: key.clone(),
            current_key: key,
            use_count: 0,
        }
    }

    /// Number of calls to [`process`](Self::process) since the last key update.
    pub fn use_count(&self) -> u32 {
        self.use_count
    }

    /// Applies the RC4 stream in place, updating the key first when 4096 packets were already processed.
    pub fn process(&mut self, data: &mut [u8]) {
        if self.use_count == KEY_UPDATE_INTERVAL {
            self.update_key();
        }

        data.copy_from_slice(&self.rc4.process(data));
        self.use_count += 1;
    }

    fn update_key(&mut self) {
        let key_len = self.initial_key.len();

        let sha = sha1_of(&[&self.initial_key, &PAD1, &self.current_key]);
        let temp_key = md5_of(&[&self.initial_key, &PAD2, &sha]);

        let temp_key = &temp_key[..key_len];
        let new_key = Rc4::new(temp_key).process(temp_key);

        self.current_key = self.strength.salt(&new_key);
        self.rc4 = Rc4::new(&self.current_key);
        self.use_count = 0;
    }
}

/// MAC over unencrypted data, MS-RDPBCGR 5.3.6.1.
pub fn mac(mac_key: &[u8], data: &[u8]) -> [u8; 8] {
    mac_over(mac_key, data, &[])
}

/// MAC salted with the number of packets this direction already encrypted, MS-RDPBCGR 5.3.6.1.1.
pub fn salted_mac(mac_key: &[u8], data: &[u8], encryption_count: u32) -> [u8; 8] {
    mac_over(mac_key, data, &encryption_count.to_le_bytes())
}

fn mac_over(mac_key: &[u8], data: &[u8], salt: &[u8]) -> [u8; 8] {
    // Frames are bounded by the 16-bit PDU lengths, so this cannot truncate.
    let data_length = u32::try_from(data.len()).unwrap_or(u32::MAX).to_le_bytes();

    let sha = sha1_of(&[mac_key, &PAD1, &data_length, data, salt]);
    let digest = md5_of(&[mac_key, &PAD2, &sha]);

    let mut signature = [0; 8];
    signature.copy_from_slice(&digest[..8]);
    signature
}

/// RSA-encrypts the client random for the Security Exchange PDU, MS-RDPBCGR 5.3.4.1.
///
/// The random is interpreted as a little-endian integer and the result is little-endian,
/// followed by 8 bytes of zero padding.
pub fn encrypt_client_random(client_random: &[u8; 32], public_key_der: &[u8]) -> ConnectorResult<Vec<u8>> {
    encrypt_with_public_key(client_random, public_key_der)
        .map_err(|e| ConnectorError::custom("encrypt client random", e))
}

fn md5_of(parts: &[&[u8]]) -> [u8; 16] {
    let mut hasher = md5::Md5::new();
    for part in parts {
        hasher.update(part);
    }
    hasher.finalize().into()
}

fn sha1_of(parts: &[&[u8]]) -> [u8; 20] {
    let mut hasher = sha1::Sha1::new();
    for part in parts {
        hasher.update(part);
    }
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Source of the server random and server public key (RSA_PUBLIC_KEY, 512-bit):
    // https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-rdpbcgr/d23f7725-876c-48d4-9e41-8288896a19d3
    // (4.1.4 Server MCS Connect Response PDU with GCC Conference Create Response)
    //
    // The spec's annotated sequence publishes the encrypted client random (4.1.9,
    // .../b6075470-bbdd-465a-b6d9-ef15941ae358) and the encrypted Client Info PDU (4.1.10,
    // .../ac6dc9ab-6f32-471e-8374-f80caab50069), but neither the plaintext client random nor the
    // server private key, so those vectors cannot be reproduced. The expected values below were
    // computed with an independent implementation of MS-RDPBCGR 5.3.4-5.3.7 (Python hashlib plus a
    // hand-written RC4) from the published server random/public key and a fixed client random.
    const SERVER_RANDOM: &str = "1011772030610a12e434a11ef2c39f317da45f01893496e0ff1108697f1ac3d2";
    const SERVER_MODULUS_BE: &str = "bc125626638883d1f6c11fa6551fb1a031fd01a2e57968ea7532ffe8ae552d3cc89bfbc4960270f83d0c5ff1cba5f1d69471f8872e5fd50555c3616dbafe81cb";
    const CLIENT_RANDOM: &str = "0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20";
    const PLAINTEXT: &str = "09040904b3430000a0a1a2a3a4a5a6a7a8a9aaabacadaeafb0b1b2b3b4b5b6b7";
    const ENCRYPTED_CLIENT_RANDOM: &str = "affb041af5d13a0b2acead80dc0392d25e9de3b00d96d28ac76a08adee8d0c0dfd52da472aaa8967fb6958c97a7177120f9cb790942f8005692db56f48bcb0150000000000000000";

    const MAC_KEY_128: &str = "340cd4ffb43d656d6d2ec9f8e6ccdfe3";
    const ENCRYPT_KEY_128: &str = "edce7bd643965ff645e9edf1e4022f15";
    const DECRYPT_KEY_128: &str = "c19554751f8b51408422925fe5e462fc";
    const MAC_128: &str = "c9cdd9bd528b5039";
    const SALTED_MAC_128_COUNT_5: &str = "f084de26b0acff30";
    const CIPHERTEXT_128: &str = "4746570038d531ae39d4e63c9fc6b6cfcf55293b658dbef321d2a13d950a7691";
    const AFTER_FIRST_UPDATE_128: &str = "4ca71188016518c973d503a5483ac25f";
    const AFTER_SECOND_UPDATE_128: &str = "ddb796f48e50adbc3b23802726d5c289";

    const MAC_KEY_56: &str = "d10cd4ffb43d656d";
    const ENCRYPT_KEY_56: &str = "d1ce7bd643965ff6";
    const CIPHERTEXT_56: &str = "fc7b78ce552e753685be8439762d2324a314892cf64a8c5cc9c52d1cb37f8dea";
    const AFTER_FIRST_UPDATE_56: &str = "f9be313d8889995cb1d45850f8a2e425";

    const MAC_KEY_40: &str = "d1269effb43d656d";
    const ENCRYPT_KEY_40: &str = "d1269ed643965ff6";
    const DECRYPT_KEY_40: &str = "d1269e751f8b5140";
    const CIPHERTEXT_40: &str = "4ff2d70dcd4a470bd9214e4c6a8610ffbd06c0543762a72fd9d7698e5bbff02d";
    const AFTER_FIRST_UPDATE_40: &str = "b96d40e0ac1c719b7710eabe36e2e22d";

    fn unhex(s: &str) -> Vec<u8> {
        s.as_bytes()
            .chunks(2)
            .map(|pair| u8::from_str_radix(core::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect()
    }

    fn random(s: &str) -> [u8; 32] {
        unhex(s).try_into().unwrap()
    }

    fn server_public_key_der() -> Vec<u8> {
        // SEQUENCE { INTEGER modulus (leading 0x00, high bit set), INTEGER 65537 }
        let mut der = vec![0x30, 0x48, 0x02, 0x41, 0x00];
        der.extend(unhex(SERVER_MODULUS_BE));
        der.extend([0x02, 0x03, 0x01, 0x00, 0x01]);
        der
    }

    fn derive(method: EncryptionMethod) -> SessionKeys {
        SessionKeys::derive(&random(CLIENT_RANDOM), &random(SERVER_RANDOM), method).unwrap()
    }

    fn encrypt_one_byte_packets(cipher: &mut Cipher, count: usize) {
        for _ in 0..count {
            cipher.process(&mut [0]);
        }
    }

    fn process_zeros(cipher: &mut Cipher) -> Vec<u8> {
        let mut buf = [0u8; 16];
        cipher.process(&mut buf);
        buf.to_vec()
    }

    #[test]
    fn encrypts_client_random_for_server_public_key() {
        let encrypted = encrypt_client_random(&random(CLIENT_RANDOM), &server_public_key_der()).unwrap();

        assert_eq!(encrypted, unhex(ENCRYPTED_CLIENT_RANDOM));
    }

    #[test]
    fn encrypted_client_random_keeps_full_modulus_length_when_high_bytes_are_zero() {
        // Chosen so that the ciphertext integer has two zero high-order bytes.
        let client_random = random("44fa02030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f");

        let encrypted = encrypt_client_random(&client_random, &server_public_key_der()).unwrap();

        assert_eq!(
            encrypted,
            unhex(
                "8f0da2792c21441b700f887d041cbb26d2b4fbcce23efe0fb1951c0927f5b3fc15b328c9adbdb8b6cdcc63cc99a18fdf47af6b969495545a20037ad55aea00000000000000000000"
            )
        );
    }

    #[test]
    fn derives_128_bit_keys_and_reproduces_mac_and_ciphertext() {
        let mut keys = derive(EncryptionMethod::BIT_128);

        assert_eq!(keys.mac_key, unhex(MAC_KEY_128));
        assert_eq!(mac(&keys.mac_key, &unhex(PLAINTEXT)), unhex(MAC_128).as_slice());

        let mut buf = unhex(PLAINTEXT);
        keys.encrypt.process(&mut buf);
        assert_eq!(buf, unhex(CIPHERTEXT_128));
    }

    #[test]
    fn client_encrypt_and_decrypt_keys_are_swapped_relative_to_server() {
        let mut keys = derive(EncryptionMethod::BIT_128);

        let mut expected_encrypt = unhex(PLAINTEXT);
        expected_encrypt = Rc4::new(&unhex(ENCRYPT_KEY_128)).process(&expected_encrypt);
        let mut expected_decrypt = unhex(PLAINTEXT);
        expected_decrypt = Rc4::new(&unhex(DECRYPT_KEY_128)).process(&expected_decrypt);

        let mut buf = unhex(PLAINTEXT);
        keys.encrypt.process(&mut buf);
        assert_eq!(buf, expected_encrypt);

        let mut buf = unhex(PLAINTEXT);
        keys.decrypt.process(&mut buf);
        assert_eq!(buf, expected_decrypt);
    }

    #[test]
    fn salting_40_bit() {
        let mut keys = derive(EncryptionMethod::BIT_40);

        assert_eq!(keys.mac_key, unhex(MAC_KEY_40));
        assert_eq!(keys.mac_key[..3], [0xD1, 0x26, 0x9E]);

        let mut buf = unhex(PLAINTEXT);
        keys.encrypt.process(&mut buf);
        assert_eq!(buf, unhex(CIPHERTEXT_40));
        assert_eq!(buf, Rc4::new(&unhex(ENCRYPT_KEY_40)).process(&unhex(PLAINTEXT)));

        let mut buf = unhex(PLAINTEXT);
        keys.decrypt.process(&mut buf);
        assert_eq!(buf, Rc4::new(&unhex(DECRYPT_KEY_40)).process(&unhex(PLAINTEXT)));
    }

    #[test]
    fn salting_56_bit() {
        let mut keys = derive(EncryptionMethod::BIT_56);

        assert_eq!(keys.mac_key, unhex(MAC_KEY_56));
        assert_eq!(keys.mac_key[0], 0xD1);

        let mut buf = unhex(PLAINTEXT);
        keys.encrypt.process(&mut buf);
        assert_eq!(buf, unhex(CIPHERTEXT_56));
        assert_eq!(buf, Rc4::new(&unhex(ENCRYPT_KEY_56)).process(&unhex(PLAINTEXT)));
    }

    #[test]
    fn key_update_after_4096_packets() {
        let mut cipher = derive(EncryptionMethod::BIT_128).encrypt;
        let mut continued_stream = Rc4::new(&unhex(ENCRYPT_KEY_128));
        continued_stream.process(&[0; 4096]);

        encrypt_one_byte_packets(&mut cipher, 4096);
        assert_eq!(cipher.use_count(), 4096);

        let after_update = process_zeros(&mut cipher);
        assert_eq!(cipher.use_count(), 1);
        assert_ne!(after_update, continued_stream.process(&[0; 16]));
        assert_eq!(after_update, unhex(AFTER_FIRST_UPDATE_128));
    }

    #[test]
    fn no_key_update_before_4096_packets() {
        let mut cipher = derive(EncryptionMethod::BIT_128).encrypt;
        let mut continued_stream = Rc4::new(&unhex(ENCRYPT_KEY_128));
        continued_stream.process(&[0; 4095]);

        encrypt_one_byte_packets(&mut cipher, 4095);

        assert_eq!(process_zeros(&mut cipher), continued_stream.process(&[0; 16]));
        assert_eq!(cipher.use_count(), 4096);
    }

    #[test]
    fn second_key_update_uses_the_updated_key_as_current() {
        let mut cipher = derive(EncryptionMethod::BIT_128).encrypt;

        encrypt_one_byte_packets(&mut cipher, 8192);

        assert_eq!(process_zeros(&mut cipher), unhex(AFTER_SECOND_UPDATE_128));
    }

    #[test]
    fn key_update_resalts_40_and_56_bit_keys() {
        let mut cipher = derive(EncryptionMethod::BIT_40).encrypt;
        encrypt_one_byte_packets(&mut cipher, 4096);
        assert_eq!(process_zeros(&mut cipher), unhex(AFTER_FIRST_UPDATE_40));

        let mut cipher = derive(EncryptionMethod::BIT_56).encrypt;
        encrypt_one_byte_packets(&mut cipher, 4096);
        assert_eq!(process_zeros(&mut cipher), unhex(AFTER_FIRST_UPDATE_56));
    }

    #[test]
    fn salted_mac_matches_vector_and_differs_by_count() {
        let key = unhex(MAC_KEY_128);
        let data = unhex(PLAINTEXT);

        assert_eq!(salted_mac(&key, &data, 5), unhex(SALTED_MAC_128_COUNT_5).as_slice());
        assert_ne!(salted_mac(&key, &data, 0), salted_mac(&key, &data, 1));
        assert_ne!(salted_mac(&key, &data, 0), mac(&key, &data));
    }

    #[test]
    fn rejects_fips_and_unknown_methods() {
        let err = SessionKeys::derive(&random(CLIENT_RANDOM), &random(SERVER_RANDOM), EncryptionMethod::FIPS)
            .err()
            .unwrap();
        assert!(
            format!("{err:?}").contains("server requires FIPS encryption, which Tessera does not support"),
            "{err:?}"
        );

        assert!(
            SessionKeys::derive(
                &random(CLIENT_RANDOM),
                &random(SERVER_RANDOM),
                EncryptionMethod::empty()
            )
            .is_err()
        );
    }
}
