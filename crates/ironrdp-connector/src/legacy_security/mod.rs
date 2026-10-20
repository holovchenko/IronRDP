//! Standard RDP Security (RC4) building blocks, see MS-RDPBCGR section 5.3.

mod keys;

pub use self::keys::{Cipher, SessionKeys, encrypt_client_random, mac, salted_mac};
