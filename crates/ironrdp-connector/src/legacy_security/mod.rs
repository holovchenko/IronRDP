//! Standard RDP Security (RC4) building blocks, see MS-RDPBCGR section 5.3.

mod codec;
mod keys;

pub use self::codec::{LegacySecurityCodec, LegacySecurityHandle};
pub use self::keys::{Cipher, SessionKeys, encrypt_client_random, mac, salted_mac};
