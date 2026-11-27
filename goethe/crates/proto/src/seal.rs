// Sealed secrets: the node encrypts a new API key secret to a one-time browser key, so only that browser can read it.

use crate::{b64, unb64};
use ring::aead::{Aad, LessSafeKey, Nonce, UnboundKey, AES_256_GCM};
use ring::agreement::{agree_ephemeral, EphemeralPrivateKey, UnparsedPublicKey, ECDH_P256};
use ring::hkdf::{Salt, HKDF_SHA256};
use ring::rand::{SecureRandom, SystemRandom};
use serde::{Deserialize, Serialize};

const INFO: &[u8] = b"goethe-seal-v1";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sealed {
    pub node_pub: String,
    pub nonce_b64: String,
    pub ct_b64: String,
}

pub fn seal(browser_pub_b64: &str, cmd_id: &str, key_id: &str, secret: &[u8]) -> Option<Sealed> {
    // ECDH P-256 with a fresh node key, HKDF-SHA256 (salt cmd_id, info goethe-seal-v1), AES-256-GCM with aad cmd_id + key_id.
    let rng = SystemRandom::new();
    let peer = unb64(browser_pub_b64)?;
    let mine = EphemeralPrivateKey::generate(&ECDH_P256, &rng).ok()?;
    let node_pub = mine.compute_public_key().ok()?;
    let key = agree_ephemeral(mine, &UnparsedPublicKey::new(&ECDH_P256, peer), |z| aead_key(z, cmd_id)).ok()??;
    let mut nonce = [0u8; 12];
    rng.fill(&mut nonce).ok()?;
    let mut ct = secret.to_vec();
    key.seal_in_place_append_tag(Nonce::assume_unique_for_key(nonce), Aad::from(aad(cmd_id, key_id)), &mut ct).ok()?;
    Some(Sealed { node_pub: b64(node_pub.as_ref()), nonce_b64: b64(&nonce), ct_b64: b64(&ct) })
}

pub(crate) fn aead_key(z: &[u8], cmd_id: &str) -> Option<LessSafeKey> {
    let prk = Salt::new(HKDF_SHA256, cmd_id.as_bytes()).extract(z);
    let okm = prk.expand(&[INFO], &AES_256_GCM).ok()?;
    Some(LessSafeKey::new(UnboundKey::from(okm)))
}

pub(crate) fn aad(cmd_id: &str, key_id: &str) -> Vec<u8> {
    [cmd_id.as_bytes(), key_id.as_bytes()].concat()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::Browser;

    #[test]
    fn only_the_browser_that_asked_can_open_the_secret_for_that_command() {
        let browser = Browser::new();
        let sealed = seal(&browser.public_b64(), "c1", "key1", b"goethe_sk_secret").unwrap();
        assert_eq!(browser.open(&sealed, "c1", "key1").as_deref(), Some(&b"goethe_sk_secret"[..]));
        let other = Browser::new();
        let sealed = seal(&other.public_b64(), "c1", "key1", b"x").unwrap();
        assert!(other.open(&sealed, "c2", "key1").is_none());
        assert!(seal("not-a-key", "c1", "key1", b"x").is_none());
    }
}
