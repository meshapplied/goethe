// Passkey (WebAuthn ES256) assertions: what the browser sends, and the one check the platform and nodes both run on it.

use crate::action::Kind;
use crate::{b64, hex, sha256, unb64};
use ring::signature::{UnparsedPublicKey, ECDSA_P256_SHA256_ASN1};
use serde::{Deserialize, Serialize};

pub(crate) const USER_PRESENT: u8 = 0x01;
pub(crate) const USER_VERIFIED: u8 = 0x04;
const AUTH_DATA_MIN: usize = 37;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Assertion {
    pub key_id: String,
    pub authenticator_data_b64: String,
    pub client_data_json_b64: String,
    pub signature_b64: String,
}

#[derive(Deserialize)]
struct ClientData {
    #[serde(rename = "type")]
    kind: String,
    challenge: String,
    origin: String,
}

pub fn verify(a: &Assertion, public_key_b64: &str, challenge: &[u8], rp_id: &str, origin: &str) -> bool {
    // True only for a user-verified assertion over `challenge` from `origin` for `rp_id`, signed by the P-256 key `public_key_b64`.
    checked(a, public_key_b64, challenge, rp_id, origin).is_some()
}

fn checked(a: &Assertion, public_key_b64: &str, challenge: &[u8], rp_id: &str, origin: &str) -> Option<()> {
    let auth = unb64(&a.authenticator_data_b64)?;
    let client = unb64(&a.client_data_json_b64)?;
    let sig = unb64(&a.signature_b64)?;
    let key = unb64(public_key_b64)?;
    let data: ClientData = serde_json::from_slice(&client).ok()?;
    let flags = USER_PRESENT | USER_VERIFIED;
    let fresh = data.kind == "webauthn.get" && data.origin == origin && unb64(&data.challenge).as_deref() == Some(challenge);
    let scoped = auth.len() >= AUTH_DATA_MIN && auth[..32] == sha256(rp_id.as_bytes()) && auth[32] & flags == flags;
    if !(fresh && scoped) {
        return None;
    }
    let mut signed = auth;
    signed.extend_from_slice(&sha256(&client));
    UnparsedPublicKey::new(&ECDSA_P256_SHA256_ASN1, key).verify(&signed, &sig).ok()
}

pub fn link_challenge(device_code_hash: &str, install_hash: &str) -> [u8; 32] {
    // What the approving person signs when linking a node: this device code for this install key, nothing else.
    sha256(format!("goethe-link-v1\n{device_code_hash}\n{install_hash}").as_bytes())
}

pub fn pair_challenge(key_id: &str, public_key: &str, user_id: &str, cap: Kind) -> [u8; 32] {
    // What a person signs in the console to hand an unattended node their passkey with this cap.
    sha256(format!("goethe-pair-v1\n{key_id}\n{public_key}\n{user_id}\n{}", cap.name()).as_bytes())
}

pub fn key_fingerprint(public_key_b64: &str) -> String {
    // The short form people compare between goethe login and the console: the first 16 hex characters of sha256 of the raw key.
    hex(&sha256(&unb64(public_key_b64).unwrap_or_default()))[..16].to_string()
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pairing {
    pub key_id: String,
    pub public_key: String,
    pub user_id: String,
    pub cap: Kind,
    pub assertion: Assertion,
}

impl Pairing {
    pub fn decode(text: &str) -> Option<Pairing> {
        // The pairing string the console shows: base64url of this JSON.
        serde_json::from_slice(&unb64(text.trim())?).ok()
    }

    pub fn encode(&self) -> String {
        b64(&serde_json::to_vec(self).unwrap_or_default())
    }

    pub fn pair_challenge(&self) -> [u8; 32] {
        pair_challenge(&self.key_id, &self.public_key, &self.user_id, self.cap)
    }

    pub fn signed(&self, challenge: &[u8], rp_id: &str, origin: &str) -> bool {
        // True when this key itself made the assertion over `challenge`, as a pairing string or a link's trust root carries it.
        self.assertion.key_id == self.key_id && !self.user_id.is_empty() && verify(&self.assertion, &self.public_key, challenge, rp_id, origin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::Passkey;
    use crate::{CONSOLE_ORIGIN, RP_ID};

    #[test]
    fn a_link_assertion_binds_device_code_and_install() {
        let pk = Passkey::new("k1");
        let c = link_challenge("dch", "ih");
        let a = pk.assert(&c, RP_ID, CONSOLE_ORIGIN, true);
        assert!(verify(&a, &pk.public_key_b64(), &c, RP_ID, CONSOLE_ORIGIN));
        assert!(!verify(&a, &pk.public_key_b64(), &link_challenge("dch", "other"), RP_ID, CONSOLE_ORIGIN));
    }

    #[test]
    fn a_pairing_string_binds_key_user_and_cap() {
        let pk = Passkey::new("k1");
        let c = pair_challenge("k1", &pk.public_key_b64(), "user1", Kind::Change);
        let p = Pairing { key_id: "k1".into(), public_key: pk.public_key_b64(), user_id: "user1".into(), cap: Kind::Change, assertion: pk.assert(&c, RP_ID, CONSOLE_ORIGIN, true) };
        let back = Pairing::decode(&p.encode()).unwrap();
        assert_eq!(back, p);
        assert!(back.signed(&back.pair_challenge(), RP_ID, CONSOLE_ORIGIN));
        let mut raised = back.clone();
        raised.cap = Kind::Trust;
        assert!(!raised.signed(&raised.pair_challenge(), RP_ID, CONSOLE_ORIGIN), "a raised cap breaks the signature");
        let mut other = back.clone();
        other.user_id = "user2".into();
        assert!(!other.signed(&other.pair_challenge(), RP_ID, CONSOLE_ORIGIN));
        assert!(Pairing::decode("not base64 json").is_none());
        assert_eq!(key_fingerprint(&pk.public_key_b64()).len(), 16);
        assert_ne!(key_fingerprint(&pk.public_key_b64()), key_fingerprint(&Passkey::new("k2").public_key_b64()));
    }
}
