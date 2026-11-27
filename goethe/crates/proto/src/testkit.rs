// Test helpers (feature testkit): a software passkey that signs like a browser, and the browser that opens a sealed secret.

use crate::command::{prepare, Command, SignedCommand, TrustEntry};
use crate::seal::{aad, aead_key, Sealed};
use crate::webauthn::{Assertion, USER_PRESENT, USER_VERIFIED};
use crate::{b64, sha256, unb64, Kind, CONSOLE_ORIGIN, RP_ID};
use ring::aead::{Aad, Nonce};
use ring::agreement::{agree_ephemeral, EphemeralPrivateKey, UnparsedPublicKey, ECDH_P256};
use ring::rand::SystemRandom;
use ring::signature::{EcdsaKeyPair, KeyPair, ECDSA_P256_SHA256_ASN1_SIGNING};

pub struct Passkey {
    pair: EcdsaKeyPair,
    pub key_id: String,
}

impl Passkey {
    pub fn new(key_id: &str) -> Self {
        let rng = SystemRandom::new();
        let pkcs8 = EcdsaKeyPair::generate_pkcs8(&ECDSA_P256_SHA256_ASN1_SIGNING, &rng).expect("p256 key");
        let pair = EcdsaKeyPair::from_pkcs8(&ECDSA_P256_SHA256_ASN1_SIGNING, pkcs8.as_ref(), &rng).expect("p256 key");
        Passkey { pair, key_id: key_id.into() }
    }

    pub fn public_key_b64(&self) -> String {
        b64(self.pair.public_key().as_ref())
    }

    pub fn trust(&self, user_id: &str, cap: Kind) -> TrustEntry {
        TrustEntry { key_id: self.key_id.clone(), public_key: self.public_key_b64(), user_id: user_id.into(), cap, added_by: user_id.into(), added_ms: 0 }
    }

    pub fn assert(&self, challenge: &[u8], rp_id: &str, origin: &str, verified: bool) -> Assertion {
        // What navigator.credentials.get returns: authenticator data, client data JSON, and a DER signature over both.
        let mut auth = sha256(rp_id.as_bytes()).to_vec();
        auth.push(if verified { USER_PRESENT | USER_VERIFIED } else { USER_PRESENT });
        auth.extend_from_slice(&[0, 0, 0, 1]);
        let client = format!(r#"{{"type":"webauthn.get","challenge":"{}","origin":"{origin}","crossOrigin":false}}"#, b64(challenge));
        let mut signed = auth.clone();
        signed.extend_from_slice(&sha256(client.as_bytes()));
        let sig = self.pair.sign(&SystemRandom::new(), &signed).expect("sign");
        Assertion {
            key_id: self.key_id.clone(),
            authenticator_data_b64: b64(&auth),
            client_data_json_b64: b64(client.as_bytes()),
            signature_b64: b64(sig.as_ref()),
        }
    }

    pub fn sign(&self, commands: &[Command]) -> Vec<SignedCommand> {
        self.sign_as(commands, RP_ID, CONSOLE_ORIGIN, true)
    }

    pub(crate) fn sign_as(&self, commands: &[Command], rp_id: &str, origin: &str, verified: bool) -> Vec<SignedCommand> {
        // One tap over the whole batch, then one signed command per node, as the console submits them.
        let p = prepare(commands).expect("commands share one batch");
        let assertion = self.assert(&p.challenge, rp_id, origin, verified);
        p.bodies_b64
            .into_iter()
            .map(|body_b64| SignedCommand { manifest_b64: p.manifest_b64.clone(), body_b64, assertion: assertion.clone() })
            .collect()
    }
}

pub struct Browser {
    key: EphemeralPrivateKey,
    public: Vec<u8>,
}

impl Browser {
    pub fn new() -> Self {
        let key = EphemeralPrivateKey::generate(&ECDH_P256, &SystemRandom::new()).expect("p256 key");
        let public = key.compute_public_key().expect("public key").as_ref().to_vec();
        Browser { key, public }
    }

    pub fn public_b64(&self) -> String {
        b64(&self.public)
    }

    pub fn open(self, sealed: &Sealed, cmd_id: &str, key_id: &str) -> Option<Vec<u8>> {
        // The WebCrypto side: same ECDH, HKDF and AES-GCM, decrypting in memory.
        let peer = unb64(&sealed.node_pub)?;
        let key = agree_ephemeral(self.key, &UnparsedPublicKey::new(&ECDH_P256, peer), |z| aead_key(z, cmd_id)).ok()??;
        let nonce: [u8; 12] = unb64(&sealed.nonce_b64)?.try_into().ok()?;
        let mut buf = unb64(&sealed.ct_b64)?;
        let plain = key.open_in_place(Nonce::assume_unique_for_key(nonce), Aad::from(aad(cmd_id, key_id)), &mut buf).ok()?;
        Some(plain.to_vec())
    }
}

impl Default for Browser {
    fn default() -> Self {
        Self::new()
    }
}
