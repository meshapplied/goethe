// The node's install key: Ed25519, made once per install; its hash names the install and its signature opens the node connection.

use crate::{b64, hex, sha256, unb64};
use ring::rand::SystemRandom;
use ring::signature::{Ed25519KeyPair, KeyPair, UnparsedPublicKey, ED25519};

pub const CONNECT_UPGRADE: &str = "goethe-node/1";
pub const PROOF_HEADER: &str = "goethe-proof";

pub struct InstallKey(Ed25519KeyPair);

impl InstallKey {
    pub fn generate() -> Option<Vec<u8>> {
        // A new key as PKCS#8 bytes, for the node to store once (mode 0600) and load with `from_pkcs8`.
        Ed25519KeyPair::generate_pkcs8(&SystemRandom::new()).ok().map(|d| d.as_ref().to_vec())
    }

    pub fn from_pkcs8(bytes: &[u8]) -> Option<Self> {
        Ed25519KeyPair::from_pkcs8(bytes).ok().map(InstallKey)
    }

    pub fn public_b64(&self) -> String {
        b64(self.0.public_key().as_ref())
    }

    pub fn hash(&self) -> String {
        install_hash(self.0.public_key().as_ref())
    }

    pub fn connect_proof(&self, nonce: &str, node_id: &str) -> String {
        b64(self.0.sign(&connect_message(nonce, node_id)).as_ref())
    }
}

pub fn install_hash(public_key: &[u8]) -> String {
    hex(&sha256(public_key))
}

pub fn verify_connect(install_pub_b64: &str, nonce: &str, node_id: &str, proof_b64: &str) -> bool {
    // True when the install key signed this nonce for this node; the platform also checks the key's hash against the one it stored at enroll.
    match (unb64(install_pub_b64), unb64(proof_b64)) {
        (Some(key), Some(sig)) => UnparsedPublicKey::new(&ED25519, key).verify(&connect_message(nonce, node_id), &sig).is_ok(),
        _ => false,
    }
}

fn connect_message(nonce: &str, node_id: &str) -> Vec<u8> {
    format!("goethe-connect-v1\n{nonce}\n{node_id}").into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_install_key_opens_the_connection_for_that_nonce_and_node() {
        let key = InstallKey::from_pkcs8(&InstallKey::generate().unwrap()).unwrap();
        let proof = key.connect_proof("nonce1", "n1");
        assert!(verify_connect(&key.public_b64(), "nonce1", "n1", &proof));
        assert!(!verify_connect(&key.public_b64(), "nonce2", "n1", &proof));
        assert!(!verify_connect(&key.public_b64(), "nonce1", "n2", &proof));
        let other = InstallKey::from_pkcs8(&InstallKey::generate().unwrap()).unwrap();
        assert!(!verify_connect(&other.public_b64(), "nonce1", "n1", &proof));
        assert_eq!(key.hash(), install_hash(&unb64(&key.public_b64()).unwrap()));
        assert_eq!(key.hash().len(), 64);
    }
}
