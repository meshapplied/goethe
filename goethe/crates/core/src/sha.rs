// SHA-256 through ring, which TLS already links in: the digest of some parts, and lowercase hex.

pub fn digest(parts: &[&[u8]]) -> [u8; 32] {
    let mut c = ring::digest::Context::new(&ring::digest::SHA256);
    for p in parts {
        c.update(p);
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(c.finish().as_ref());
    out
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn digest_hex(bytes: &[u8]) -> String {
    hex(&digest(&[bytes]))
}

#[cfg(test)]
mod tests {
    #[test]
    fn matches_the_known_sha256_of_abc() {
        assert_eq!(super::digest_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert_eq!(super::digest(&[b"a", b"bc"]), super::digest(&[b"abc"]));
    }
}
