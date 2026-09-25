//! The ROW on the wire (`gen3_core_obs_wire_v1`): the observation travels in the reply of a pipe
//! that already exists (program M4 "Transport"), as a self-describing frame
//!
//! ```text
//! {"dtype":"<f4","shape":[2501],"b64":"<the row's little-endian float32 bytes, base64>"}
//! ```
//!
//! which Python wraps with `np.frombuffer` (`agents.battle.core_obs.wrap_row`) — and REFUSES,
//! never converts, a frame whose dtype, shape or byte length is not the observation's.

use super::OBS_DIM;

/// The frame's declared dtype: little-endian float32.
pub const DTYPE: &str = "<f4";

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Standard base64 (RFC 4648, padded).
pub fn base64(bytes: &[u8]) -> String {
    let mut o = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        o.push(B64[(n >> 18) as usize & 63] as char);
        o.push(B64[(n >> 12) as usize & 63] as char);
        o.push(if c.len() > 1 { B64[(n >> 6) as usize & 63] as char } else { '=' });
        o.push(if c.len() > 2 { B64[n as usize & 63] as char } else { '=' });
    }
    o
}

/// The row's bytes, little-endian float32 (the byte gate compares exactly these).
pub fn row_bytes(row: &[f32; OBS_DIM]) -> Vec<u8> {
    let mut b = Vec::with_capacity(OBS_DIM * 4);
    for x in row {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

/// The frame JSON for one row.
pub fn frame(row: &[f32; OBS_DIM]) -> String {
    format!("{{\"dtype\":\"{DTYPE}\",\"shape\":[{OBS_DIM}],\"b64\":\"{}\"}}", base64(&row_bytes(row)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_rfc4648_vectors() {
        for (i, o) in [("", ""), ("f", "Zg=="), ("fo", "Zm8="), ("foo", "Zm9v"), ("foob", "Zm9vYg=="), ("fooba", "Zm9vYmE="), ("foobar", "Zm9vYmFy")] {
            assert_eq!(base64(i.as_bytes()), o);
        }
    }

    #[test]
    fn the_frame_declares_the_row() {
        let mut row = [0.0f32; OBS_DIM];
        row[0] = 1.5;
        let f = frame(&row);
        assert!(f.starts_with("{\"dtype\":\"<f4\",\"shape\":[2501],\"b64\":\"AADAPw"), "{}", &f[..60]);
    }
}
