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

/// The two base64 chars of every 12-bit group (half a 3-byte chunk per lookup).
const B64_PAIRS: [[u8; 2]; 4096] = {
    let mut t = [[0u8; 2]; 4096];
    let mut i = 0;
    while i < 4096 {
        t[i] = [B64[i >> 6], B64[i & 63]];
        i += 1;
    }
    t
};

/// Standard base64 (RFC 4648, padded).
pub fn base64(bytes: &[u8]) -> String {
    let mut o = String::with_capacity(bytes.len().div_ceil(3) * 4);
    base64_into(bytes, &mut o);
    o
}

/// [`base64`], appended to `out`.
pub fn base64_into(bytes: &[u8], out: &mut String) {
    let mut o = Vec::with_capacity(bytes.len().div_ceil(3) * 4);
    base64_bytes_into(bytes, &mut o);
    out.push_str(std::str::from_utf8(&o).expect("base64 is ASCII"));
}

/// [`base64`] as bytes, appended to `out` (the output length is exact, so it is written in place).
pub fn base64_bytes_into(bytes: &[u8], out: &mut Vec<u8>) {
    let start = out.len();
    out.resize(start + bytes.len().div_ceil(3) * 4, 0);
    let dst = &mut out[start..];
    let full = bytes.chunks_exact(3);
    let rest = full.remainder();
    let mut d = dst.chunks_exact_mut(4);
    for (c, o) in full.zip(&mut d) {
        let n = (c[0] as usize) << 16 | (c[1] as usize) << 8 | c[2] as usize;
        o[..2].copy_from_slice(&B64_PAIRS[n >> 12]);
        o[2..].copy_from_slice(&B64_PAIRS[n & 0xfff]);
    }
    if let Some(o) = d.next() {
        let n = (rest[0] as u32) << 16 | (*rest.get(1).unwrap_or(&0) as u32) << 8;
        o[0] = B64[(n >> 18) as usize & 63];
        o[1] = B64[(n >> 12) as usize & 63];
        o[2] = if rest.len() > 1 { B64[(n >> 6) as usize & 63] } else { b'=' };
        o[3] = b'=';
    }
}

/// The row's bytes, little-endian float32 (the byte gate compares exactly these).
pub fn row_bytes(row: &[f32; OBS_DIM]) -> Vec<u8> {
    row_array(row).to_vec()
}

/// [`row_bytes`] on the stack.
fn row_array(row: &[f32; OBS_DIM]) -> [u8; OBS_DIM * 4] {
    let mut b = [0u8; OBS_DIM * 4];
    for (o, x) in b.chunks_exact_mut(4).zip(row) {
        o.copy_from_slice(&x.to_le_bytes());
    }
    b
}

/// The frame JSON for one row.
pub fn frame(row: &[f32; OBS_DIM]) -> String {
    let mut o = String::with_capacity(FRAME_LEN);
    frame_into(row, &mut o);
    o
}

/// The byte length of every [`frame`] (the row's length is fixed).
pub const FRAME_LEN: usize = FRAME_HEAD.len() + (OBS_DIM * 4).div_ceil(3) * 4 + 2;
use super::layout::FRAME_HEAD;

/// [`frame`], appended to `out`.
pub fn frame_into(row: &[f32; OBS_DIM], out: &mut String) {
    let mut o = Vec::with_capacity(FRAME_LEN);
    frame_bytes_into(row, &mut o);
    out.push_str(std::str::from_utf8(&o).expect("the frame is ASCII"));
}

/// [`frame`] as bytes, appended to `out`.
pub fn frame_bytes_into(row: &[f32; OBS_DIM], out: &mut Vec<u8>) {
    out.extend_from_slice(FRAME_HEAD.as_bytes());
    base64_bytes_into(&row_array(row), out);
    out.extend_from_slice(b"\"}");
}

/// The `__OBS__` JSON of `side`'s decision on `v` — `sim_bridge`'s core observation frame
/// (`gen3_bridge_core_obs_v1`, `designs/rust_sim/encoder.md` §5a): the request at stream line `line`,
/// the side's `n`-th frame (0-based) of the battle — appended to `out` (UTF-8):
///
/// ```text
/// {"frame":<frame>,"mask":[11 × 0|1],"tokens":{"<idx>":"<choice>",…},"turn":<int>,"line":<int>,"rqid":<int>|null,"n":<int>}
/// ```
///
/// On `Err` (the encode, the legality, the tokens or a non-integer rqid — named `core_obs: …`)
/// nothing is appended.
pub fn obs_json_into(v: &crate::version::BattleVersion, side: usize, line: usize, n: u32, out: &mut Vec<u8>) -> Result<(), String> {
    use crate::core_events::jsonval::Val;
    use crate::present;
    let tag = side + 1;
    let mut row = [0.0f32; OBS_DIM];
    v.encode(side, &mut row).map_err(|e| format!("core_obs: encode p{tag}: {}", e.message()))?;
    let legal = v.legal(side).ok_or_else(|| format!("core_obs: p{tag}: a decision with no legality"))?;
    let reading = &v.stream(side).ok_or_else(|| format!("core_obs: p{tag}: no stream"))?.board_reading;
    let tokens = present::choice_tokens(reading, &legal).map_err(|e| format!("core_obs: tokens p{tag}: {}", e.message()))?;
    let mask = present::mask(&legal);
    let rqid = match reading.last_request.as_ref().and_then(|r| r.get("rqid")) {
        None | Some(Val::Null) => None,
        Some(Val::Int(i)) => Some(*i),
        Some(other) => return Err(format!("core_obs: p{tag}: the request's rqid is not an integer: {other:?}")),
    };
    out.reserve(FRAME_LEN + 256);
    out.extend_from_slice(b"{\"frame\":");
    frame_bytes_into(&row, out);
    out.extend_from_slice(b",\"mask\":[");
    for (i, m) in mask.iter().enumerate() {
        if i > 0 {
            out.push(b',');
        }
        out.extend_from_slice(m.to_string().as_bytes());
    }
    out.extend_from_slice(b"],\"tokens\":");
    out.extend_from_slice(present::tokens_json(&tokens).as_bytes());
    let rqid = rqid.map_or_else(|| "null".to_string(), |i| i.to_string());
    out.extend_from_slice(format!(",\"turn\":{},\"line\":{line},\"rqid\":{rqid},\"n\":{n}}}", reading.turn).as_bytes());
    Ok(())
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
    fn base64_into_matches_the_reference_at_every_remainder() {
        // the reference: the char-at-a-time encoder `base64_into` replaced
        fn reference(bytes: &[u8]) -> String {
            let mut o = String::new();
            for c in bytes.chunks(3) {
                let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
                o.push(B64[(n >> 18) as usize & 63] as char);
                o.push(B64[(n >> 12) as usize & 63] as char);
                o.push(if c.len() > 1 { B64[(n >> 6) as usize & 63] as char } else { '=' });
                o.push(if c.len() > 2 { B64[n as usize & 63] as char } else { '=' });
            }
            o
        }
        let bytes: Vec<u8> = (0..1000u32).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();
        for n in 0..40 {
            assert_eq!(base64(&bytes[..n]), reference(&bytes[..n]), "n={n}");
        }
        assert_eq!(base64(&bytes), reference(&bytes));
        let mut row = [0.0f32; OBS_DIM];
        for (i, x) in row.iter_mut().enumerate() {
            *x = (i as f32 * 0.37).sin() * if i % 7 == 0 { -1.0 } else { 1.0 };
        }
        row[5] = -0.0;
        row[6] = f32::NAN;
        let f = frame(&row);
        assert_eq!(f.len(), FRAME_LEN);
        assert_eq!(f, format!("{{\"dtype\":\"{DTYPE}\",\"shape\":[{OBS_DIM}],\"b64\":\"{}\"}}", reference(&row_bytes(&row))));
    }

    #[test]
    fn the_frame_declares_the_row() {
        let mut row = [0.0f32; OBS_DIM];
        row[0] = 1.5;
        let f = frame(&row);
        let head = format!("{{\"dtype\":\"<f4\",\"shape\":[{OBS_DIM}],\"b64\":\"AADAPw");
        assert!(f.starts_with(&head), "{}", &f[..60]);
    }
}
