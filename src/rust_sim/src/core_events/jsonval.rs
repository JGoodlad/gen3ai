//! [`Val`] — a small byte-level JSON reader for the core: it keeps what `crate::json` does not (an
//! integer vs a float literal, an object's key ORDER) and does not copy the input into a
//! `Vec<char>`, so it is also the fast path for the `|request|` frames the reading folds.

/// A JSON value that keeps what the record needs and `crate::json` does not: an integer vs a
/// float literal, and an object's key ORDER.
#[derive(Debug, Clone, PartialEq)]
pub enum Val {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(JStr),
    Arr(Vec<Val>),
    Obj(Vec<(JStr, Val)>),
}

/// A JSON string — BORROWED from [`interned`] when it is one of the words every `|request|`
/// repeats (its keys, a few values), owned otherwise. It reads, compares, clones and prints
/// (`Debug`) exactly as a `String` of the same text; the borrow only saves the allocation (a
/// request carries ~190 strings, ~115 of them keys).
pub type JStr = std::borrow::Cow<'static, str>;

/// The interned words, by length: the `|request|` vocabulary (Showdown's `side.getRequestData` /
/// `getMoveRequestData` keys, the move targets, and the values every request repeats). Membership is
/// a speed choice only — any string reads the same either way.
fn interned(len: usize) -> &'static [&'static str] {
    match len {
        2 => &["id", "p1", "p2", "pp"],
        3 => &["all", "any", "atk", "def", "spa", "spd", "spe"],
        4 => &["item", "move", "name", "rqid", "self", "side", "wait"],
        5 => &["ident", "maxpp", "moves", "stats"],
        6 => &["active", "allies", "normal", "target"],
        7 => &["ability", "details", "foeSide", "pokemon", "trapped"],
        8 => &["allySide", "disabled", "noCancel", "pokeball", "reviving", "scripted"],
        9 => &["condition", "leftovers"],
        10 => &["commanding"],
        11 => &["adjacentFoe", "allAdjacent", "baseAbility", "forceSwitch", "teamPreview"],
        12 => &["adjacentAlly", "maybeTrapped", "randomNormal"],
        15 => &["allAdjacentFoes"],
        18 => &["adjacentAllyOrSelf"],
        _ => &[],
    }
}

fn intern(t: &str) -> JStr {
    match interned(t.len()).iter().find(|w| **w == t) {
        Some(w) => JStr::Borrowed(w),
        None => JStr::Owned(t.to_string()),
    }
}

impl Val {
    pub fn parse(s: &str) -> Result<Val, String> {
        let b = s.as_bytes();
        let mut p = 0usize;
        let v = Self::value(s, b, &mut p)?;
        Self::ws(b, &mut p);
        if p != b.len() {
            return Err(format!("trailing data at {p}"));
        }
        Ok(v)
    }
    pub fn get(&self, key: &str) -> Option<&Val> {
        match self {
            Val::Obj(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    pub fn str_at(&self, key: &str) -> Option<&str> {
        match self.get(key) {
            Some(Val::Str(s)) => Some(s.as_ref()),
            _ => None,
        }
    }
    fn ws(b: &[u8], p: &mut usize) {
        while *p < b.len() && matches!(b[*p], b' ' | b'\t' | b'\n' | b'\r') {
            *p += 1;
        }
    }
    fn value(s: &str, b: &[u8], p: &mut usize) -> Result<Val, String> {
        Self::ws(b, p);
        match b.get(*p) {
            None => Err("unexpected end".into()),
            Some(b'n') => Self::lit(b, p, "null", Val::Null),
            Some(b't') => Self::lit(b, p, "true", Val::Bool(true)),
            Some(b'f') => Self::lit(b, p, "false", Val::Bool(false)),
            Some(b'"') => Ok(Val::Str(Self::string(s, b, p)?)),
            Some(b'[') => {
                *p += 1;
                let mut out = Vec::new();
                Self::ws(b, p);
                if b.get(*p) == Some(&b']') {
                    *p += 1;
                    return Ok(Val::Arr(out));
                }
                loop {
                    out.push(Self::value(s, b, p)?);
                    Self::ws(b, p);
                    match b.get(*p) {
                        Some(b',') => *p += 1,
                        Some(b']') => {
                            *p += 1;
                            return Ok(Val::Arr(out));
                        }
                        _ => return Err(format!("expected , or ] at {p}")),
                    }
                }
            }
            Some(b'{') => {
                *p += 1;
                let mut out = Vec::new();
                Self::ws(b, p);
                if b.get(*p) == Some(&b'}') {
                    *p += 1;
                    return Ok(Val::Obj(out));
                }
                loop {
                    Self::ws(b, p);
                    let k = Self::string(s, b, p)?;
                    Self::ws(b, p);
                    if b.get(*p) != Some(&b':') {
                        return Err(format!("expected : at {p}"));
                    }
                    *p += 1;
                    let v = Self::value(s, b, p)?;
                    out.push((k, v));
                    Self::ws(b, p);
                    match b.get(*p) {
                        Some(b',') => *p += 1,
                        Some(b'}') => {
                            *p += 1;
                            return Ok(Val::Obj(out));
                        }
                        _ => return Err(format!("expected , or }} at {p}")),
                    }
                }
            }
            Some(_) => {
                let start = *p;
                while *p < b.len() && matches!(b[*p], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') {
                    *p += 1;
                }
                let t = &s[start..*p];
                if t.contains(['.', 'e', 'E']) {
                    t.parse::<f64>().map(Val::Float).map_err(|e| format!("number {t:?}: {e}"))
                } else {
                    t.parse::<i64>().map(Val::Int).map_err(|e| format!("number {t:?}: {e}"))
                }
            }
        }
    }
    fn lit(b: &[u8], p: &mut usize, word: &str, v: Val) -> Result<Val, String> {
        if b[*p..].starts_with(word.as_bytes()) {
            *p += word.len();
            Ok(v)
        } else {
            Err(format!("bad literal at {p}"))
        }
    }
    fn string(s: &str, b: &[u8], p: &mut usize) -> Result<JStr, String> {
        if b.get(*p) != Some(&b'"') {
            return Err(format!("expected string at {p}"));
        }
        *p += 1;
        // no escape before the closing quote (every request string): the text is one run
        if let Some(k) = b[*p..].iter().position(|&c| c == b'"' || c == b'\\') {
            if b[*p + k] == b'"' {
                let t = &s[*p..*p + k];
                *p += k + 1;
                return Ok(intern(t));
            }
        }
        let mut out = String::new();
        loop {
            // The run up to the next `"` or `\\` is copied whole: both are ASCII, so the run ends on
            // a char boundary, and every other char (a control char included) is kept verbatim.
            let k = b[*p..].iter().position(|&c| c == b'"' || c == b'\\').ok_or("unterminated string")?;
            out.push_str(&s[*p..*p + k]);
            *p += k + 1;
            match b[*p - 1] {
                b'"' => return Ok(JStr::Owned(out)),
                _ => {
                    let e = s[*p..].chars().next().ok_or("bad escape")?;
                    *p += 1;
                    match e {
                        '"' => out.push('"'),
                        '\\' => out.push('\\'),
                        '/' => out.push('/'),
                        'n' => out.push('\n'),
                        'r' => out.push('\r'),
                        't' => out.push('\t'),
                        'b' => out.push('\u{8}'),
                        'f' => out.push('\u{c}'),
                        'u' => {
                            let hex = s.get(*p..*p + 4).ok_or("bad \\u escape")?;
                            let code = u32::from_str_radix(hex, 16).map_err(|e| e.to_string())?;
                            *p += 4;
                            out.push(char::from_u32(code).ok_or("bad \\u code point")?);
                        }
                        o => return Err(format!("bad escape \\{o}")),
                    }
                }
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::{JStr, Val};

    /// The char-at-a-time string reader the run-copying one replaced (the reference).
    fn reference_string(s: &str) -> Result<String, String> {
        let mut p = 1usize;
        let mut out = String::new();
        loop {
            let c = s[p..].chars().next().ok_or("unterminated string")?;
            p += c.len_utf8();
            match c {
                '"' => return Ok(out),
                '\\' => {
                    let e = s[p..].chars().next().ok_or("bad escape")?;
                    p += 1;
                    match e {
                        '"' => out.push('"'),
                        '\\' => out.push('\\'),
                        '/' => out.push('/'),
                        'n' => out.push('\n'),
                        'r' => out.push('\r'),
                        't' => out.push('\t'),
                        'b' => out.push('\u{8}'),
                        'f' => out.push('\u{c}'),
                        'u' => {
                            let hex = s.get(p..p + 4).ok_or("bad \\u escape")?;
                            let code = u32::from_str_radix(hex, 16).map_err(|e| e.to_string())?;
                            p += 4;
                            out.push(char::from_u32(code).ok_or("bad \\u code point")?);
                        }
                        o => return Err(format!("bad escape \\{o}")),
                    }
                }
                c => out.push(c),
            }
        }
    }

    #[test]
    fn a_string_reads_exactly_as_the_char_at_a_time_reader_read_it() {
        for lit in [
            r#""""#,
            r#""plain""#,
            r#""p1a: Mr. Mime""#,
            r#""Flabébé ♀ — 100/100 par""#,
            "\"tab\there, newline\nthere\"",
            r#""a\"b\\c\/d\ne\rf\tg\bh\fi""#,
            r#""é—A mixed é \\ end""#,
            r#""\\""#,
            r#""trailing \\\"""#,
            r#""unterminated"#,
            r#""bad \q escape""#,
            r#""bad \u12""#,
            r#""bad \uZZZZ x""#,
            r#""ends in a backslash \"#,
        ] {
            let got = Val::parse(lit);
            let want = reference_string(lit).map(|s| Val::Str(s.into()));
            match (&got, &want) {
                (Ok(g), Ok(w)) => assert_eq!(g, w, "{lit:?}"),
                // every input is one string literal, so the refusal is the reference's, word for word
                (Err(g), Err(w)) => assert_eq!(g, w, "{lit:?}"),
                _ => panic!("{lit:?}: parse {got:?} vs reference {want:?}"),
            }
        }
        let v = Val::parse(r#"{"a":"x\"y","b":["é",{"c":"\\"}],"rqid":3}"#).unwrap();
        assert_eq!(v.str_at("a"), Some("x\"y"));
        assert_eq!(v.get("rqid"), Some(&Val::Int(3)));
        // an interned word reads, compares and prints exactly as its owned twin
        let (a, b) = (Val::parse(r#"{"pokeball":"pokeball"}"#).unwrap(), Val::Obj(vec![("pokeball".to_string().into(), Val::Str("pokeball".to_string().into()))]));
        assert!(matches!(&a, Val::Obj(kv) if matches!(kv[0].0, JStr::Borrowed(_)) && matches!(&kv[0].1, Val::Str(JStr::Borrowed(_)))));
        assert_eq!(a, b);
        assert_eq!(format!("{a:?}"), format!("{b:?}"));
        assert_eq!(a.str_at("pokeball"), Some("pokeball"));
    }
}
