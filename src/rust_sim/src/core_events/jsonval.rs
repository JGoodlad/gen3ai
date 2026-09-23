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
    Str(String),
    Arr(Vec<Val>),
    Obj(Vec<(String, Val)>),
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
            Some(Val::Str(s)) => Some(s),
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
    fn string(s: &str, b: &[u8], p: &mut usize) -> Result<String, String> {
        if b.get(*p) != Some(&b'"') {
            return Err(format!("expected string at {p}"));
        }
        *p += 1;
        let mut out = String::new();
        loop {
            let c = s[*p..].chars().next().ok_or("unterminated string")?;
            *p += c.len_utf8();
            match c {
                '"' => return Ok(out),
                '\\' => {
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
                c => out.push(c),
            }
        }
    }
}

