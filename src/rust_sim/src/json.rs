//! A minimal, std-only JSON value parser.
//!
//! The crate is deliberately dependency-free (see `CLAUDE.md`), so rather than
//! pull in serde we parse our own `data/pokemon/*.json` with this small
//! recursive-descent reader. It is used only at load time (the [`crate::dex`]),
//! never in the deterministic battle path, so parse order can't affect anything.
//!
//! Scope: the standard JSON grammar (objects, arrays, strings with `\uXXXX`
//! escapes, numbers, `true`/`false`/`null`). Numbers are kept as `f64`, which is
//! exact for every integer our data files hold (dex nums, base stats, BP, …).

use std::collections::HashMap;

/// A parsed JSON value.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(HashMap<String, Json>),
}

impl Json {
    /// Parse a complete JSON document. Returns an error message on malformed input.
    pub fn parse(input: &str) -> Result<Json, String> {
        let mut p = Parser {
            chars: input.chars().collect(),
            pos: 0,
        };
        p.skip_ws();
        let v = p.parse_value()?;
        p.skip_ws();
        if p.pos != p.chars.len() {
            return Err(format!("trailing data at char {}", p.pos));
        }
        Ok(v)
    }

    pub fn as_object(&self) -> Option<&HashMap<String, Json>> {
        match self {
            Json::Obj(m) => Some(m),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(a) => Some(a),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(n) => Some(*n),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }
    pub fn is_null(&self) -> bool {
        matches!(self, Json::Null)
    }

    /// Object-field lookup (`None` if not an object or key absent).
    pub fn get(&self, key: &str) -> Option<&Json> {
        self.as_object().and_then(|m| m.get(key))
    }

    // --- convenience coercions used by the dex (lenient: wrong-type/absent -> default) ---

    /// A field read as an integer (rounded), or `default`.
    pub fn int_or(&self, key: &str, default: i64) -> i64 {
        self.get(key).and_then(Json::as_f64).map_or(default, |n| n as i64)
    }
    /// A field read as `f64`, or `default`.
    pub fn f64_or(&self, key: &str, default: f64) -> f64 {
        self.get(key).and_then(Json::as_f64).unwrap_or(default)
    }
    /// A field read as a bool, or `default`.
    pub fn bool_or(&self, key: &str, default: bool) -> bool {
        self.get(key).and_then(Json::as_bool).unwrap_or(default)
    }
    /// A field read as a string slice, or `None` (also `None` for JSON `null`).
    pub fn str_at(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(Json::as_str)
    }
}

struct Parser {
    chars: Vec<char>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }
    fn bump(&mut self) -> Option<char> {
        let c = self.peek();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }
    fn skip_ws(&mut self) {
        while let Some(c) = self.peek() {
            if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }
    fn expect(&mut self, c: char) -> Result<(), String> {
        match self.bump() {
            Some(got) if got == c => Ok(()),
            other => Err(format!("expected {c:?} at char {}, found {other:?}", self.pos)),
        }
    }

    fn parse_value(&mut self) -> Result<Json, String> {
        self.skip_ws();
        match self.peek() {
            Some('{') => self.parse_object(),
            Some('[') => self.parse_array(),
            Some('"') => Ok(Json::Str(self.parse_string()?)),
            Some('t') | Some('f') => self.parse_bool(),
            Some('n') => self.parse_null(),
            Some(c) if c == '-' || c.is_ascii_digit() => self.parse_number(),
            other => Err(format!("unexpected {other:?} at char {}", self.pos)),
        }
    }

    fn parse_object(&mut self) -> Result<Json, String> {
        self.expect('{')?;
        let mut map = HashMap::new();
        self.skip_ws();
        if self.peek() == Some('}') {
            self.pos += 1;
            return Ok(Json::Obj(map));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.skip_ws();
            self.expect(':')?;
            let val = self.parse_value()?;
            map.insert(key, val);
            self.skip_ws();
            match self.bump() {
                Some(',') => continue,
                Some('}') => break,
                other => return Err(format!("expected ',' or '}}' at char {}, found {other:?}", self.pos)),
            }
        }
        Ok(Json::Obj(map))
    }

    fn parse_array(&mut self) -> Result<Json, String> {
        self.expect('[')?;
        let mut arr = Vec::new();
        self.skip_ws();
        if self.peek() == Some(']') {
            self.pos += 1;
            return Ok(Json::Arr(arr));
        }
        loop {
            let val = self.parse_value()?;
            arr.push(val);
            self.skip_ws();
            match self.bump() {
                Some(',') => continue,
                Some(']') => break,
                other => return Err(format!("expected ',' or ']' at char {}, found {other:?}", self.pos)),
            }
        }
        Ok(Json::Arr(arr))
    }

    fn parse_string(&mut self) -> Result<String, String> {
        self.expect('"')?;
        let mut s = String::new();
        loop {
            match self.bump() {
                None => return Err("unterminated string".to_string()),
                Some('"') => break,
                Some('\\') => match self.bump() {
                    Some('"') => s.push('"'),
                    Some('\\') => s.push('\\'),
                    Some('/') => s.push('/'),
                    Some('b') => s.push('\u{0008}'),
                    Some('f') => s.push('\u{000C}'),
                    Some('n') => s.push('\n'),
                    Some('r') => s.push('\r'),
                    Some('t') => s.push('\t'),
                    Some('u') => s.push(self.parse_unicode_escape()?),
                    other => return Err(format!("bad escape \\{other:?}")),
                },
                Some(c) => s.push(c),
            }
        }
        Ok(s)
    }

    fn parse_unicode_escape(&mut self) -> Result<char, String> {
        let cp = self.read_hex4()?;
        // Handle a UTF-16 surrogate pair if present.
        if (0xD800..=0xDBFF).contains(&cp) {
            if self.peek() == Some('\\') {
                self.pos += 1;
                self.expect('u')?;
                let lo = self.read_hex4()?;
                let combined = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                return char::from_u32(combined).ok_or_else(|| "bad surrogate pair".to_string());
            }
            return Err("lone high surrogate".to_string());
        }
        char::from_u32(cp).ok_or_else(|| "bad unicode escape".to_string())
    }

    fn read_hex4(&mut self) -> Result<u32, String> {
        let mut v = 0u32;
        for _ in 0..4 {
            let c = self.bump().ok_or("truncated \\u escape")?;
            let d = c.to_digit(16).ok_or_else(|| format!("bad hex digit {c:?}"))?;
            v = v * 16 + d;
        }
        Ok(v)
    }

    fn parse_number(&mut self) -> Result<Json, String> {
        let start = self.pos;
        if self.peek() == Some('-') {
            self.pos += 1;
        }
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-' {
                self.pos += 1;
            } else {
                break;
            }
        }
        let text: String = self.chars[start..self.pos].iter().collect();
        text.parse::<f64>()
            .map(Json::Num)
            .map_err(|_| format!("bad number {text:?}"))
    }

    fn parse_bool(&mut self) -> Result<Json, String> {
        if self.take_literal("true") {
            Ok(Json::Bool(true))
        } else if self.take_literal("false") {
            Ok(Json::Bool(false))
        } else {
            Err(format!("bad literal at char {}", self.pos))
        }
    }

    fn parse_null(&mut self) -> Result<Json, String> {
        if self.take_literal("null") {
            Ok(Json::Null)
        } else {
            Err(format!("bad literal at char {}", self.pos))
        }
    }

    fn take_literal(&mut self, lit: &str) -> bool {
        let lit: Vec<char> = lit.chars().collect();
        if self.pos + lit.len() <= self.chars.len() && self.chars[self.pos..self.pos + lit.len()] == lit[..] {
            self.pos += lit.len();
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Json;

    #[test]
    fn parses_nested() {
        let j = Json::parse(r#"{"a": [1, 2.5, true, null, "hi\n"], "b": {"c": -3}}"#).unwrap();
        assert_eq!(j.get("a").unwrap().as_array().unwrap().len(), 5);
        assert_eq!(j.get("a").unwrap().as_array().unwrap()[1].as_f64(), Some(2.5));
        assert_eq!(j.get("b").unwrap().int_or("c", 0), -3);
        assert_eq!(j.get("a").unwrap().as_array().unwrap()[4].as_str(), Some("hi\n"));
    }

    #[test]
    fn empty_containers_and_escapes() {
        assert!(matches!(Json::parse("{}").unwrap(), Json::Obj(_)));
        assert!(matches!(Json::parse("[]").unwrap(), Json::Arr(_)));
        assert_eq!(Json::parse(r#""a\"bA""#).unwrap().as_str(), Some("a\"bA"));
    }

    #[test]
    fn rejects_trailing_garbage() {
        assert!(Json::parse("{} x").is_err());
    }
}
