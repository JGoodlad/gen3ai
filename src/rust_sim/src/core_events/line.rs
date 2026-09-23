//! [`Line`] — ONE protocol line as a typed value, whose text is a pure rendering of it.
//!
//! A line is a keyword ([`Kw`]) plus its `|`-separated fields, each TYPED: a mon identifier
//! ([`Ident`]), a side reference, an HP field ([`Hp`]), a `[from]` cause ([`Cause`]), an `[of]`
//! source, a bracket tag, or plain text. [`Line::render`] and [`Line::parse`] are exact inverses on
//! every line the port emits and every line a Showdown server sends a player (gated per corpus
//! line by the core's round-trip tests), so the typed value and the text can never disagree —
//! there is only ONE representation.
//!
//! Classification is deliberately conservative: a field is typed only where the keyword's grammar
//! puts that type (an identifier in an EVENT keyword, an HP at the keyword's HP position, a side
//! reference as a side-condition's first field) AND it re-renders to the same bytes; everything
//! else is [`Field::Text`]. So parsing is LOSSLESS for any text, and a shape the grammar does not
//! know degrades to text rather than being mis-typed.

use std::fmt::Write as _;

use super::schema::Kw;
use super::Route;

/// A mon identifier as the protocol prints it: `p1a: Nick` (on the field) or `p1: Nick`
/// (slot-less — a mon that is not active, Showdown's `Pokemon.toString()` off the field).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ident {
    /// 0 = p1, 1 = p2.
    pub side: u8,
    /// The position letter (`a` in singles); `None` for the slot-less form.
    pub pos: Option<char>,
    /// The on-field name (the NICKNAME — never parsed as a species).
    pub name: String,
}

impl Ident {
    pub fn active(side: usize, name: &str) -> Ident {
        Ident { side: side as u8, pos: Some('a'), name: name.to_string() }
    }
    /// Strict parse of `pN[a-z]: name`; `None` for anything else.
    pub fn parse(s: &str) -> Option<Ident> {
        let b = s.as_bytes();
        if b.len() < 4 || b[0] != b'p' || !(b[1] == b'1' || b[1] == b'2') {
            return None;
        }
        let side = b[1] - b'1';
        let (pos, rest) = if b[2].is_ascii_lowercase() { (Some(b[2] as char), &s[3..]) } else { (None, &s[2..]) };
        let name = rest.strip_prefix(": ")?;
        Some(Ident { side, pos, name: name.to_string() })
    }
    pub fn render_into(&self, out: &mut String) {
        out.push('p');
        out.push((b'1' + self.side) as char);
        if let Some(p) = self.pos {
            out.push(p);
        }
        out.push_str(": ");
        out.push_str(&self.name);
    }
    pub fn render(&self) -> String {
        let mut s = String::new();
        self.render_into(&mut s);
        s
    }
}

/// An HP field: `0 fnt`, `224/341`, or `116/524 slp`. The viewer's rendering — exact integers on
/// the omniscient stream and for the owner; `pct/100` for the other side in a percent format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Hp {
    Fainted,
    Alive { hp: u32, max: u32, status: Option<String> },
}

impl Hp {
    pub fn from_status(hp: &crate::protocol::HpStatus) -> Hp {
        if hp.hp == 0 {
            Hp::Fainted
        } else {
            Hp::Alive { hp: hp.hp as u32, max: hp.maxhp as u32, status: hp.status.map(str::to_string) }
        }
    }
    fn parse(s: &str) -> Option<Hp> {
        if s == "0 fnt" {
            return Some(Hp::Fainted);
        }
        let (frac, status) = match s.split_once(' ') {
            Some((f, st)) if !st.is_empty() && !st.contains(' ') => (f, Some(st.to_string())),
            Some(_) => return None,
            None => (s, None),
        };
        let (a, b) = frac.split_once('/')?;
        let hp: u32 = a.parse().ok()?;
        let max: u32 = b.parse().ok()?;
        let out = Hp::Alive { hp, max, status };
        (out.render() == s).then_some(out)
    }
    pub fn render_into(&self, out: &mut String) {
        match self {
            Hp::Fainted => out.push_str("0 fnt"),
            Hp::Alive { hp, max, status } => {
                let _ = write!(out, "{hp}/{max}");
                if let Some(st) = status {
                    out.push(' ');
                    out.push_str(st);
                }
            }
        }
    }
    pub fn render(&self) -> String {
        let mut s = String::new();
        self.render_into(&mut s);
        s
    }
}

/// The text of a `[from] …` clause, typed by its prefix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Cause {
    Item(String),
    Ability(String),
    Move(String),
    /// A bare cause (`Sandstorm`, `psn`, `Recoil`, `lockedmove`, a called move's `Sleep Talk`).
    Bare(String),
}

impl Cause {
    /// The CANONICAL typing of a `[from]` text (the one [`Line::parse`] produces).
    pub fn from_text(s: &str) -> Cause {
        if let Some(r) = s.strip_prefix("item: ") {
            Cause::Item(r.to_string())
        } else if let Some(r) = s.strip_prefix("ability: ") {
            Cause::Ability(r.to_string())
        } else if let Some(r) = s.strip_prefix("move: ") {
            Cause::Move(r.to_string())
        } else {
            Cause::Bare(s.to_string())
        }
    }
    pub fn from_protocol(c: &crate::protocol::Cause) -> Cause {
        use crate::protocol::Cause as P;
        match c {
            P::Item(s) => Cause::Item(s.clone()),
            P::Ability(s) => Cause::Ability(s.clone()),
            P::Move(s) => Cause::Move(s.clone()),
            P::Bare(s) => Cause::from_text(s),
        }
    }
    /// The clause text WITHOUT `[from] ` — the string `Gen3Battle._parse_from` stores.
    pub fn text(&self) -> String {
        match self {
            Cause::Item(s) => format!("item: {s}"),
            Cause::Ability(s) => format!("ability: {s}"),
            Cause::Move(s) => format!("move: {s}"),
            Cause::Bare(s) => s.clone(),
        }
    }
}

/// One typed field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Field {
    Ident(Ident),
    /// `pN: PlayerName` — the first field of a side-condition line.
    SideRef { side: u8, name: String },
    Hp(Hp),
    From(Cause),
    Of(Ident),
    /// `[wisher] Name` (the Wish heal's caster).
    Wisher(String),
    /// A bracket flag, verbatim (`[still]`, `[miss]`, `[silent]`, `[msg]`, `[upkeep]`, …).
    Tag(String),
    Text(String),
    Empty,
}

impl Field {
    pub fn text(s: impl Into<String>) -> Field {
        Field::Text(s.into())
    }
    pub fn tag(s: &str) -> Field {
        Field::Tag(s.to_string())
    }
    pub fn mon(m: &crate::protocol::MonRef) -> Field {
        Field::Ident(Ident::active(m.side, &m.name))
    }
    pub fn of(m: &crate::protocol::MonRef) -> Field {
        Field::Of(Ident::active(m.side, &m.name))
    }
    pub fn hp(h: &crate::protocol::HpStatus) -> Field {
        Field::Hp(Hp::from_status(h))
    }
    pub fn from(c: &crate::protocol::Cause) -> Field {
        Field::From(Cause::from_protocol(c))
    }
    /// `[from] <text>`, typed canonically.
    pub fn from_text(s: &str) -> Field {
        Field::From(Cause::from_text(s))
    }
    pub fn render_into(&self, out: &mut String) {
        match self {
            Field::Ident(i) => i.render_into(out),
            Field::SideRef { side, name } => {
                let _ = write!(out, "p{}: {name}", side + 1);
            }
            Field::Hp(h) => h.render_into(out),
            Field::From(c) => {
                out.push_str("[from] ");
                out.push_str(&c.text());
            }
            Field::Of(i) => {
                out.push_str("[of] ");
                i.render_into(out);
            }
            Field::Wisher(n) => {
                out.push_str("[wisher] ");
                out.push_str(n);
            }
            Field::Tag(t) | Field::Text(t) => out.push_str(t),
            Field::Empty => {}
        }
    }
    /// The field's text (the `split('|')` token poke-env sees).
    pub fn render(&self) -> String {
        let mut s = String::new();
        self.render_into(&mut s);
        s
    }
}

/// One protocol line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Line {
    pub kw: Kw,
    pub fields: Vec<Field>,
}

/// Why a text line could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LineError {
    /// A keyword the schema does not know — the parser REFUSES (a guessed line is worse than a
    /// lost game; `program_rust_core.md` §6b).
    UnknownKeyword(String),
}

impl std::fmt::Display for LineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LineError::UnknownKeyword(k) => write!(f, "unknown protocol keyword {k:?}"),
        }
    }
}

/// Keywords whose first field is a SIDE reference, not a mon.
fn side_ref_kw(kw: Kw) -> bool {
    matches!(kw, Kw::Sidestart | Kw::Sideend | Kw::Swapsideconditions)
}

/// The field index (0-based, after the keyword) of the HP field, if the keyword has one.
fn hp_index(kw: Kw) -> Option<usize> {
    match kw {
        Kw::Switch | Kw::Drag => Some(2),
        Kw::Damage | Kw::Heal | Kw::Sethp => Some(1),
        _ => None,
    }
}

/// Whether identifiers / tags / causes are typed in this keyword's fields. The EVENT keywords
/// and the handful of cosmetic lines that name a mon; every other keyword is free text.
fn typed_fields(kw: Kw) -> bool {
    matches!(kw.route(), Route::Event(_))
        || matches!(kw, Kw::Anim | Kw::Hitcount | Kw::Block | Kw::Waiting | Kw::Ohko | Kw::Combine)
}

fn classify(kw: Kw, idx: usize, tok: &str) -> Field {
    if tok.is_empty() {
        return Field::Empty;
    }
    if !typed_fields(kw) {
        return Field::Text(tok.to_string());
    }
    if idx == 0 && side_ref_kw(kw) {
        let b = tok.as_bytes();
        if b.len() >= 4 && b[0] == b'p' && (b[1] == b'1' || b[1] == b'2') && &tok[2..4] == ": " {
            return Field::SideRef { side: b[1] - b'1', name: tok[4..].to_string() };
        }
        return Field::Text(tok.to_string());
    }
    if hp_index(kw) == Some(idx) {
        if let Some(h) = Hp::parse(tok) {
            return Field::Hp(h);
        }
        return Field::Text(tok.to_string());
    }
    if let Some(r) = tok.strip_prefix("[from] ") {
        return Field::From(Cause::from_text(r));
    }
    if let Some(r) = tok.strip_prefix("[of] ") {
        return match Ident::parse(r) {
            Some(i) if i.render() == r => Field::Of(i),
            _ => Field::Text(tok.to_string()),
        };
    }
    if let Some(r) = tok.strip_prefix("[wisher] ") {
        return Field::Wisher(r.to_string());
    }
    if tok.starts_with('[') && tok.ends_with(']') {
        return Field::Tag(tok.to_string());
    }
    if let Some(i) = Ident::parse(tok) {
        return Field::Ident(i);
    }
    Field::Text(tok.to_string())
}

impl Line {
    pub fn new(kw: Kw, fields: Vec<Field>) -> Line {
        Line { kw, fields }
    }

    /// Parse one line of protocol text. Lossless: `render(parse(s)) == s` for every `s` that
    /// parses (a pinned property).
    pub fn parse(s: &str) -> Result<Line, LineError> {
        let Some(body) = s.strip_prefix('|') else {
            return Ok(Line { kw: Kw::Plain, fields: vec![Field::Text(s.to_string())] });
        };
        let mut it = body.split('|');
        let kw_text = it.next().unwrap_or("");
        let kw = Kw::from_protocol(kw_text).ok_or_else(|| LineError::UnknownKeyword(kw_text.to_string()))?;
        let fields = it.enumerate().map(|(i, t)| classify(kw, i, t)).collect();
        Ok(Line { kw, fields })
    }

    pub fn render_into(&self, out: &mut String) {
        if self.kw == Kw::Plain {
            if let Some(f) = self.fields.first() {
                f.render_into(out);
            }
            return;
        }
        out.push('|');
        out.push_str(self.kw.as_str());
        for f in &self.fields {
            out.push('|');
            f.render_into(out);
        }
    }

    /// The protocol text.
    pub fn render(&self) -> String {
        let mut s = String::with_capacity(32);
        self.render_into(&mut s);
        s
    }

    /// `render().split('|')` — exactly the `split_message` poke-env builds.
    pub fn split_message(&self) -> Vec<String> {
        if self.kw == Kw::Plain {
            return vec![self.render()];
        }
        let mut v = Vec::with_capacity(self.fields.len() + 2);
        v.push(String::new());
        v.push(self.kw.as_str().to_string());
        for f in &self.fields {
            v.push(f.render());
        }
        v
    }

    /// The field at `idx`, if present.
    pub fn field(&self, idx: usize) -> Option<&Field> {
        self.fields.get(idx)
    }
    /// The identifier at `idx`, if that field is one.
    pub fn ident(&self, idx: usize) -> Option<&Ident> {
        match self.fields.get(idx) {
            Some(Field::Ident(i)) => Some(i),
            _ => None,
        }
    }
    /// The LAST `[from]` cause (`Gen3Battle._parse_from`'s last-token-wins rule).
    pub fn from_cause(&self) -> Option<&Cause> {
        self.fields.iter().rev().find_map(|f| match f {
            Field::From(c) => Some(c),
            _ => None,
        })
    }
    /// The LAST `[of]` source.
    pub fn of_source(&self) -> Option<&Ident> {
        self.fields.iter().rev().find_map(|f| match f {
            Field::Of(i) => Some(i),
            _ => None,
        })
    }
    pub fn has_tag(&self, tag: &str) -> bool {
        self.fields.iter().any(|f| matches!(f, Field::Tag(t) if t == tag))
    }
    /// The HP field, if the keyword has one and it is typed.
    pub fn hp(&self) -> Option<&Hp> {
        match hp_index(self.kw).and_then(|i| self.fields.get(i)) {
            Some(Field::Hp(h)) => Some(h),
            _ => None,
        }
    }
    /// The identifier that OWNS this line for the per-side privacy split (`bridge::ident_owner`):
    /// the side of a `pN…:` first field.
    pub fn owner_side(&self) -> Option<u8> {
        match self.fields.first() {
            Some(Field::Ident(i)) => Some(i.side),
            Some(Field::SideRef { side, .. }) => Some(*side),
            Some(Field::Text(t)) => {
                let b = t.as_bytes();
                (b.len() >= 3 && b[0] == b'p' && (b[1] == b'1' || b[1] == b'2') && (b[2] == b'a' || b[2] == b':'))
                    .then(|| b[1] - b'1')
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rt(s: &str) -> Line {
        let l = Line::parse(s).expect("parse");
        assert_eq!(l.render(), s, "render(parse(s)) must be s");
        assert_eq!(l.split_message(), s.split('|').map(str::to_string).collect::<Vec<_>>());
        l
    }

    #[test]
    fn every_shape_round_trips_and_types_its_fields() {
        let l = rt("|move|p1a: Mr. Mime|Psychic|p2a: Blissey|[miss]");
        assert_eq!(l.kw, Kw::Move);
        assert_eq!(l.ident(0).unwrap().name, "Mr. Mime");
        assert!(l.has_tag("[miss]"));
        let l = rt("|move|p1a: Skarmory|Protect||[still]");
        assert_eq!(l.fields[2], Field::Empty);
        let l = rt("|-damage|p2a: Snorlax|116/524 slp|[from] Leech Seed|[of] p1a: Venusaur");
        assert_eq!(l.hp(), Some(&Hp::Alive { hp: 116, max: 524, status: Some("slp".into()) }));
        assert_eq!(l.from_cause(), Some(&Cause::Bare("Leech Seed".into())));
        assert_eq!(l.of_source().unwrap().side, 0);
        let l = rt("|switch|p1a: Electhor|Zapdos, M|0 fnt");
        assert_eq!(l.hp(), Some(&Hp::Fainted));
        let l = rt("|-sidestart|p2: Some Player|Spikes");
        assert!(matches!(l.fields[0], Field::SideRef { side: 1, .. }));
        let l = rt("|-curestatus|p1: Blissey|slp|[silent]");
        assert_eq!(l.ident(0).unwrap().pos, None);
        rt("|");
        rt("|upkeep");
        rt("|player|p1|P1||");
        rt("|t:|<NORMALIZED>");
        rt("|request|{\"active\":[{\"moves\":[]}],\"side\":{\"name\":\"p1: x\"}}");
        rt("|-activate|p1a: Gengar|Skill Swap|||[of] p2a: Blissey");
        rt("|-heal|p1a: Blissey|524/524|[from] move: Wish|[wisher] Jirachi");
        rt("|-immune|p1a: Gengar|confusion|[from] ability: Own Tempo");
        rt("|cant|p1a: Quagsire|ability: Damp|Self-Destruct|[of] p2a: Snorlax");
        rt("|bigerror|You will auto-tie if the battle doesn't end in 1 turn (on turn 1000).");
        rt("|-hint|In Gen 3, Intimidate does not activate if every target has a Substitute.");
        let l = rt("not a protocol line");
        assert_eq!(l.kw, Kw::Plain);
        // A non-canonical HP degrades to text rather than being mis-typed.
        let l = rt("|-damage|p1a: X|007/100");
        assert!(matches!(l.fields[1], Field::Text(_)));
    }

    #[test]
    fn an_unknown_keyword_is_refused() {
        assert_eq!(Line::parse("|-dynamax|p1a: X"), Err(LineError::UnknownKeyword("-dynamax".into())));
    }
}
