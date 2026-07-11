//! Type-effectiveness chart, parsed from `data/pokemon/gen3_type_chart.json`.
//! Keyed `{DEFENDING: {ATTACKING: multiplier}}`, mirroring
//! `agents.gen3_data.type_chart`.

use super::types::Type;
use crate::json::Json;
use std::collections::HashMap;

#[derive(Debug, Clone, Default)]
pub struct TypeChart {
    // (defending, attacking) -> multiplier
    table: HashMap<(Type, Type), f64>,
}

impl TypeChart {
    pub(super) fn parse(root: &Json) -> Result<TypeChart, String> {
        let obj = root.as_object().ok_or("type_chart: expected a JSON object")?;
        let mut table = HashMap::new();
        for (def_name, row) in obj {
            let Some(def_t) = Type::from_name(def_name) else {
                continue;
            };
            let row = row.as_object().ok_or("type_chart row: expected a JSON object")?;
            for (att_name, mult) in row {
                if let (Some(att_t), Some(m)) = (Type::from_name(att_name), mult.as_f64()) {
                    table.insert((def_t, att_t), m);
                }
            }
        }
        Ok(TypeChart { table })
    }

    /// Effectiveness of `attacking` vs a single `defending` type (default 1.0 if
    /// the pair isn't in the chart).
    pub fn multiplier(&self, defending: Type, attacking: Type) -> f64 {
        self.table.get(&(defending, attacking)).copied().unwrap_or(1.0)
    }

    /// Combined effectiveness of `attacking` vs a mono/dual `defending` type set
    /// (the product of per-type multipliers — 0× if any type is immune).
    pub fn effectiveness(&self, attacking: Type, defending: &[Type]) -> f64 {
        defending.iter().map(|d| self.multiplier(*d, attacking)).product()
    }
}
