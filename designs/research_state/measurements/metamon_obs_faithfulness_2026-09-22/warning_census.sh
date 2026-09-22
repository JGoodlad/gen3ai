#!/usr/bin/env bash
# Count, by CLASS, every way Metamon could tell us it did not understand our stream.
# A class with 0 hits is reported as 0 — an absent grep is not the same as a clean grep.
set -u
declare -A PAT=(
  [unparsed_line]='unparsed|could not parse|cannot parse|unhandled|Unhandled|unknown message|NotImplemented|NotImplementedError'
  [unknown_species]='UnknownPokemon|unknown pokemon|Unknown species|not exist|KeyError'
  [unknown_move]='unknown move|Unknown move|invalid move|no move named'
  [unknown_item_ability]='unknown item|unknown ability|Unknown item|Unknown ability'
  [fallback_tokenize]='Adding: `|UNKNOWN_TOKEN|unknown token'
  [exception]='Traceback|Exception|Error:|RecursionError|AssertionError'
  [warning]='WARNING|Warning:|warn\('
  [invalid_action]='invalid|Invalid|choose_random_move'
)
for f in "$@"; do
  echo "== $f"
  for k in "${!PAT[@]}"; do
    n=$(grep -Ec "${PAT[$k]}" "$f" || true)
    printf '   %-20s %s\n' "$k" "$n"
  done
done
