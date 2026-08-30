import hashlib
import os
import requests
import json
import re
import argparse
import sys
from typing import List, Dict, Any

# Add the project root to sys.path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.utils.bridge.team_validator import validate_teams_locally

DEFAULT_URL = "https://pokepast.es/aed6c2ad0c5c2593"
DEFAULT_OUTPUT_DIR = "data/teams/others/yak_attack"


def _strip_mon_prefix(name: str) -> str:
    """Some PokePaste dumps (e.g. Yak Attack) name a team once PER POKÉMON as
    ``"<Mon>/<Team Name>"`` — six headers, one shared 6-mon team text. When we collapse those
    duplicate headers into a single entry we strip the leading ``"<Mon>/"`` so the kept name is
    the real team name (``"Cloy Aero"``, not ``"Aerodactyl/Cloy Aero"``)."""
    return name.split("/", 1)[1].strip() if "/" in name else name


def collapse_duplicate_teams(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse manifest rows that point at the SAME team into ONE row — the
    one-entry-per-team contract every other source already satisfies.

    A dump that lists a team once per Pokémon (the Yak Attack bug) otherwise inflates that team's
    uniform training/eval draw weight 6–12×: ``team_loader`` appends the file text once per
    ENTRY, so 174 real teams became 1056 pool slots and ~66% of every episode's team draw. Keyed
    by ``id`` (== the team-text hash == the on-disk filename), so the collapse is exact regardless
    of how the dump was authored. Order-preserving; the first occurrence wins for
    ``file``/``format``/``id``/``source``. ``valid`` = AND over the group (a team is kept only if
    every one of its headers validated — conservative; in the real data validity never differs
    WITHIN a group, so this also equals "any"). ``errors`` = order-preserving union. Idempotent:
    an already-collapsed manifest passes through unchanged."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for row in rows:
        rid = row["id"]
        if rid not in groups:
            groups[rid] = []
            order.append(rid)
        groups[rid].append(row)

    collapsed: List[Dict[str, Any]] = []
    for rid in order:
        grp = groups[rid]
        first = grp[0]
        errors: List[str] = []
        for row in grp:
            for err in row.get("errors", []) or []:
                if err not in errors:
                    errors.append(err)
        collapsed.append({
            "id": rid,
            # strip the per-mon prefix only when the team recurred under multiple headers, so a
            # normal single-occurrence team whose name legitimately contains "/" is kept verbatim
            "name": _strip_mon_prefix(first["name"]) if len(grp) > 1 else first["name"],
            "format": first.get("format"),
            "valid": all(row.get("valid", False) for row in grp),
            "errors": errors,
            "file": first["file"],
            "source": first.get("source"),
        })
    return collapsed


def sync_dump(urls: List[str], output_dir: str, format_id: str = "gen3ou", validate: bool = True, append: bool = False):
    """Fetches, parses, and validates PokePaste team dumps."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    metadata_path = os.path.join(output_dir, "teams.json")
    all_teams_metadata = []
    
    if append and os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                all_teams_metadata = json.load(f)
            print(f"Appending to existing metadata with {len(all_teams_metadata)} teams.")
        except Exception as e:
            print(f"Warning: Could not load existing metadata: {e}")

    # Track existing IDs to avoid duplicates if appending
    existing_ids = {team["id"] for team in all_teams_metadata}
    
    for url in urls:
        raw_url = url.rstrip('/') + "/raw"
        print(f"\n--- Fetching dump from {raw_url} ---")
        try:
            res = requests.get(raw_url, timeout=60)
            if res.status_code != 200:
                print(f"Failed to fetch dump: {res.status_code}")
                continue
        except Exception as e:
            print(f"Error fetching dump: {e}")
            continue
        
        # 🚨 requests falls back to ISO-8859-1 for a text/* response with no charset (RFC 2616),
        # so `res.text` on a UTF-8 paste decodes `é` as `Ã©` — and the file is then WRITTEN back as
        # UTF-8, baking the mojibake in permanently. That is not hypothetical: the committed
        # `data/teams/others/mcmegan/*.txt` hold `PtÃ©ra` where `Ptéra` was meant, which surfaced
        # years later as a `KeyError: 'ptãra'` in a search replay and was chased as a transport
        # bug (`gen3_search_depth2_chunk_gap_v1`). A pastebin raw dump is UTF-8; say so.
        res.encoding = res.encoding or "utf-8"
        if (res.encoding or "").lower() in ("iso-8859-1", "latin-1", "latin1"):
            res.encoding = res.apparent_encoding or "utf-8"
        content = res.text
        
        pattern = r"===\s*(?:\[(.*?)\])?\s*(.*?)\s*===\s*\n(.*?)(?=\n===\s*(?:\[.*?\])?\s*.*?\s*===|$)"
        matches = list(re.finditer(pattern, content, re.DOTALL))
        
        if not matches:
            if content.strip():
                 matches = [(None, "Unnamed Team", content.strip())]
            else:
                print("Dump is empty.")
                continue

        print(f"Processing {len(matches)} potential teams...")
        
        teams_to_validate = []
        team_data = []
        
        for match in matches:
            if isinstance(match, tuple):
                team_format, full_name, team_text = match
            else:
                team_format = match.group(1)
                full_name = match.group(2).strip()
                team_text = match.group(3).strip()
            
            if not team_text:
                continue

            current_format = team_format if team_format else format_id
            team_id = hashlib.sha256(team_text.encode()).hexdigest()[:16]
            
            if team_id in existing_ids:
                # print(f"Skipping duplicate team: {full_name} ({team_id})")
                continue
                
            team_data.append({
                "id": team_id,
                "name": full_name,
                "format": current_format,
                "text": team_text
            })
            teams_to_validate.append(team_text)

        if not teams_to_validate:
            print("No new unique teams found in this dump.")
            continue

        # Perform batch validation
        validation_results = []
        if validate:
            print(f"Validating {len(teams_to_validate)} new teams in batch...")
            validation_results = validate_teams_locally(format_id, teams_to_validate)
        else:
            validation_results = [{"valid": True, "errors": []} for _ in teams_to_validate]

        valid_count = 0
        invalid_count = 0
        
        for i, data in enumerate(team_data):
            team_id = data["id"]
            full_name = data["name"]
            team_text = data["text"]
            current_format = data["format"]
            
            res = validation_results[i]
            is_valid = res.get("valid", False)
            validation_errors = res.get("errors", [])
            
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
                # print(f"Invalid team: {full_name} ({team_id}) - {validation_errors[0] if validation_errors else 'Unknown error'}")

            filename = f"{team_id}.txt"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w') as f:
                f.write(team_text)
                
            all_teams_metadata.append({
                "id": team_id,
                "name": full_name,
                "format": current_format,
                "valid": is_valid,
                "errors": validation_errors,
                "file": os.path.relpath(filepath, "data"),
                "source": url
            })
            existing_ids.add(team_id)
        
        print(f"Synced {len(team_data)} new teams. {valid_count} valid, {invalid_count} invalid")

    # ROOT FIX: a dump that names a team once per Pokémon (the Yak Attack format) yields one
    # manifest row PER MON pointing at the SAME file. The team loader appends the file once per
    # row, so those teams get 6–12× the uniform draw weight of every one-row source. Collapse to
    # one entry per distinct team (== per file) so the manifest can't reproduce that inflation.
    n_rows = len(all_teams_metadata)
    all_teams_metadata = collapse_duplicate_teams(all_teams_metadata)
    if len(all_teams_metadata) != n_rows:
        print(f"Collapsed {n_rows} per-header rows -> {len(all_teams_metadata)} distinct teams "
              f"(one entry per team; prevents per-Pokémon draw-weight inflation)")

    with open(metadata_path, 'w') as f:
        json.dump(all_teams_metadata, f, indent=2)

    print(f"\nFinal Summary: {len(all_teams_metadata)} total teams in {output_dir}")
    print(f"Metadata index saved to {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and validate teams from PokePaste")
    parser.add_argument("--url", type=str, nargs="+", default=[DEFAULT_URL], help="PokePaste URL(s)")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--format", type=str, default="gen3ou", help="Default format if not specified in dump")
    parser.add_argument("--no-validate", action="store_true", help="Skip team validation")
    parser.add_argument("--append", action="store_true", help="Append to existing metadata instead of overwriting")
    
    args = parser.parse_args()
    
    sync_dump(args.url, args.output, args.format, not args.no_validate, args.append)

