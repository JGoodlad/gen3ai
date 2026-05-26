"""Download Smogon chaos stats over a rolling window of months.

Smogon publishes per-month JSON dumps at
    https://www.smogon.com/stats/YYYY-MM/chaos/<format>.json

For our priors (Hidden Power type distributions, item/ability priors, etc.) a
single month is a noisy sample — usage shifts week-to-week with meta trends.
Pulling the last 12 months and summing the move/item/ability counts yields a
much steadier distribution.

This script:
  1. Iterates `--months` months back from `--end-month`.
  2. Downloads each month's chaos JSON into `--monthly-dir/YYYY-MM_<format>.json`.
     Files already on disk are skipped unless `--force`.
  3. Merges them into a single aggregated file at `--output` with the same
     shape as a single-month chaos JSON, so downstream consumers
     (src/scripts/compute_hidden_power_priors.py and friends) work unchanged.

Aggregation rules:
  - Sum value-counts in the dict-of-numbers fields: Moves, Abilities, Items,
    Spreads, Tera Types, Happiness, Teammates.
  - Sum Raw count.
  - Sum top-level info.number_of_battles.
  - Take Viability Ceiling, Checks and Counters, and usage from the latest
    month present — these aren't safely summable.

A 404 on any given month (the most-recent month might not be posted yet,
or older months may have been pruned) is logged and skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date
from typing import Iterator, Optional

import requests

DEFAULT_FORMAT = "gen3ou-1500"
DEFAULT_OUTPUT = "data/pokemon/gen3_smogon_stats.json"
DEFAULT_MONTHLY_DIR = "data/pokemon/smogon_stats_monthly"
DEFAULT_MONTHS = 12
SMOGON_URL_TEMPLATE = "https://www.smogon.com/stats/{year:04d}-{month:02d}/chaos/{fmt}.json"

# Fields under data[species] that are flat {string: number} dicts and can be
# summed across months element-wise.
_SUMMABLE_FIELDS = ("Moves", "Abilities", "Items", "Spreads", "Tera Types",
                    "Happiness", "Teammates")
# Fields we take from the latest month (not safely summable).
_LATEST_FIELDS = ("Viability Ceiling", "Checks and Counters", "usage")


# ---------------------------------------------------------------------------
# Month iteration
# ---------------------------------------------------------------------------

def _previous_month(year: int, month: int) -> tuple[int, int]:
    if month == 1:
        return year - 1, 12
    return year, month - 1


def _parse_end_month(s: Optional[str]) -> tuple[int, int]:
    """Parse 'YYYY-MM' or default to the previous calendar month from today."""
    if s is None:
        today = date.today()
        return _previous_month(today.year, today.month)
    try:
        y_str, m_str = s.split("-")
        year = int(y_str)
        month = int(m_str)
        if not (1 <= month <= 12) or year < 2000:
            raise ValueError
    except ValueError as exc:
        raise SystemExit(f"--end-month must be YYYY-MM, got {s!r}") from exc
    return year, month


def iter_months_backward(end_year: int, end_month: int, n: int) -> Iterator[tuple[int, int]]:
    """Yield (year, month) tuples starting from (end_year, end_month) going back n months."""
    y, m = end_year, end_month
    for _ in range(n):
        yield y, m
        y, m = _previous_month(y, m)


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def _monthly_path(monthly_dir: str, year: int, month: int, fmt: str) -> str:
    return os.path.join(monthly_dir, f"{year:04d}-{month:02d}_{fmt}.json")


def fetch_month(
    year: int,
    month: int,
    fmt: str,
    monthly_dir: str,
    force: bool,
    timeout: int = 60,
) -> tuple[Optional[dict], bool]:
    """Download one month's chaos JSON.

    Returns (parsed_dict_or_None, came_from_cache). 404 (newest not posted yet,
    or old month pruned) and other failures return (None, False).

    Caches to monthly_dir/YYYY-MM_<fmt>.json. Subsequent calls reuse the cache
    unless force=True.
    """
    path = _monthly_path(monthly_dir, year, month, fmt)
    if os.path.exists(path) and not force:
        with open(path) as f:
            print(f"  {year:04d}-{month:02d}: cached")
            return json.load(f), True

    url = SMOGON_URL_TEMPLATE.format(year=year, month=month, fmt=fmt)
    print(f"  {year:04d}-{month:02d}: fetching ... ", end="", flush=True)
    try:
        response = requests.get(url, timeout=timeout)
    except requests.RequestException as e:
        print(f"NETWORK ERROR ({e})")
        return None, False

    if response.status_code == 404:
        print("not posted (404 — skipping)")
        return None, False
    if not response.ok:
        print(f"HTTP {response.status_code} (skipping)")
        return None, False

    try:
        data = response.json()
    except ValueError as e:
        print(f"BAD JSON ({e}) — skipping")
        return None, False

    if "data" not in data:
        print("missing 'data' key — skipping")
        return None, False

    os.makedirs(monthly_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    n_species = len(data["data"])
    n_battles = data.get("info", {}).get("number of battles", 0) or 0
    print(f"OK ({n_species} species, {n_battles:,} battles)")
    return data, False


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _merge_species(dst: dict, src: dict, is_first_month_for_species: bool) -> None:
    """Merge one month's species record into the accumulator dict, in place."""
    # Raw count sums.
    raw = src.get("Raw count")
    if isinstance(raw, (int, float)):
        dst["Raw count"] = dst.get("Raw count", 0) + raw

    # Flat dict-of-numbers fields sum element-wise.
    for field in _SUMMABLE_FIELDS:
        src_val = src.get(field)
        if not isinstance(src_val, dict):
            continue
        bucket = dst.setdefault(field, {})
        for k, v in src_val.items():
            if isinstance(v, (int, float)):
                bucket[k] = bucket.get(k, 0.0) + float(v)

    # Latest-wins fields — set only when this is the first (newest) month we've
    # seen this species in. Iteration order is newest → oldest.
    if is_first_month_for_species:
        for field in _LATEST_FIELDS:
            if field in src:
                dst[field] = src[field]


def merge_chaos(months: list[tuple[tuple[int, int], dict]], fmt: str) -> dict:
    """Aggregate a list of (year_month, chaos_dict) into one chaos-shaped dict.

    `months` is ordered newest → oldest. Summable fields sum across all
    months; latest-wins fields take the newest month a given species appears in.
    """
    out: dict = {"info": None, "data": {}}
    summed_battles = 0

    for idx, ((y, m), chaos) in enumerate(months):
        info = chaos.get("info", {}) or {}
        if out["info"] is None:
            out["info"] = {
                "metagame": info.get("metagame"),
                "cutoff": info.get("cutoff"),
                "cutoff deviation": info.get("cutoff deviation"),
                "team type": info.get("team type"),
                "number of battles": 0,         # filled in after the loop
                "aggregated": True,
                "window_months": [],
                "format": fmt,
            }
        summed_battles += int(info.get("number of battles", 0) or 0)
        out["info"]["window_months"].append(f"{y:04d}-{m:02d}")

        for species, rec in chaos.get("data", {}).items():
            first_for_species = species not in out["data"]
            dst = out["data"].setdefault(species, {})
            _merge_species(dst, rec, is_first_month_for_species=first_for_species)

    if out["info"] is not None:
        out["info"]["number of battles"] = summed_battles

    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def sync_stats(
    end_month: Optional[str],
    n_months: int,
    fmt: str,
    output_path: str,
    monthly_dir: str,
    force: bool,
    delay_seconds: float,
) -> None:
    end_y, end_m = _parse_end_month(end_month)
    print(f"Smogon stats sync: {n_months} months back from {end_y:04d}-{end_m:02d} "
          f"(format={fmt})")
    print(f"  monthly cache: {monthly_dir}")
    print(f"  merged output: {output_path}")

    fetched: list[tuple[tuple[int, int], dict]] = []
    skipped: list[tuple[int, int]] = []
    months = list(iter_months_backward(end_y, end_m, n_months))
    for i, (y, m) in enumerate(months):
        data, from_cache = fetch_month(y, m, fmt, monthly_dir, force)
        if data is None:
            skipped.append((y, m))
            continue
        fetched.append(((y, m), data))
        # Politeness delay between network hits — skipped on cache hits and on
        # the final month.
        is_last = (i + 1 == len(months))
        if not from_cache and not is_last and delay_seconds > 0:
            time.sleep(delay_seconds)

    if not fetched:
        print("ERROR: no months fetched successfully — aborting.", file=sys.stderr)
        sys.exit(1)

    merged = merge_chaos(fetched, fmt)
    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)

    n_species = len(merged["data"])
    n_battles = merged["info"]["number of battles"]
    window = merged["info"]["window_months"]
    print()
    print(f"Merged {len(fetched)} months covering {window[-1]} .. {window[0]}")
    print(f"  species: {n_species}")
    print(f"  battles: {n_battles:,}")
    if skipped:
        sk = ", ".join(f"{y:04d}-{m:02d}" for y, m in skipped)
        print(f"  skipped months: {sk}")
    print(f"Wrote {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Smogon chaos stats over a rolling window of months "
                    "and produce an aggregated stats file for priors computation.",
    )
    parser.add_argument(
        "--months", type=int, default=DEFAULT_MONTHS,
        help=f"Number of consecutive months to fetch, going backward from --end-month "
             f"(default: {DEFAULT_MONTHS}).",
    )
    parser.add_argument(
        "--end-month", default=None,
        help="Most recent month to include, YYYY-MM. Defaults to the previous "
             "calendar month from today (since the latest month is often not yet posted).",
    )
    parser.add_argument(
        "--format", dest="fmt", default=DEFAULT_FORMAT,
        help=f"Smogon format slug (default: {DEFAULT_FORMAT}).",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Aggregated output path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--monthly-dir", default=DEFAULT_MONTHLY_DIR,
        help=f"Per-month cache directory (default: {DEFAULT_MONTHLY_DIR}).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download monthly files even if they already exist on disk.",
    )
    parser.add_argument(
        "--delay-seconds", type=float, default=1.0,
        help="Polite sleep between network requests (default: 1.0). "
             "Skipped on cache hits.",
    )
    args = parser.parse_args()

    sync_stats(
        end_month=args.end_month,
        n_months=args.months,
        fmt=args.fmt,
        output_path=args.output,
        monthly_dir=args.monthly_dir,
        force=args.force,
        delay_seconds=args.delay_seconds,
    )
