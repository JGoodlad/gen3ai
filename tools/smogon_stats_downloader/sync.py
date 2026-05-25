import argparse
import json
import os
import sys

import requests

DEFAULT_URL = "https://www.smogon.com/stats/2026-03/chaos/gen3ou-1500.json"
DEFAULT_OUTPUT = "data/pokemon/gen3_smogon_stats.json"


def sync_stats(url: str, output_path: str) -> None:
    print(f"Fetching Smogon stats from {url} ...")
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching stats: {e}", file=sys.stderr)
        sys.exit(1)

    data = response.json()
    if "data" not in data:
        print("Error: response JSON missing 'data' key", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    species_count = len(data["data"])
    print(f"Saved {species_count} species to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Smogon chaos stats for Gen3 OU")
    parser.add_argument("--url", default=DEFAULT_URL, help="Smogon chaos JSON URL")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output path for stats JSON")
    args = parser.parse_args()
    sync_stats(args.url, args.output)
