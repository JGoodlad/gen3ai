"""THE OFFER FILTER for the cloud-GPU benchmark (pure: no network; unit-tested in ``offers_test.py``).

The criteria (owner 2026-10-12 + the benchmark's needs; the reasoning is ``designs/ops/cloud_gpu_benchmark.md``
"Offer criteria"):

- **NO CHINA** (owner: "don't use a 5090 from a china provider, all other countries are fine"). An offer whose
  ``geolocation`` names China / CN, or whose ``geolocode`` is CN, is EXCLUDED. A MISSING geolocation is excluded
  too — an unknown country cannot be shown not to be China (a refusal, not a guess).
- one GPU of the asked model, verified, rentable, on-demand (the search query asks for these; re-checked here);
- ``reliability2`` >= 0.98;
- a CUDA 12.8+ driver (``cuda_max_good`` >= 12.8): Blackwell (sm_120) needs a CUDA 12.8 build of torch;
- >= 8 effective CPU threads and >= 30 GB host RAM (our trainer: one hot main thread + 8 Rust env workers,
  ~5 physical cores busy); >= 60 GB disk; >= 200 Mbit/s down (a ~10 GB install); >= 10 GB/s measured host-to-GPU
  PCIe bandwidth (``pcie_bw``; the local 3080 Ti sits on PCIe 4.0 x16 — a host at 6.8 GB/s was seen 2026-10-12, and
  the rollout -> update copy would read the slot, not the card);
- a price ceiling (``dph_total``, $/h, all-in GPU + storage).

RANKING among the survivors: a CPU like ours first (the local box is a Ryzen 7 9800X3D: a desktop Zen 4/5 part
with high single-thread clocks, which is what the trainer's hot main thread rides), then more effective threads
(up to 16), then price, then reliability. The rank is a preference, never a hard filter: a server CPU still
qualifies, it just sorts later.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

#: Desktop Zen 4 / Zen 5 Ryzens — the CPU class of the local 9800X3D (fast single thread). Matched on cpu_name.
_LIKE_OURS = re.compile(r"Ryzen\s+(5|7|9)\s+(7\d{2}0|9\d{2}0)X3?D?\b", re.I)


@dataclass(frozen=True)
class Criteria:
    gpu_name: str = "RTX 5090"
    min_reliability: float = 0.98
    min_cuda: float = 12.8
    min_cpu_threads: float = 8.0
    min_ram_gb: float = 30.0
    min_disk_gb: float = 60.0
    min_inet_down_mbps: float = 200.0
    min_pcie_bw_gbs: float = 10.0
    max_dph: float = 1.00
    excluded_countries: Tuple[str, ...] = field(default=("CN",))


#: The country NAMES vast prints for an excluded code (geolocation looks like "Guangdong, CN" or "China").
_COUNTRY_NAMES = {"CN": ("china",)}


def country_code(offer: Dict[str, Any]) -> str:
    """The offer's ISO country code from ``geolocation`` ("Alberta, CA" -> "CA"); "" when unknown."""
    geo = str(offer.get("geolocation") or "").strip()
    if "," in geo:
        tail = geo.rsplit(",", 1)[1].strip()
        if re.fullmatch(r"[A-Za-z]{2}", tail):
            return tail.upper()
    if re.fullmatch(r"[A-Za-z]{2}", geo):
        return geo.upper()
    return ""


def is_excluded_country(offer: Dict[str, Any], excluded: Tuple[str, ...] = ("CN",)) -> bool:
    """True when the offer is in an excluded country — or its country cannot be read at all."""
    geo = str(offer.get("geolocation") or "").strip().lower()
    code = country_code(offer)
    if not geo and not code:
        return True
    if code in excluded:
        return True
    for c in excluded:
        if any(name in geo for name in _COUNTRY_NAMES.get(c, ())):
            return True
    return False


def cpu_like_ours(offer: Dict[str, Any]) -> bool:
    return bool(_LIKE_OURS.search(str(offer.get("cpu_name") or "")))


def reject_reason(offer: Dict[str, Any], c: Criteria) -> str:
    """'' when the offer passes every hard criterion, else the FIRST failing one (for the search report)."""
    if is_excluded_country(offer, c.excluded_countries):
        return f"country {country_code(offer) or 'UNKNOWN'} excluded ({offer.get('geolocation')!r})"
    checks = [
        (offer.get("gpu_name") == c.gpu_name, f"gpu {offer.get('gpu_name')!r}"),
        (int(offer.get("num_gpus") or 0) == 1, f"num_gpus {offer.get('num_gpus')}"),
        (bool(offer.get("verified", offer.get("verification") == "verified")), "unverified"),
        (bool(offer.get("rentable", True)) and not offer.get("rented", False), "not rentable"),
        (float(offer.get("reliability2") or offer.get("reliability") or 0) >= c.min_reliability,
         f"reliability {offer.get('reliability2')}"),
        (float(offer.get("cuda_max_good") or 0) >= c.min_cuda, f"cuda_max_good {offer.get('cuda_max_good')}"),
        (float(offer.get("cpu_cores_effective") or 0) >= c.min_cpu_threads,
         f"cpu threads {offer.get('cpu_cores_effective')}"),
        (float(offer.get("cpu_ram") or 0) / 1024.0 >= c.min_ram_gb, f"ram {offer.get('cpu_ram')} MB"),
        (float(offer.get("disk_space") or 0) >= c.min_disk_gb, f"disk {offer.get('disk_space')} GB"),
        (float(offer.get("inet_down") or 0) >= c.min_inet_down_mbps, f"inet_down {offer.get('inet_down')}"),
        (float(offer.get("pcie_bw") or 0) >= c.min_pcie_bw_gbs, f"pcie_bw {offer.get('pcie_bw')}"),
        (float(offer.get("dph_total") or 1e9) <= c.max_dph, f"price {offer.get('dph_total')}"),
    ]
    for ok, why in checks:
        if not ok:
            return why
    return ""


def rank_key(offer: Dict[str, Any]) -> Tuple[int, float, float, float]:
    return (0 if cpu_like_ours(offer) else 1, -min(float(offer.get("cpu_cores_effective") or 0), 16.0),
            float(offer.get("dph_total") or 1e9), -float(offer.get("reliability2") or 0))


def select(offers: List[Dict[str, Any]], c: Criteria = Criteria()) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """``(survivors ranked best-first, {reject reason: count})``."""
    keep, why = [], {}  # type: List[Dict[str, Any]], Dict[str, int]
    for o in offers:
        r = reject_reason(o, c)
        if r:
            k = r.split(" ", 1)[0] if not r.startswith("country") else "country"
            why[k] = why.get(k, 0) + 1
        else:
            keep.append(o)
    return sorted(keep, key=rank_key), why


def search_query(c: Criteria = Criteria(), limit: int = 200) -> Dict[str, Any]:
    """The ``POST /bundles/`` body: the hard criteria the server can apply (the rest are re-checked locally)."""
    return {"gpu_name": {"eq": c.gpu_name}, "num_gpus": {"eq": 1}, "rentable": {"eq": True},
            "rented": {"eq": False}, "verified": {"eq": True}, "reliability2": {"gte": c.min_reliability},
            "cuda_max_good": {"gte": c.min_cuda}, "type": "on-demand", "order": [["dph_total", "asc"]],
            "limit": limit}


def summary(o: Dict[str, Any]) -> Dict[str, Any]:
    cols = ("id", "machine_id", "geolocation", "dph_total", "reliability2", "cuda_max_good", "driver_version",
            "cpu_name", "cpu_cores_effective", "cpu_ram", "disk_space", "inet_down", "inet_down_cost",
            "inet_up_cost", "storage_cost", "pcie_bw", "gpu_ram")
    return {k: (round(o[k], 3) if isinstance(o.get(k), float) else o.get(k)) for k in cols}
