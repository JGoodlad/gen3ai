"""The cloud-GPU offer filter (pure; no network). The NO-CHINA rule is the owner's (2026-10-12) and is tested first."""
from tools.cloud.vast import offers as O


def offer(**kw):
    base = {"id": 1, "gpu_name": "RTX 5090", "num_gpus": 1, "verification": "verified", "rentable": True,
            "rented": False, "reliability2": 0.995, "cuda_max_good": 13.0, "cpu_cores_effective": 16.0,
            "cpu_ram": 31197, "disk_space": 300.0, "inet_down": 500.0, "pcie_bw": 48.0, "dph_total": 0.47,
            "geolocation": "Alberta, CA", "cpu_name": "AMD Ryzen 9 7900X 12-Core Processor"}
    base.update(kw)
    return base


def test_china_is_excluded_by_code_by_name_and_unknown_country_is_excluded_too():
    for geo in ("Guangdong, CN", "CN", "china", "Shanghai, China", " Beijing,  cn "):
        assert O.reject_reason(offer(geolocation=geo), O.Criteria()).startswith("country"), geo
    for geo in ("", None):
        assert O.reject_reason(offer(geolocation=geo), O.Criteria()).startswith("country UNKNOWN")
    # other countries are fine (owner: "all other countries are fine") — including near neighbours
    for geo in ("South Korea, KR", "Taiwan, TW", "Hong Kong, HK", "Alberta, CA", "Texas, US", "Romania, RO"):
        assert O.reject_reason(offer(geolocation=geo), O.Criteria()) == "", geo


def test_country_code_parses_the_trailing_iso_code():
    assert O.country_code(offer(geolocation="South Korea, KR")) == "KR"
    assert O.country_code(offer(geolocation="US")) == "US"
    assert O.country_code(offer(geolocation="Somewhere")) == ""


def test_each_hard_criterion_rejects_on_its_own():
    c = O.Criteria()
    cases = {
        "gpu": offer(gpu_name="RTX 4090"), "num_gpus": offer(num_gpus=2), "unverified": offer(verification="unverified"),
        "not": offer(rented=True), "reliability": offer(reliability2=0.97), "cuda_max_good": offer(cuda_max_good=12.6),
        "cpu": offer(cpu_cores_effective=4.0), "ram": offer(cpu_ram=16000), "disk": offer(disk_space=40.0),
        "inet_down": offer(inet_down=100.0), "pcie_bw": offer(pcie_bw=6.8), "price": offer(dph_total=1.5),
    }
    for want, o in cases.items():
        assert O.reject_reason(o, c).startswith(want), (want, O.reject_reason(o, c))
    assert O.reject_reason(offer(), c) == ""


def test_thresholds_are_inclusive_and_far_from_the_boundary_inputs():
    c = O.Criteria()
    assert O.reject_reason(offer(reliability2=0.98, cuda_max_good=12.8, dph_total=1.0), c) == ""


def test_rank_prefers_a_cpu_like_ours_then_threads_then_price():
    epyc_cheap = offer(id=1, cpu_name="AMD EPYC 7742 64-Core Processor", cpu_cores_effective=32.0, dph_total=0.41)
    ryzen_8t = offer(id=2, cpu_name="AMD Ryzen 7 7800X3D 8-Core Processor", cpu_cores_effective=8.0, dph_total=0.40)
    ryzen_16t = offer(id=3, cpu_name="AMD Ryzen 7 7800X3D 8-Core Processor", cpu_cores_effective=16.0, dph_total=0.47)
    ryzen_16t_dear = offer(id=4, cpu_name="AMD Ryzen 9 9950X3D 16-Core Processor", cpu_cores_effective=16.0,
                           dph_total=0.60)
    china = offer(id=5, geolocation="Guangdong, CN", cpu_name="AMD Ryzen 7 9800X3D 8-Core Processor", dph_total=0.30)
    keep, why = O.select([epyc_cheap, ryzen_8t, ryzen_16t, ryzen_16t_dear, china])
    assert [o["id"] for o in keep] == [3, 4, 2, 1]
    assert why == {"country": 1}


def test_cpu_like_ours_matches_desktop_zen4_zen5_only():
    for name in ("AMD Ryzen 7 9800X3D 8-Core Processor", "AMD Ryzen 9 7900X 12-Core Processor",
                 "AMD Ryzen 9 9950X3D 16-Core Processor", "AMD Ryzen 7 7700X 8-Core Processor"):
        assert O.cpu_like_ours({"cpu_name": name}), name
    for name in ("AMD EPYC 9554 64-Core Processor", "AMD Ryzen Threadripper 2920X 12-Core Processor",
                 "12th Gen Core i5-12400F", ""):
        assert not O.cpu_like_ours({"cpu_name": name}), name


def test_search_query_asks_the_server_for_the_hard_filters():
    q = O.search_query()
    assert q["gpu_name"] == {"eq": "RTX 5090"} and q["verified"] == {"eq": True} and q["type"] == "on-demand"
    assert q["reliability2"] == {"gte": 0.98} and q["cuda_max_good"] == {"gte": 12.8}
