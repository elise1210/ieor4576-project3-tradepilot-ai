from tests.evals.benchmark_cases import BENCHMARK_CASES
from tests.evals.benchmark_cases_v2 import BENCHMARK_CASES_V2


BENCHMARK_SUITES = {
    "v1": BENCHMARK_CASES,
    "v2": BENCHMARK_CASES_V2,
}


def get_benchmark_cases(suite: str = "v1") -> list[dict]:
    key = (suite or "v1").strip().lower()
    if key not in BENCHMARK_SUITES:
        raise ValueError(f"Unknown benchmark suite: {suite}")
    return list(BENCHMARK_SUITES[key])


__all__ = ["BENCHMARK_SUITES", "get_benchmark_cases"]
