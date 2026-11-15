"""Method registry."""
from __future__ import annotations

from . import fedapen, fedavg, fedfv, phpfl

RUNNERS = {
    "FedAvg": fedavg.run,
    "FedAPEN": fedapen.run,
    "FedFV": fedfv.run,
    "PHP-FL": phpfl.run,
}


def get_runner(method: str):
    if method not in RUNNERS:
        raise KeyError(f"unknown method {method!r}; known: {sorted(RUNNERS)}")
    return RUNNERS[method]
