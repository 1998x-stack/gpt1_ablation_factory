from __future__ import annotations

import inspect
from typing import Any, Callable, Dict


class Registry:
    """Simple pluggable registry with safe kwargs filtering."""

    def __init__(self) -> None:
        self._fns: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            if name in self._fns:
                raise KeyError(f"Duplicate registration: {name}")
            self._fns[name] = fn
            return fn
        return deco

    def create(self, name: str, **kwargs) -> Any:
        if name not in self._fns:
            raise KeyError(f"{name} not found, available: {list(self._fns)}")
        fn = self._fns[name]

        # Drop None values first
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        # If target accepts **kwargs keep all; otherwise filter by signature
        sig = inspect.signature(fn)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return fn(**kwargs)

        allowed = set(sig.parameters.keys())
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        return fn(**filtered)


MODELS = Registry()
DATASETS = Registry()
TRAINERS = Registry()
