from __future__ import annotations

from typing import Any, Callable, Dict


class Registry:
    """简单可插拔 Registry。

    示例：
        MODELS.register("gpt_decoder")(GPTDecoder)
        model = MODELS.create("gpt_decoder", **kwargs)
    """
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
        return self._fns[name](**kwargs)


MODELS = Registry()
DATASETS = Registry()
TRAINERS = Registry()
