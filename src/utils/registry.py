"""
Generic name -> function registry.

This is the ONE mechanism every pluggable step in this pipeline uses to
support variants (preprocessing methods, feature-extraction strategies,
model architectures, etc). It replaces having a bespoke if/elif chain or
a bespoke dict in each module.

Usage pattern (identical in every module that has variants):

    from src.utils.registry import Registry

    FILTER_REGISTRY = Registry(step_name="filter")

    @FILTER_REGISTRY.register("bandpass")
    def _bandpass_filter(eeg, fs, **kwargs):
        ...

    @FILTER_REGISTRY.register("wavelet_denoise")
    def _wavelet_denoise_filter(eeg, fs, **kwargs):
        ...

    # Elsewhere, resolving a variant purely from a config string:
    filter_fn = FILTER_REGISTRY.get(cfg.preprocessing.filter_variant)
    filtered = filter_fn(eeg, fs=cfg.data.sample_rate_hz)

To add a new variant to ANY pluggable step in this codebase: write one
function matching that step's contract (documented at the top of that
step's module), decorate it with `@THAT_STEP_REGISTRY.register("name")`,
and set the config's variant field to "name". No other file needs to
change - the calling code only ever asks the registry for "whichever
function is registered under this name", so it never needs an
if/elif branch added for it.
"""

from typing import Callable


class Registry:
    """
    A name -> function lookup table for one pluggable pipeline step.

    Args:
        step_name: human-readable label for this step (e.g. "filter",
            "feature_extractor", "model"). Used only to make error
            messages clear about which registry a bad name was looked
            up in - has no effect on behavior.
    """

    def __init__(self, step_name: str):
        self.step_name = step_name
        self._entries: dict[str, Callable] = {}

    def register(self, name: str) -> Callable:
        """
        Decorator that adds a function to this registry under `name`.

        Raises if `name` is already taken, so two variants can never
        silently collide (e.g. two people adding a variant called
        "default" independently) - the second registration attempt
        fails loudly instead of overwriting the first.
        """
        def decorator(fn: Callable) -> Callable:
            if name in self._entries:
                raise ValueError(
                    f"{self.step_name!r} registry already has a variant named {name!r} "
                    f"(registered by function {self._entries[name].__name__!r}). "
                    "Choose a different name."
                )
            self._entries[name] = fn
            return fn
        return decorator

    def get(self, name: str) -> Callable:
        """
        Looks up the function registered under `name`.

        Raises a ValueError listing every valid variant name if `name`
        isn't registered - this is the error a typo in a config file
        or CLI argument will surface, so it needs to be informative on
        its own without needing to read the source.
        """
        if name not in self._entries:
            raise ValueError(
                f"Unknown {self.step_name} variant {name!r}. "
                f"Available variants: {self.names()}"
            )
        return self._entries[name]

    def names(self) -> list[str]:
        """All currently-registered variant names, sorted for stable/readable error messages."""
        return sorted(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries
