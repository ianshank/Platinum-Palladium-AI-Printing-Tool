"""Test utilities and helpers for PTPD Calibration tests.

This re-exported thirteen names that none of the three modules define, so
``import tests.utils`` raised ``ImportError`` and the package was unusable.
Nothing imported it, which is why no test ever failed over it.

Rather than list every helper by hand again -- the way the list drifted in the
first place -- each module's public surface is taken from the module itself, so
a helper added there is exported here and a helper removed cannot leave a stale
name behind.
"""

from types import ModuleType

from . import assertions, data_builders, mock_factories


def _public(module) -> list[str]:
    """Names a helper module offers, honouring ``__all__`` when it defines one."""
    declared = getattr(module, "__all__", None)
    if declared is not None:
        return list(declared)
    return [name for name in vars(module) if not name.startswith("_")]


__all__ = sorted(
    {
        name
        for module in (assertions, data_builders, mock_factories)
        for name in _public(module)
        # Skip what a module merely imported, so `from tests.utils import np`
        # does not become part of this package's surface. A module object has
        # no __module__ of its own, so it needs excluding by type as well.
        if not isinstance(vars(module)[name], ModuleType)
        and getattr(vars(module)[name], "__module__", module.__name__) == module.__name__
    }
)

for _module in (assertions, data_builders, mock_factories):
    for _name in _public(_module):
        if _name in __all__:
            globals()[_name] = vars(_module)[_name]
del _module, _name
