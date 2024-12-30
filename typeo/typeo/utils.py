import argparse
import inspect
from typing import Callable, List


class CustomHelpFormatter(argparse.RawDescriptionHelpFormatter):
    def _get_help_string(self, action):
        return argparse.ArgumentDefaultsHelpFormatter._get_help_string(
            self, action
        )


def make_dummy(func: Callable, exclude: List[str]) -> Callable:
    def dummy(*args, **kwargs):
        return

    params = []
    for param in inspect.signature(func).parameters.values():
        if param.name not in exclude:
            params.append(param)

    dummy.__signature__ = inspect.Signature(params)
    dummy.__doc__ = func.__doc__
    return dummy


def is_kwargs(param: inspect.Parameter):
    return param.kind is inspect._ParameterKind.VAR_KEYWORD


def is_args(param: inspect.Parameter):
    return param.kind is inspect._ParameterKind.VAR_POSITIONAL
