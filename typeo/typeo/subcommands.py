import argparse
import inspect
from typing import Any, Callable, Dict, List

from typeo.doc_utils import parse_doc
from typeo.parser import make_parser
from typeo.utils import CustomHelpFormatter, is_kwargs, make_dummy


class Subcommand:
    def __init__(self, name: str, **commands: Callable) -> None:
        self.name = name
        self.subcommands = commands
        self.parsers = []

    def add_subparser(
        self, parser: argparse.ArgumentParser, func_args: List[str]
    ):
        subparsers = parser.add_subparsers(dest=self.name, required=True)
        booleans = {}
        for func_name, func in self.subcommands.items():
            description, _ = parse_doc(func)
            subparser = subparsers.add_parser(
                func_name.replace("_", "-"),
                description=description,
                formatter_class=CustomHelpFormatter,
            )
            self.parsers.append(subparser)

            dummy = make_dummy(func, func_args)
            bools = make_parser(dummy, subparser)
            booleans.update(bools)
        return booleans

    def __call__(self, kwargs: Dict[str, Any], func_args: List[str]):
        subcommand = kwargs.pop(self.name)
        subcommand = self.subcommands[subcommand.replace("-", "_")]
        subparams = inspect.signature(subcommand).parameters
        subkw = {}
        for name, param in subparams.items():
            if is_kwargs(param):
                continue
            elif name not in func_args:
                subkw[name] = kwargs.pop(name)
            else:
                subkw[name] = kwargs[name]
        return subcommand, subkw
