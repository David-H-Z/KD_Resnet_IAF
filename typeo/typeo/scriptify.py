import argparse
import inspect
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

import typeo.actions as actions
from typeo.doc_utils import parse_doc
from typeo.parser import make_parser
from typeo.subcommands import Subcommand
from typeo.utils import CustomHelpFormatter, is_kwargs, make_dummy


class _RemainderParser:
    def __init__(
        self, argname: str, func: Callable, exclude: List[str]
    ) -> None:
        self.argname = argname
        dummy = make_dummy(func, exclude)
        if not len(dummy.__signature__.parameters):
            self.parser = None
        else:
            self.parser = argparse.ArgumentParser(
                prog=func.__name__, add_help=False
            )

        self.bools = make_parser(dummy, self.parser)

    def __call__(
        self,
        remainder: List[str],
        parser: argparse.ArgumentParser,
        kw: Dict[str, Any],
    ):
        if remainder and self.parser is None:
            parser.error(
                "unrecognized arguments: {}".format(" ".join(remainder))
            )
        elif self.parser is not None and not remainder:
            self.parser.error(
                f"Missing arguments for function {self.parser.prog}"
            )
        elif remainder:
            try:
                self.parser.parse_args(remainder)
            except SystemExit as e:
                self.parser.error(
                    "Error while parsing *args from function {}: "
                    "{}".format(self.parser.prog, str(e))
                )
            kw[self.argname] = remainder


def _make_subcommands(
    parser: argparse.ArgumentParser,
    kwargs: Dict[str, Any],
    func_args: List[str],
    full_args: List[str],
    booleans: Dict[str, bool],
):
    subcommands, remainder_parser = [], None
    parsers = [parser]
    for argname, options in kwargs.items():
        if argname in full_args and isinstance(options, Callable):
            remainder_parser = _RemainderParser(argname, options, func_args)
            booleans.update(remainder_parser.bools)
        elif not isinstance(options, dict):
            raise ValueError(
                "Can't add parser for argument {} "
                "with value {}".format(argname, options)
            )
        else:
            subcommand = Subcommand(argname, **options)
            for p in parsers:
                bools = subcommand.add_subparser(p, func_args)
            parsers = subcommand.parsers

            subcommands.append(subcommand)
            booleans.update(bools)

    if remainder_parser and subcommands:
        raise ValueError(
            "Can't parse remaining args for function {} "
            "and have subcommands".format(remainder_parser.parser.prog)
        )
    if remainder_parser is None:
        remainder_parser = _RemainderParser(None, lambda: None, [])
    return subcommands, remainder_parser


def _make_wrapper(
    func: Callable,
    prog: Optional[str] = None,
    return_result: bool = False,
    kwargs: Optional[Callable] = None,
    **other_kwargs,
) -> Callable:
    # start with a parent parser that will initially
    # try to parse a typeo toml config argument
    # let the downstream parser worry about handling help.
    # Don't add the --typeo argument to it yet though, since
    # this will need information about boolean arguments
    # extracted from the downstream parsers
    parent_parser = argparse.ArgumentParser(
        prog="config-parser", add_help=False, conflict_handler="resolve"
    )

    # now build a parser for the main function `f` which
    # inherits from this parser and can parse whatever
    # the config parser can't understand. The point of
    # inheritance is so that the `-h` flag will trigger
    # help from this parser and include the '--typeo' flag
    description, _ = parse_doc(func)
    parser = argparse.ArgumentParser(
        prog=prog or func.__name__,
        description=description.rstrip(),
        formatter_class=CustomHelpFormatter,
        parents=[parent_parser],
    )

    # exlude any parameterized arguments from the
    # arguments created by the parser
    dummy_func = make_dummy(func, list(other_kwargs))
    booleans = make_parser(dummy_func, parser)

    func_params = inspect.signature(func).parameters
    full_args = list(func_params)
    func_args = list(inspect.signature(dummy_func).parameters)

    # include any arguments we may want to pass to **kwargs
    if kwargs is not None:
        if not any(map(is_kwargs, func_params.values())):
            raise ValueError(
                "No **kwargs to pass arguments of function {} to".format(
                    kwargs
                )
            )

        dummy = make_dummy(kwargs, func_args)
        bools = make_parser(dummy, parser)
        booleans.update(bools)

    subcommands, remainder_parser = _make_subcommands(
        parser, other_kwargs, func_args, full_args, booleans
    )

    # now add an argument for parsing a config file, using
    # info about booleans stripped from downstream parsers
    parent_parser.add_argument(
        "--typeo",
        bools=booleans,
        subcommands=subcommands,
        nargs="*",
        required=False,
        default=None,
        action=actions.TypeoTomlAction,
        help=(
            "Path to a typeo TOML config file of the form "
            "`path(:section)(:command)`, where `section` "
            "and `command` are optional. `path` can either be "
            "the path to a config file or to a directory with "
            "a `pyproject.toml` that will be used as the config. "
            "If left blank, a `pyproject.toml` file will be "
            "searched for in the current working directory. "
            "`section` specifies a subtable of the config in "
            "which to search for arguments, and `command` specifies "
            "a subcommand of the main function to execute, whose "
            "arguments are assumed to fall in a subtable of the config "
            "by that name."
        ),
    )

    # now build a wrapper for the function `f` that
    # parses from the command line if no arguments
    # are passed in, and otherwise just calls `f`
    # regularly
    @wraps(func)
    def wrapper(*args, **kw):
        if not len(args) == len(kw) == 0:
            # if any arguments at all were provided, run f normally
            return func(*args, **kw)

        config_args, remainder = parent_parser.parse_known_args()
        if config_args.typeo is not None:
            # TODO: what's the best way to have command line
            # arguments override those in the typeo config?
            if remainder:
                raise ValueError(
                    "Found additional arguments '{}' when passing "
                    "typeo config".format(remainder)
                )
            remainder = config_args.typeo

        kw, remainder = parser.parse_known_args(remainder)
        kw = vars(kw)
        remainder_parser(remainder, parser, kw)

        post_commands = []
        for subcommand in subcommands:
            cls, subkw = subcommand(kw, func_args)
            if subcommand.name in full_args:
                kw[subcommand.name] = cls(**subkw)
            else:
                post_commands.append((cls, subkw))

        # run the main function and potentially the subcommand
        result = func(*args, **kw)
        for cmd, subkw in post_commands:
            # if the main function returned a dictionary,
            # pass it as kwargs to the subcommand
            if isinstance(result, dict):
                subkw.update(result)
            result = cmd(**subkw)

        if return_result:
            return result

    return wrapper


def scriptify(*args, **kwargs) -> Callable:
    """Function wrapper for passing command line args to functions

    Builds a command line parser for the arguments
    of a function so that if it is called without
    any arguments, its arguments will be attempted
    to be parsed from `sys.argv`.

    Usage:
        If your file `adder.py` looks like ::

            from typeo import scriptify


            @scriptify
            def f(a: int, other_number: int = 1) -> int:
                '''Adds two numbers together

                Longer description of the process of adding
                two numbers together.

                Args:
                    a:
                        The first number to add
                    other_number:
                        The other number to add whose description
                        inexplicably spans multiple lines
                '''

                print(a + other_number)


            if __name__ == "__main__":
                f()

        Then from the command line (note that underscores
        get replaced by dashes!) ::
            $ python adder.py --a 1 --other-number 2
            3
            $ python adder.py --a 4
            5
            $ python adder.py -h
            usage: f [-h] --a A [--other-number OTHER_NUMBER]

            Adds two numbers together

                Longer description of the process of adding
                two numbers together.

            optional arguments:
              -h, --help            show this help message and exit
              --a A                 The first number to add
              --other-number OTHER_NUMBER
                                    The other number to add whose description inexplicably spans multiple lines  # noqa

    Args:
        f: The function to expose via a command line parser
        prog:
            The name to assign to command line parser `prog`
            argument. If not provided, `f.__name__` will
            be used.
        extends:
            Tuple specifying a function `g` whose arguments `f`
            is meant to provide, as well as the names of
            the arguments of `g` that the outputs of `f` will be
            used to feed. If specified, all other arguments of
            `g` will be exposed to the command line and passed
            through straight-fowardly. In this case, `f` is
            executed first, then its outputs are passed to
            the corresponding inputs of `g`.
        return_result:
            Whether to return the output of any functions executed
            when `f` is called without any arguments. If left as
            `False`, `f()` will return None. This is the default
            behavior due to issues with conda raising an error
            on script-executed functions with return values, see
            https://github.com/ML4GW/BBHNet/issues/173
    """

    # the only argument is the function itself,
    # so just treat this like a simple wrapper
    if len(args) == 1 and isinstance(args[0], Callable):
        return _make_wrapper(args[0], **kwargs)
    else:
        # we provided arguments to typeo above the
        # decorated function, so wrap the wrapper
        # using the provided arguments

        @wraps(scriptify)
        def wrapperwrapper(f):
            return _make_wrapper(f, *args, **kwargs)

        return wrapperwrapper
