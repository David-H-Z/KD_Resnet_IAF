import argparse
import inspect
import types
from collections import abc
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import typeo.actions as actions
from typeo.doc_utils import parse_doc, parse_help
from typeo.utils import is_args, is_kwargs

if TYPE_CHECKING:
    try:
        from types import GenericAlias
    except ImportError:
        from typing import _GenericAlias as GenericAlias


_LIST_ORIGINS = (list, abc.Sequence, abc.Iterable)
_ARRAY_ORIGINS = _LIST_ORIGINS + (tuple,)
_DICT_ORIGINS = (dict, abc.Mapping)
_ANNOTATION = Union[type, "GenericAlias"]
_MAYBE_TYPE = Optional[type]


def _parse_union(param: inspect.Parameter) -> Tuple[type, _MAYBE_TYPE]:
    annotation = param.annotation

    try:
        # type_ will be the type that we pass to the
        # parser. second_type can either be `None` or
        # some array of type_
        type_, second_type = annotation.__args__
    except TypeError:
        raise ValueError(
            "Can't parse argument {} with annotation {} "
            "that is a Union of more than 2 types.".format(
                param.name, annotation
            )
        )

    try:
        origin = second_type.__origin__
    except AttributeError:
        # if the second argument to Union doesn't have
        # an origin, it's not array-like and so is
        # only allowed if it's None (which basically
        # corresponds to the `Optional` case)
        try:
            # see if the second type in the Union is NoneType
            is_none = isinstance(None, second_type)
        except TypeError:
            # annotation.__args__[1] is not a type that we
            # can check `None` as an instance of, so the
            # `isinstance` check above raises a TypeError.
            is_none = False

        if is_none:
            # if this second argument is None, i.e. we
            # have an Optional, the default must be None
            # otherwise there's no way for us to ever
            # be able to parse that `None` value from the
            # command line
            if param.default is not None:
                raise ValueError(
                    "Argument {} with Union of type {} and "
                    "NoneType must have a default of None".format(
                        param.name, type_
                    )
                )
            # we're done parsing now so return the two types
            return type_, None
        else:
            # this annotation isn't NoneType and doesn't have an
            # __origin__, so we can infer that it's not array-like
            # and we don't know how to parse this arg
            raise TypeError(
                "Arg {} has Union of types {} and {}".format(
                    param.name, type_, second_type
                )
            )

    # check if the type passed to the array-like
    # second argument to Union matches with the
    # type of the first argument to Union
    if origin in _LIST_ORIGINS:
        idx_to_check = 0
    elif origin in _DICT_ORIGINS:
        idx_to_check = 1
    elif origin is tuple:
        idx_to_check = None
    else:
        raise TypeError(
            "Arg {} has Union of type {} and type {} "
            "with unknown origin {}".format(
                param.name, type_, second_type, origin
            )
        )

    try:
        args = second_type.__args__
    except AttributeError:
        # in py3.9, generic aliases with no type
        # specified won't have __args__ at all,
        # so this is the same as being TypeVar
        # in py<3.9 and we're ok
        pass
    else:
        if idx_to_check is not None:
            args = [args[idx_to_check]]

        for arg in args:
            if arg not in (type_, Ellipsis) and not isinstance(arg, TypeVar):
                raise TypeError(
                    "Type argument {} passed to annotation {} of "
                    "parameter {} can't be parsed".format(
                        arg, annotation, param.name
                    )
                )
    return second_type, type_


def _get_origin_and_type(
    annotation: _ANNOTATION, type_: _MAYBE_TYPE = None
) -> Tuple[_MAYBE_TYPE, _MAYBE_TYPE]:
    """Utility for parsing the origin of an annotation

    Returns:
        If the annotation has an origin, this will be that origin.
            Otherwise it will be `None`
        If the annotation does not have an origin, this will
            be the annotation. Otherwise it will be `None`
    """

    try:
        return annotation.__origin__, type_
    except AttributeError:
        # annotation has no origin, so assume it's
        # a valid type on its own
        return None, annotation


def _parse_literal(annotation: _ANNOTATION):
    args = annotation.__args__
    type_ = type(args[0])
    if len(args) > 1:
        assert all([isinstance(i, type_) for i in args[1:]])

    if isinstance(args[0], Callable):
        type_ = abc.Callable
        args = [f"{i.__module__}.{i.__name__}" for i in args]
    return type_, args


def _is_untyped(args):
    # args being None indicates untyped lists and tuples in py3.9,
    # and args[0] being TypeVar indicates untyped lists in py3.8
    return args is None or isinstance(args[0], TypeVar)


def _parse_array_like(
    annotation: _ANNOTATION, origin: _MAYBE_TYPE, type_: type
) -> Tuple[type, Optional[Callable], Optional[List]]:
    """Grab types and expected actions for array-like annotations"""

    try:
        args = annotation.__args__
    except AttributeError:
        args = None

    if origin in _DICT_ORIGINS:
        action = actions.MappingAction
        if not _is_untyped(args):
            # make sure that the expected type
            # for the dictionary key is string
            # TODO: add kwarg for parsing non-str
            # dictionary keys
            assert args[0] is str

            # the type used to parse the values for
            # the dictionary will be the type passed
            # the parser action
            type_ = args[1]
    else:
        action = None
        try:
            if not _is_untyped(args):
                type_ = args[0]
        except IndexError:
            # untyped Tuples in py3.8 will have an empty __args__
            pass
        else:
            # for tuples make sure that everything
            # has the same type
            if origin is tuple and not _is_untyped(args):
                # TODO: use a custom action to verify the
                # number of arguments and map to a tuple
                try:
                    for arg in annotation.__args__[1:]:
                        if arg is not Ellipsis:
                            assert arg == type_
                except IndexError:
                    # if the Tuple only has one arg, we don't need
                    # to worry about checking everything else
                    pass

    # check to see if this array-like container
    # contains literals, in which case parse out
    # the type and choices expected by the literal
    try:
        is_literal = type_.__origin__ is Literal
    except (TypeError, AttributeError):
        choices = None
    else:
        if is_literal:
            type_, choices = _parse_literal(type_)
        else:
            choices = None

    return type_, action, choices


def _parse_container(
    annotation: _ANNOTATION, origin: type, kwargs: dict, type_: type
) -> _MAYBE_TYPE:
    """Make sure container-like arguments pass the right type to the parser

    For an annotation with an origin, do some checks on the
    origin to make sure that the type and action argparse
    uses to parse the argument is correct. If the annotation
    doesn't have an origin, returns `None`.

    Args:
        annotation:
            The annotation for the argument
        origin:
            The origin of the annotation, if it exists,
            otherwise `None`
        kwargs:
            The dictionary of keyword arguments to be
            used to add an argument to the parser
    """

    if origin in _ARRAY_ORIGINS + _DICT_ORIGINS:
        kwargs["nargs"] = "+"
        type_, action, choices = _parse_array_like(annotation, origin, type_)

        # check if the type contained in the array-like
        # annotation requires any kind of special action
        # or is a literal and so needs specific choices
        if action is not None:
            kwargs["action"] = action
        if choices is not None:
            kwargs["choices"] = choices

    elif origin is abc.Callable:
        type_ = origin
    elif origin is Literal:
        type_, choices = _parse_literal(annotation)
        kwargs["choices"] = choices
    else:
        # this is a type with some unknown origin
        raise TypeError(f"Can't help with arg of type {origin}")

    return type_


def _standardize_none_origin(type_):
    """
    Standardize py39 and py310 annotations to the format
    required to express them in py38
    """

    origin = None
    if type_ in _ARRAY_ORIGINS:
        # this will happen for untyped containers in py39+,
        # so just treat it like the case for py38
        origin, type_ = type_, None
    else:
        try:
            if isinstance(type_, types.UnionType):
                origin = Union
        except AttributeError:
            pass
    return origin, type_


def make_parser(
    func: Callable,
    parser: argparse.ArgumentParser,
) -> Dict[str, bool]:
    """Build an argument parser for a function

    Builds an `argparse.ArgumentParser` object by using
    the arguments to a function `f`, as well as their
    annotations and docstrings (for help printing).
    The type support for annotations is pretty limited
    and needs more documenting here, but for a better
    idea see the unit tests in `../tests/unit/test_scriptify.py`.

    Args:
        f:
            The function to construct a command line
            argument parser for
        parser:
            An existing parser to which to add arguments.
    Returns:
        A mapping from the names of any boolean arguments
            to the function to their default values, to be
            used for typeo config parsing.
    """

    _, args = parse_doc(func)
    parameters = inspect.signature(func).parameters

    # now iterate through the arguments of f
    # and add them as options to the parser
    booleans = {}
    for name, param in parameters.items():
        if is_args(param) or is_kwargs(param):
            # This skips **kwargs style arguments, which we won't
            # be able to parse
            continue

        annotation = param.annotation
        kwargs = {}

        # check to see if the annotation represents
        # a type that can be used by the parser, or
        # represents some container that needs
        # further parsing
        origin, type_ = _get_origin_and_type(annotation)

        if origin is None:
            # do some standardization of things that are
            # allowable in py39 and py310 that require a
            # different syntax in py38
            origin, type_ = _standardize_none_origin(type_)

        # if the annotation can have multiple types,
        # figure out which type to pass to the parser
        if origin is Union:
            annotation, type_ = _parse_union(param)

            # check the chosen type again to
            # see if it's a container of some kind
            origin, type_ = _get_origin_and_type(annotation, type_)
            if origin in _ARRAY_ORIGINS:
                kwargs["action"] = actions.MaybeIterableAction

        if origin is not None:
            # if the annotation has some sort of origin, this
            # indicates a container type, so do some further
            # parsing of it to see what types of objects this
            # container is meant to contain so we can pass
            # those to the parser
            type_ = _parse_container(annotation, origin, kwargs, type_)

        # our last origin check to see if type_ is typing.Callable,
        # in which case the origin will be abc.Callable which
        # is the type that we want
        origin, type_ = _get_origin_and_type(type_)
        if origin is not None:
            type_ = origin

        # add the argument docstring to the parser help
        kwargs["help"] = parse_help(args, name)

        if type_ is bool:
            if param.default is inspect._empty:
                # if the argument is a boolean and doesn't
                # provide a default, assume that setting it
                # as a flag indicates a `True` status
                kwargs["action"] = "store_true"
                booleans[name] = False
            else:
                # otherwise set the action to be the
                # _opposite_ of whatever the default is
                # so that if it's not set, the default
                # becomes the values
                booleans[name] = param.default
                action = str(not param.default).lower()
                kwargs["action"] = f"store_{action}"
        else:
            kwargs["type"] = type_

            # args without default are required,
            # otherwise pass the default to the parser
            if param.default is inspect._empty:
                kwargs["required"] = True
            else:
                kwargs["default"] = param.default

            if type_ is abc.Callable:
                kwargs["action"] = actions.CallableAction
            elif type_ is not None and issubclass(type_, Enum):
                kwargs["action"] = actions.EnumAction
                kwargs["choices"] = [i.value for i in type_]

        # use dashes instead of underscores for argument names
        name = name.replace("_", "-")
        parser.add_argument(f"--{name}", **kwargs)
    return booleans
