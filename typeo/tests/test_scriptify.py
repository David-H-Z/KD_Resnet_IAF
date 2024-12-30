import argparse
import math
import sys
import typing
from collections import abc
from enum import Enum
from functools import partial

import pytest

from typeo import scriptify

# spoof this up front since we'll want returns
# to ensure things are actually being executed
scriptify = partial(scriptify, return_result=True)


@pytest.fixture(
    params=[
        typing.List,
        typing.Iterable,
        typing.Tuple,
        typing.Sequence,
        pytest.param(list, marks=pytest.mark.gtpy38),
        pytest.param(abc.Iterable, marks=pytest.mark.gtpy38),
        pytest.param(tuple, marks=pytest.mark.gtpy38),
        pytest.param(abc.Sequence, marks=pytest.mark.gtpy38),
    ]
)
def array_container(request):
    return request.param


@pytest.fixture(
    params=[
        typing.Callable,
        pytest.param(abc.Callable, marks=pytest.mark.gtpy38),
    ]
)
def callable_annotation(request):
    return request.param


def set_argv(*args):
    sys.argv = [None] + list(args)


def test_scriptify():
    def func(a: int, b: int = 10):
        return a + b

    # test simplest functionality
    assert func(1, 2) == scriptify(func)(1, 2) == 3
    set_argv("--a", "1", "--b", "2")
    assert scriptify(func)() == 3

    # test that default is set correctly
    set_argv("--a", "2")
    assert scriptify(func)() == 12

    # now test that we can pass a name to the scriptify parser
    assert scriptify("script name")(func)() == 12


@pytest.mark.depends(on=["test_scriptify"])
def test_scriptify_without_returns():
    from typeo import scriptify

    def func(a: int, b: int = 10):
        return a + b

    set_argv("--a", "1", "--b", "2")
    assert scriptify(func)() is None


@pytest.mark.depends(on=["test_scriptify"])
def test_scriptify_with_array_like(array_container):
    def seq_func(a: array_container[str]):
        return "".join(a)

    arg = list("test")
    assert seq_func(arg) == "test"
    assert scriptify(seq_func)(arg) == "test"

    set_argv("--a", *arg)
    assert scriptify(seq_func)() == "test"

    # do some additional checks if we're looking at a tuple
    if array_container not in (tuple, typing.Tuple):
        return

    # verify that we can use the ellipsis syntax
    def tuple_func(a: array_container[str, ...]):
        return "".join(a)

    assert scriptify(tuple_func)() == "test"

    # ensure that tuples must all have same type
    def bad_tuple_func(a: array_container[str, int]):
        pass

    with pytest.raises(AssertionError):
        scriptify(bad_tuple_func)


@pytest.mark.depends(on=["test_scriptify"])
def test_scriptify_with_optional_arg():
    def optional_func(a: int, b: typing.Optional[str] = None):
        return (b or "test") * a

    # verify functionality
    assert optional_func(1, "not none") == "not none"
    assert optional_func(2) == "testtest"

    # now scriptify-fy it
    set_argv("--a", "2")
    assert scriptify(optional_func)() == "testtest"

    set_argv("--a", "1", "--b", "not none")
    assert scriptify(optional_func)() == "not none"

    # make sure that optional args have a default of `None`
    def bad_optional_func(a: typing.Optional[str]):
        return

    with pytest.raises(ValueError):
        scriptify(bad_optional_func)

    def bad_optional_func(a: typing.Optional[str] = "nope"):
        return

    with pytest.raises(ValueError):
        scriptify(bad_optional_func)


@pytest.fixture(params=[True, False])
def bad_second_annotation(request, array_container):
    if request:
        return array_container[int]
    return int


@pytest.mark.depends(on="test_scriptify")
@pytest.mark.parametrize(
    "union", [None, pytest.param("py310", marks=pytest.mark.gtpy39)]
)
def test_scriptify_with_union(array_container, union, bad_second_annotation):
    if union is None:
        annotation = typing.Union[str, array_container[str]]
    else:
        annotation = str | array_container[str]

    def union_func(a: annotation):
        if isinstance(a, str):
            return a + " no sequence"
        else:
            return "".join(a) + " yes sequence"

    # verify expected functionality
    assert union_func("testing") == "testing no sequence"
    assert union_func(["t", "e", "s", "t"]) == "test yes sequence"

    # check scriptify-fied behavior
    set_argv("--a", "test")
    assert scriptify(union_func)() == "test no sequence"

    set_argv("--a", *"test")
    assert scriptify(union_func)() == "test yes sequence"

    # now ensure that a union can only be done
    # with a container containing the same type
    # as the first argument to the union
    if union is None:
        annotation = typing.Union[str, bad_second_annotation]
    else:
        annotation = str | bad_second_annotation

    with pytest.raises(TypeError):

        @scriptify
        def bad_union_func(a: annotation):
            return


@pytest.mark.depends(on=["test_scriptify"])
def test_subparsers():
    d = {}

    def f1(a: int, b: int):
        return a + b

    def f2(a: int, c: int):
        return a - c

    @scriptify(cmds=dict(add=f1, subtract=f2))
    def f(i: int):
        d["f"] = i

    set_argv("--i", "2", "add", "--a", "1", "--b", "2")
    assert f() == 3
    assert d["f"] == 2

    set_argv("--i", "4", "subtract", "--a", "9", "--c", "3")
    assert f() == 6
    assert d["f"] == 4


@pytest.mark.depends(on=["test_scriptify"])
def test_subparsers_with_returns():
    def f1(b: int, **kwargs):
        return kwargs["a"] + b

    def f2(c: int, **kwargs):
        return kwargs["a"] - c

    @scriptify(cmds=dict(add=f1, subtract=f2))
    def f(i: int):
        return {"a": i * 2}

    set_argv("--i", "2", "add", "--b", "2")
    assert f() == 6

    set_argv("--i", "4", "subtract", "--c", "3")
    assert f() == 5


@pytest.mark.depends(on=["test_scriptify"])
def test_enums(array_container):
    class Member(Enum):
        SINGER = "Thom"
        GUITAR = "Jonny"
        DRUMS = "Phil"

    # make sure that the parsed value comes
    # out as the appropriate Enum instance
    @scriptify
    def f(member: Member):
        return member

    set_argv("--member", "Thom")
    assert f() == Member.SINGER

    # make sure that it's argparse that
    # catches if the choice is invalid
    set_argv("--member", "error")
    with pytest.raises(SystemExit):
        f()

    # make sure that sequences of enums get
    # mapped to lists of the Enum instances
    @scriptify
    def f(members: array_container[Member]):
        return members

    set_argv("--members", "Thom", "Thom", "Jonny")
    assert f() == [Member.SINGER, Member.SINGER, Member.GUITAR]


@pytest.fixture(params=[int, float, str, "callable"])
def literal_annotation(request):
    if request.param == int:
        values = [10, 20]
        args = list(map(str, values))
        annotation = typing.Literal[10, 20]
    elif request.param == float:
        values = [2.1, 3.2]
        args = list(map(str, values))
        annotation = typing.Literal[2.1, 3.2]
    elif request.param == str:
        args = values = ["Thom", "Jonny"]
        annotation = typing.Literal["Thom", "Jonny"]
    elif request.param == "callable":
        values = [math.sqrt, math.pow]
        args = ["math." + i.__name__ for i in values]
        annotation = typing.Literal[math.sqrt, math.pow]

    return annotation, args, values


@pytest.mark.depends(on=["test_scriptify"])
def test_scriptify_with_literal(array_container, literal_annotation):
    annotation, args, values = literal_annotation

    def f(member: annotation):
        return member

    set_argv("--member", args[0])
    assert scriptify(f)() == values[0]

    # make sure that unallowed values raise an error
    set_argv("--member", "Phil")
    with pytest.raises(SystemExit):
        scriptify(f)()


@pytest.mark.depends(on=["test_scriptify_with_literal"])
def test_scriptify_with_containered_literals(
    array_container, literal_annotation
):
    annotation, args, values = literal_annotation
    annotation = array_container[annotation]

    def f(member: annotation):
        return member

    set_argv("--member", args[0], args[1], args[0])
    assert scriptify(f)() == [values[0], values[1], values[0]]

    set_argv("--member", args[1], "Phil")
    with pytest.raises(SystemExit):
        scriptify(f)()


@pytest.mark.depends(on=["test_scriptify"])
def test_blank_generics(array_container):
    """Untyped generics should default to parsing as strings"""

    @scriptify
    def blank_generic_func(a: array_container):
        return [i + "a" for i in a]

    args = ["test", "one", "two"]
    set_argv("--a", *args)

    assert blank_generic_func() == [i + "a" for i in args]

    set_argv("--a", *"123")
    assert blank_generic_func() == ["1a", "2a", "3a"]


@pytest.mark.depends(on=["test_maybe_sequence_funcs", "test_blank_generics"])
def test_unions_with_blank_generics(array_container):
    """Test generics used as the second argument to a Union

    Generic sequence types should default to the type
    of the first argument of the Union when used with
    a Union.
    """

    @scriptify
    def blank_generic_func(a: typing.Union[str, array_container]):
        return [i + "a" for i in a]

    args = ["test", "one", "two"]
    set_argv("--a", *args)

    result = blank_generic_func()

    # TODO: uncomment this when we implement
    # action for mapping to tuples
    assert isinstance(result, abc.Sequence)
    assert result == [i + "a" for i in args]

    @scriptify
    def blank_generic_func(a: typing.Union[int, array_container]):
        return [i + 2 for i in a]

    set_argv("--a", *"123")
    assert blank_generic_func() == [3, 4, 5]


@pytest.mark.depends(on=["test_scriptify"])
def test_callables(array_container, callable_annotation):
    @scriptify
    def func_of_func(f: callable_annotation):
        return f(3, 2)

    assert func_of_func(divmod) == (1, 1)
    set_argv("--f", "math.pow")
    assert func_of_func() == 9

    @scriptify
    def func_of_funcs(fs: array_container[callable_annotation]):
        return sum([f(3) for f in fs])

    answer = math.sqrt(3) + math.log(3)
    assert func_of_funcs([math.sqrt, math.log]) == answer

    set_argv("--fs", "math.sqrt", "math.log")
    assert func_of_funcs() == answer

    with pytest.raises(argparse.ArgumentTypeError):
        set_argv("--f", "bad.libary.name")
        func_of_func()


@pytest.mark.depends(on=["test_scriptify"])
def test_scriptify_kwargs():
    def reusable_func(a: int, b: str):
        return a * b

    def script_func(c: str):
        return

    with pytest.raises(ValueError):
        scriptify(kwargs=reusable_func)(script_func)

    def script_func(c: str, **kwargs):
        output = reusable_func(**kwargs)
        return c + " " + output

    func = scriptify(kwargs=reusable_func)(script_func)
    assert func("Thom", a=2, b="Yorke") == "Thom YorkeYorke"

    set_argv("--a", "2", "--b", "Yorke", "--c", "Thom")
    assert func() == "Thom YorkeYorke"

    def script_with_reused_args(a: int, c: str, **kwargs):
        return c * a + " " + reusable_func(a=a, **kwargs)

    func = scriptify(kwargs=reusable_func)(script_with_reused_args)
    assert func(a=2, c="Thom", b="Yorke") == "ThomThom YorkeYorke"
    set_argv("--a", "2", "--b", "Yorke", "--c", "Thom")
    assert func() == "ThomThom YorkeYorke"


@pytest.mark.depends(on=["test_scriptify"])
def test_scriptify_remainder():
    def reusable_func(a: int, b: str):
        return a * b

    def script_func(c: str, rest: str):
        output = " ".join(rest)
        return c + " " + output

    func = scriptify(rest=reusable_func)(script_func)
    set_argv("--a", "2", "--b", "Yorke", "--c", "Thom")
    assert func() == "Thom --a 2 --b Yorke"
