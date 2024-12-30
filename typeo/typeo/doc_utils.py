import re
from typing import Callable


def parse_help(args: str, arg_name: str) -> str:
    """Find the help string for an argument

    Search through the `Args` section of a function's
    doc string for the lines describing a particular
    argument. Returns the empty string if no description
    is found

    Args:
        args:
            The arguments section of a function docstring.
            Should be formatted like
            ```
            '''
            arg1: The description for arg1 in one line
            arg2:
                The description for arg 2 that spans
                multiple lines
            arg3:
                Another description on the line below
            '''
            Argument descriptions should either be
            fully inline with the argument name, or
            on lines below the argument name each indented
            by four spaces.
            ```
        arg_name:
            The name of the argument whose help string
            to search for
    Returns:
        The help string for the argument with leading
        spaces stripped for each line and newlines
        replaced by spaces
    """

    # check if there's an entry in the doc_str for
    # the argument name at all
    match = re.search(rf"(?m)(?P<spaces>^( )+){arg_name}:", args)
    if match is None:
        return ""

    # strip out the number of spaces before the
    # argument name to put all args at 0 indent
    args = re.sub("(?m)^" + match.group("spaces"), "", args)

    # now check either for the description on the
    # same line as the argument, or in any indented
    # rows immediately following it
    indent = r"\n {4}"
    desc = f"(.+|({indent}.+)+)"
    match = re.search(rf"(?m)(?<=^{arg_name}:){desc}", args)

    # if there's no description that looks like this
    # then short circuit now and just return nothing
    if match is None:
        return ""

    # now strip out the newlines and indententations
    description = match.group(0)
    return re.sub("(?m)" + indent, " ", description).strip()


def parse_doc(f: Callable):
    """Grab any documentation and argument help from a function"""

    # start by grabbing the function description
    # and any arguments that might have been
    # described in the docstring
    try:
        # split thet description and the args
        # by the expected argument section header
        doc, args = f.__doc__.split("Args:\n")
    except AttributeError:
        # raised if f doesn't have documentation
        doc, args = "", ""
    except ValueError:
        # raised if f only has a description but
        # no argument documentation. Set `args`
        # to the empty string
        doc, args = f.__doc__, ""
    else:
        # try to strip out any returns from the
        # arguments section by using the expected
        # returns header. If there are None, just
        # keep moving

        try:
            args, _ = args.split("Returns:\n")
        except ValueError:
            pass

    # check if the doc is blank
    if not doc.strip():
        doc = ""
    return doc, args
