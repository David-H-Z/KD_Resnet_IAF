from unittest.mock import Mock

import pytest

from typeo import doc_utils as doc


@pytest.mark.parametrize("num_spaces", [0, 1, 4])
def test_parse_help(num_spaces):
    help_str = """
        winkin:
            An argument on the line below
        blinkin:
            An argument with multiple lines
            found below the name
        nod: An argument on the same line
    """

    help_str = [" " * num_spaces + i for i in help_str.splitlines()]
    help_str = "\n".join(help_str)

    desc = doc.parse_help(help_str, "winkin")
    assert desc == "An argument on the line below"

    desc = doc.parse_help(help_str, "blinkin")
    assert desc == "An argument with multiple lines found below the name"

    desc = doc.parse_help(help_str, "nod")
    assert desc == "An argument on the same line"


def test_parse_doc():
    desc = """Some description of the function
        that spans multiple lines"""

    args = """winkin: arg description
            blinkin:
                another arg description
            nod:
                the longest of all the
                arg descriptions"""

    # check the default expected case: all parts are there
    f = Mock()
    f.__doc__ = f"""
        {desc}
        Args:
            {args}
        Returns:
            Some value
        """

    parsed_doc, parsed_args = doc.parse_doc(f)
    assert parsed_doc.strip() == desc
    assert parsed_args.strip() == args

    # now check if desciption is missing
    f.__doc__ = f"""
        Args:
            {args}
        Returns:
            Some value
    """

    parsed_doc, parsed_args = doc.parse_doc(f)
    assert parsed_doc == ""
    assert parsed_args.strip() == args

    # now check if args are missing
    f.__doc__ = f"""
        {desc}
    """
    parsed_doc, parsed_args = doc.parse_doc(f)
    assert parsed_doc.strip() == desc
    assert parsed_args.strip() == ""

    # finally check if returns are missing
    f.__doc__ = f"""
        {desc}
        Args:
            {args}
    """
    parsed_doc, parsed_args = doc.parse_doc(f)
    assert parsed_doc.strip() == desc
    assert parsed_args.strip() == args
