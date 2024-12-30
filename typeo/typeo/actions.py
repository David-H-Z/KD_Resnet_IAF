import argparse
import importlib
import os
import re
from typing import Callable, List, Mapping, Optional

import toml

from typeo.subcommands import Subcommand


class MaybeIterableAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) == 1:
            setattr(namespace, self.dest, self.type(values[0]))
            return
        setattr(namespace, self.dest, list(map(self.type, values)))


class MappingAction(argparse.Action):
    """Action for parsing dictionary arguments

    Parse dictionary arguments using the form `key=value`,
    with the `type` argument specifying the type of `value`.
    The type of `key` must be a string. Alternatively, if
    a single argument is passed without `=` in it, it will
    be set as the value of the flag using `type`.

    Example ::

        parser = argparse.ArgumentParser()
        parser.add_argument("--a", type=int, action=_DictParsingAction)
        args = parser.parse_args(["--a", "foo=1", "bar=2"])
        assert args.a["foo"] == 1
    """

    def __init__(self, *args, **kwargs) -> None:
        self._type = kwargs["type"]
        kwargs["type"] = str
        super().__init__(*args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        if len(values) == 1 and "=" not in values[0]:
            setattr(namespace, self.dest, self._type(values[0]))
            return

        dict_value = {}
        for value in values:
            try:
                k, v = value.split("=")
            except ValueError:
                raise argparse.ArgumentTypeError(
                    self,
                    "Couldn't parse value {} passed to "
                    "argument {}".format(value, self.dest),
                )

            # TODO: introduce try-catch here
            dict_value[k] = self._type(v)
        setattr(namespace, self.dest, dict_value)


class EnumAction(argparse.Action):
    def __init__(self, *args, **kwargs) -> None:
        self._type = kwargs["type"]
        kwargs["type"] = str
        super().__init__(*args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        if self.nargs == "+":
            value = []
            for v in values:
                value.append(self._type(v))
        else:
            value = self._type(values)

        setattr(namespace, self.dest, value)


class CallableAction(argparse.Action):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["type"] = str
        super().__init__(*args, **kwargs)

    def _import_callable(self, callable: str) -> Callable:
        fn, module = callable[::-1].split(".", maxsplit=1)
        module, fn = module[::-1], fn[::-1]

        try:
            lib = importlib.import_module(module)
        except ModuleNotFoundError:
            raise argparse.ArgumentTypeError(
                self,
                "Could not find module {} for callable argument {}".format(
                    module, self.dest
                ),
            )

        try:
            # TODO: add inspection of function to make sure it
            # aligns with the Callable __args__ if there are any
            return getattr(lib, fn)
        except AttributeError:
            raise argparse.ArgumentTypeError(
                self,
                "Module {} has no function {} for callable argument {}".format(
                    module, fn, self.dest
                ),
            )

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        if self.nargs == "+":
            value = []
            for v in values:
                value.append(self._import_callable(v))
        else:
            value = self._import_callable(values)

        setattr(namespace, self.dest, value)


class TypeoTomlAction(argparse.Action):
    def __init__(
        self,
        *args,
        subcommands: List[Subcommand],
        bools: Optional[Mapping[str, bool]] = None,
        **kwargs,
    ) -> None:
        self.bools = bools
        self.subcommands = subcommands
        assert kwargs["nargs"] == "*"

        self.base_regex = re.compile(r"(?<!\\)\$\{base.(\w+)\}")
        self.env_regex = re.compile(r"(?<!\\)\$\{(\w+)\}")
        super().__init__(*args, **kwargs)

    def _get_base_value(self, arg, value):
        try:
            return self.base[value]
        except KeyError:
            raise ValueError(
                "Argument {} reference base config "
                "argument {} which doesn't exist".format(arg, value)
            )

    def _parse_string(self, value):
        # check if the value is formatted in such a way
        # as to indicate either an environment variable
        # or typeo base-section wildcard by being formatted
        # as ${} (with a \ escaping the $ if it's there)
        value = str(value)
        matches = list(set(self.base_regex.findall(value)))
        for match in matches:
            base_value = self._get_base_value(value, match)
            replace = self._parse_value(base_value)
            value = re.sub(rf"\${{base.{match}}}", replace, value)

        matches = list(set(self.env_regex.findall(value)))
        for match in matches:
            try:
                replace = os.environ[match]
            except KeyError:
                raise ValueError(
                    "No environment variable {}, referenced "
                    "in typeo config value {}".format(match, value)
                )
            value = re.sub(rf"\${{{match}}}", replace, value)
        return value

    def _parse_value(self, value):
        if isinstance(value, bool):
            return ""
        elif isinstance(value, dict):
            args = ""
            for k, v in value.items():
                v = self._parse_string(v)
                args += f"{k}={v} "
            return args[:-1]
        elif isinstance(value, list):
            args = ""
            for v in map(self._parse_string, value):
                args += v + " "
            return args[:-1]
        else:
            return self._parse_string(value)

    def _check_if_bool(self, arg, value):
        try:
            bool_default = self.bools[arg]
        except (KeyError, TypeError):
            bool_default = None

        if bool_default is None and isinstance(value, bool):
            raise argparse.ArgumentTypeError(
                self,
                "Can't parse non-boolean argument "
                "'{}' with value {}".format(arg, value),
            )
        elif bool_default is not None:
            if isinstance(value, str):
                match = self.base_regex.search(value)
                if match is not None:
                    field = match.group(1)
                    value = self._get_base_value(arg, field)

            if bool_default == value:
                return "", value
            else:
                return "--" + arg.replace("_", "-"), value
        else:
            return "--" + arg.replace("_", "-"), value

    def _parse_section(self, section):
        if isinstance(section, str):
            try:
                value = self.base_regex.search(section).group(1)
            except AttributeError:
                raise ValueError(f"Section {section} not parseable")
            else:
                section = self._get_base_value(section, value)

        if not isinstance(section, dict):
            raise ValueError(f"Section {section} not parseable")

        args = ""
        for arg, value in section.items():
            flag, value = self._check_if_bool(arg, value)
            args += flag + " "

            value = self._parse_value(value)
            if value:
                args += value + " "
        return args

    def _get_subsection(self, config, section, filename):
        if section is None:
            return config, None

        try:
            # check to see if there are any script-specific
            # config sections at all
            scripts = config.pop("scripts")
        except KeyError:
            if section is not None:
                raise argparse.ArgumentTypeError(
                    "Specified script '{}' but no 'typeo.scripts' "
                    "table found in config file '{}'".format(section, filename)
                ) from None
            section = config
            scripts = None
        else:
            try:
                section = scripts[section]
            except KeyError:
                raise argparse.ArgumentTypeError(
                    "Specified script '{}' but 'typeo.scripts.{}' "
                    "table found in config file '{}'".format(
                        section, section, filename
                    )
                ) from None
            scripts = section
        return section, scripts

    def _get_sections(self, config, section, subcommands, filename):
        section, scripts = self._get_subsection(config, section, filename)

        commands = {}
        for command, value in subcommands.items():
            try:
                cmd_section = section.pop(command)
            except KeyError:
                commands[command] = (value, {})
                continue

            try:
                subsection = cmd_section[value]
            except KeyError:
                # let this go uncaught in case this function
                # takes no arguments
                commands[command] = (value, {})
            else:
                commands[command] = (value, subsection)
        return scripts, commands

    def _parse_cmd_line(self, values):
        filename, section, subcommands = None, None, {}

        commands_to_find = [i.name for i in self.subcommands]
        if values is None and len(self.subcommands):
            raise ValueError(
                "Must specify commands " + " ".join(commands_to_find)
            )
        elif values is None:
            return "pyproject.toml", section, subcommands

        for i, value in enumerate(values):
            try:
                key, value = value.split("=")
            except ValueError:
                if i:
                    raise ValueError(
                        f"Can't parse typeo argument {value}"
                    ) from None
                filename = value
            else:
                if key == "config" and filename is not None:
                    raise ValueError(
                        "Got multiple values for config: {} and {}".format(
                            value, filename
                        )
                    )
                if key == "script":
                    section = value
                else:
                    try:
                        commands_to_find.remove(key)
                    except ValueError:
                        raise ValueError(
                            f"Subcommand {key} not found"
                        ) from None
                    subcommands[key] = value
        if len(commands_to_find):
            raise ValueError(
                "No subcommand for commands {} specified".format(
                    " ".join(commands_to_find)
                )
            )
        filename = filename or "pyproject.toml"
        return filename, section, subcommands

    def __call__(self, parser, namespace, values, option_string=None):
        values = values or None
        filename, section, subcommands = self._parse_cmd_line(values)

        if os.path.isdir(filename):
            filename = os.path.join(filename, "pyproject.toml")

        try:
            # try to load the config file
            with open(filename, "r") as f:
                config = toml.load(f)
        except FileNotFoundError:
            dirname = os.path.dirname(filename) or "."
            basename = os.path.basename(filename)
            raise argparse.ArgumentTypeError(
                self,
                "Could not find typeo config file {} in directory {}".format(
                    basename, dirname
                ),
            )

        if os.path.basename(filename) == "pyproject.toml":
            # if the config file is a pyproject.toml from
            # anywhere, assume that the file uses the
            # standard that all tool configs fall in a
            # `tool` table in the config file
            config = config["tool"]

        # now grab the typeo-specific config
        try:
            config = config["typeo"]
        except KeyError:
            raise argparse.ArgumentTypeError(
                self, f"No 'typeo' section in config file {filename}"
            )

        try:
            self.base = config.pop("base")
        except KeyError:
            self.base = {}

        scripts, commands = self._get_sections(
            config, section, subcommands, filename
        )

        # start by parsing the root typeo-level config options
        args = self._parse_section(config)

        if scripts is not None:
            # if there are any script-specific args to parse,
            # parse them out of the corresponding section
            args += self._parse_section(scripts)

        for subcommand in self.subcommands:
            value, section = commands[subcommand.name]
            args += value + " "
            args += self._parse_section(section)

        setattr(namespace, self.dest, args.split())
