# ruff: noqa

import os
import re
import subprocess
import sys
import textwrap
from shutil import which
from typing import Tuple


class _Color:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def _execute_command(*args: str) -> str:
    return subprocess.Popen(args, stdout=subprocess.PIPE, text=True).stdout.read()  # type: ignore


def _find_version(input: str) -> Tuple[int, int, int]:
    m = re.search(r"(\d+\.)?(\d+\.)?(\*|\d+)", input)
    assert m is not None
    version = m.group().split(".")
    return (int(version[0]), int(version[1]), int(version[2]))


def _prompt_poetry_warning() -> None:
    output = _execute_command("poetry", "--version")
    poetry_version = _find_version(output)
    poetry_version_warning_text = f"""
    {_Color.BLUE} ==> {_Color.END}{_Color.BOLD}⚠️  Old Poetry Version{_Color.END}

    You seem to have version `{".".join(map(str, poetry_version))}` of poetry which is not supported within this project.
    This may lead to problems downwards as poetry has breaking changes is its api.
    Run the following command to update poetry:

        poetry self update

    This may not work depending on how you installed poetry.
    If you installed poetry via `pipx` run:

        pipx upgrade poetry
    """

    if poetry_version < (1, 3, 2):
        print(textwrap.dedent(poetry_version_warning_text))
        sys.exit("Poetry out of date. Exiting installation.")


def _prompt_getting_started() -> None:
    getting_started_text = f"""
    {_Color.BLUE} ==> {_Color.END}{_Color.BOLD}🚀 Getting Started{_Color.END}

    You have successfully created the poetry project!

    Several make commands have been configured to cover most of the basic code tasks.
    To start the project from the main entry point, run:

        make run

    To test the code of the project using `pytest` run:

        make test

    To lint the project run:

        make lint
    """

    print(textwrap.dedent(getting_started_text))


def _prompt_ending() -> None:
    ending_text = f"""
    {_Color.BLUE}========{_Color.END} {_Color.BOLD}🎉  Happy coding!!! 🎉{_Color.END} {_Color.BLUE}========{_Color.END}

    """
    print(textwrap.dedent(ending_text))


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg == "poetry_version_check":
        _prompt_poetry_warning()
    else:
        _prompt_getting_started()
        _prompt_ending()
