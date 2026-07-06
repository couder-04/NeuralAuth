"""
training/smoke_test_config.py

Manual smoke test for training/config.py's `Config` loader -- confirms
`.env`/dotenv discovery works from a given working directory. NOT a
pytest test: it used to be named `test_api.py`, which (a) made bare
`pytest` fail to even collect (its unqualified `from config import
Config` only resolves when CWD is `training/`), and (b) printed the
loaded API key to stdout. Renamed out of pytest's `test_*.py` discovery
and guarded behind `__main__`; the secret is no longer printed.

Run explicitly with (from the `training/` directory):
    python smoke_test_config.py
"""

import os

from dotenv import find_dotenv

from config import Config


def main() -> None:
    print("cwd =", os.getcwd())
    print("dotenv =", find_dotenv())

    cfg = Config()

    # Never print the raw secret -- just confirm one was loaded.
    print("api_key loaded:", bool(cfg.api_key))


if __name__ == "__main__":
    main()