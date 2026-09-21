"""Small command line helper: ``personos <command>``.

Deliberately tiny. This is a library, not a CLI tool; the commands here exist
for things you cannot do from Python in the situation where you need them —
printing schema before a database exists, and diagnosing configuration before
anything works.
"""

from __future__ import annotations

import argparse
import sys


def _schema(args: argparse.Namespace) -> int:
    from personos.storage.db.ddl import mysql_schema_sql, sqlite_ddl

    if args.sqlite:
        print(";\n".join(sqlite_ddl()) + ";")
    else:
        print(mysql_schema_sql(), end="")
    return 0


def _doctor(args: argparse.Namespace) -> int:
    from personos.diagnostics import inspect, render

    caps = inspect()
    print(render(caps))
    return 0 if all(c.available for c in caps if c.required) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="personos")
    sub = parser.add_subparsers(dest="command", required=True)

    schema = sub.add_parser(
        "schema",
        help="print the database schema (useful when the account has no DDL rights)")
    group = schema.add_mutually_exclusive_group()
    group.add_argument("--mysql", action="store_true", default=True)
    group.add_argument("--sqlite", action="store_true")
    schema.set_defaults(func=_schema)

    doctor = sub.add_parser(
        "doctor", help="report what the current configuration can do, and how to unlock the rest")
    doctor.set_defaults(func=_doctor)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
