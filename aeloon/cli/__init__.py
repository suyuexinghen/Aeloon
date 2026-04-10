"""CLI module for aeloon."""

__all__ = ["app"]


def __getattr__(name: str):
    if name == "app":
        from aeloon.cli.app import app

        return app
    raise AttributeError(name)
