"""Command-line interface."""

import click


@click.command()
@click.version_option()
def main() -> None:
    """SSB pre-system."""


if __name__ == "__main__":
    main(prog_name="ssb-pre-system")  # pragma: no cover
