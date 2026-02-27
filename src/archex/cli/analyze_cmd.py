"""CLI analyze subcommand: acquire and analyze a repository, writing an ArchProfile."""

from __future__ import annotations

import click

from archex.api import analyze
from archex.models import Config, RepoSource


@click.command("analyze")
@click.argument("source")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "markdown"]),
    default="json",
    show_default=True,
    help="Output format.",
)
@click.option(
    "-l",
    "--language",
    "languages",
    multiple=True,
    help="Filter by language (may be repeated).",
)
def analyze_cmd(source: str, output_format: str, languages: tuple[str, ...]) -> None:
    """Analyze a repository and produce an architecture profile."""
    if source.startswith("http://") or source.startswith("https://"):
        source_obj = RepoSource(url=source)
    else:
        source_obj = RepoSource(local_path=source)

    lang_list: list[str] | None = list(languages) if languages else None
    config = Config(languages=lang_list)

    profile = analyze(source_obj, config)

    if output_format == "json":
        click.echo(profile.to_json())
    else:
        click.echo(profile.to_markdown())
