from __future__ import annotations

import argparse
import os
from pathlib import Path

from . import (
    Analyzer,
    SkillPackage,
    build_report,
    build_skill_package,
    normalize_graph,
    validate_graph,
)
from . import (
    compile as compile_skill,
)


def _prepare_cache(package, cache_dir: Path) -> tuple[Path, bool, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{package.slug}.json"
    manifest_path = cache_dir / f"{package.slug}.manifest.json"
    use_cache = True

    if manifest_path.exists():
        try:
            previous = SkillPackage.load(manifest_path)
            if previous.package_hash != package.package_hash:
                use_cache = False
        except Exception:
            use_cache = False

    package.save(manifest_path)
    return cache_path, use_cache, manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile SKILL.md (or a skill directory) into a standalone LangGraph workflow."
    )
    parser.add_argument("skill", help="Path to SKILL.md or a skill directory")
    parser.add_argument(
        "-o", "--output", default="", help="Output Python file path (required for full compile)"
    )
    parser.add_argument("--model", default="openai/gpt-5.4", help="Analyzer model")
    parser.add_argument(
        "--runtime-model", default="", help="Optional runtime model for generated LLM nodes"
    )
    parser.add_argument(
        "--base-url", default="https://openrouter.ai/api/v1", help="OpenAI-compatible base URL"
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENROUTER_API_KEY", ""),
        help="API key (default: OPENROUTER_API_KEY)",
    )
    parser.add_argument("--cache-dir", default="output/graphs", help="Graph cache directory")
    parser.add_argument(
        "--report-path", default="", help="Optional explicit path for compile report JSON"
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only run analysis+grounding and cache the graph",
    )
    parser.add_argument(
        "--validate-only", action="store_true", help="Only run analysis+validation and emit report"
    )
    parser.add_argument(
        "--strict-validate",
        action="store_true",
        help="Fail compile if SkillGraph validation finds errors (default: loose mode)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.analyze_only and args.validate_only:
        parser.error("--analyze-only and --validate-only cannot be used together")

    if not args.output and not args.analyze_only and not args.validate_only:
        parser.error("-o/--output is required unless using --analyze-only or --validate-only")

    skill_path = Path(args.skill)
    cache_dir = Path(args.cache_dir)

    if args.analyze_only or args.validate_only:
        package = build_skill_package(skill_path)
        cache_path, use_cache, _ = _prepare_cache(package, cache_dir)
        entry_skill = Path(package.skill_root) / package.entry_skill

        analyzer = Analyzer(model=args.model, api_key=args.api_key, base_url=args.base_url)
        graph = analyzer.analyze(entry_skill, cache_path=cache_path, use_cache=use_cache)
        graph = normalize_graph(graph)

        if args.analyze_only:
            print(f"analyzed: {package.slug} | steps={len(graph.steps)} edges={len(graph.edges)}")
            print(cache_path)
            return

        validation = validate_graph(graph)
        if validation.errors:
            for issue in validation.errors:
                print(f"ERROR [{issue.code}] {issue.message} ({issue.step_id or 'graph'})")
        if validation.warnings:
            for issue in validation.warnings:
                print(f"WARN  [{issue.code}] {issue.message} ({issue.step_id or 'graph'})")

        report_target = (
            Path(args.report_path)
            if args.report_path
            else cache_dir / f"{package.slug}.report.json"
        )
        report = build_report(graph, package, args.output or "(validate-only)", validation)
        report.save(report_target)
        print(report_target)

        if args.strict_validate and validation.errors:
            raise SystemExit(1)
        return

    output = compile_skill(
        skill_path=skill_path,
        output_path=Path(args.output),
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        runtime_model=args.runtime_model or None,
        cache_dir=cache_dir,
        strict_validate=args.strict_validate,
        report_path=Path(args.report_path) if args.report_path else None,
    )
    print(output)


if __name__ == "__main__":
    main()
