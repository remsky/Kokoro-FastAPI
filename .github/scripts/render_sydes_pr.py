#!/usr/bin/env python3
"""Render a reviewer-facing Markdown summary from a Sydes result JSON.

Reads the machine-readable result written by `sydes verify-change --json` and
emits the concise body used for both the PR comment and the Actions job
summary.

Diagnostics are deliberately never rendered here. CBM timings, graph-slice
counts, route-graph internals, prompt sizes and guide counters belong in the
job log and the uploaded artifact, not in a reviewer's PR conversation. This
script reads only the user-facing fields, and tolerates any of them being
absent so a partial or failed run still produces something readable.

Usage:
    render_sydes_pr.py RESULT_JSON --out comment.md
                       [--diagnostics-out diagnostics.json] [--run-url URL]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MARKER = "<!-- sydes-verification-comment -->"

#: Verdicts Sydes can report, and how they read at a glance. An unknown
#: verdict still renders (with a neutral marker) rather than being dropped.
_VERDICT_ICONS = {
    "VERIFIED": "✅",
    "VERIFICATION INCOMPLETE": "⚠️",
    "ACTION REQUIRED": "❌",
}

_SEVERITY_ICONS = {"P0": "🔴", "P1": "🟠", "P2": "🟡", "P3": "⚪"}

#: Keep the comment scannable; the full lists live in the JSON artifact.
_MAX_BEHAVIOR_CHANGES = 5
_MAX_GAPS = 5
_MAX_FINDINGS = 10
_MAX_SYMBOLS = 15
_MAX_IMPACTS = 10
_MAX_NOTES = 6


def _get(mapping: Any, *keys: str, default: Any = None) -> Any:
    """Walk nested dicts without assuming any level exists."""
    current = mapping
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _clean(text: Any, limit: int = 500) -> str:
    """Collapse a model-authored string to one safe, bounded Markdown line."""
    if not isinstance(text, str):
        return ""
    flattened = " ".join(text.split())
    if len(flattened) > limit:
        flattened = flattened[: limit - 1].rstrip() + "…"
    return flattened


def _test_file_paths(result: dict[str, Any]) -> set[str]:
    """Paths whose file role marks them as tests, for the production split."""
    paths: set[str] = set()
    for item in _as_list(_get(result, "change", "files", default=[])):
        role = str(_get(item, "role", default="") or "")
        path = _get(item, "path", default="")
        if path and "test" in role.lower():
            paths.add(str(path))
    return paths


def _render_header(result: dict[str, Any], lines: list[str]) -> None:
    verdict = str(_get(result, "summary", "verdict", default="UNKNOWN"))
    risk = str(_get(result, "summary", "risk", default="UNKNOWN"))
    analysis = str(_get(result, "analysis_status", default="unknown")).upper()
    icon = _VERDICT_ICONS.get(verdict, "ℹ️")

    lines.append("## Sydes verification")
    lines.append("")
    lines.append("| | |")
    lines.append("| --- | --- |")
    lines.append(f"| **Verdict** | {icon} `{verdict}` |")
    lines.append(f"| **Risk** | `{risk}` |")
    lines.append(f"| **Analysis** | `{analysis}` |")
    lines.append("")

    headline = _clean(_get(result, "summary", "headline", default=""))
    if headline:
        lines.append(f"> {headline}")
        lines.append("")

    # The check and the verdict answer different questions; say so plainly so a
    # green check next to an incomplete verdict does not read as a contradiction.
    lines.append(
        "_The check reports whether Sydes ran successfully. The verdict reports what Sydes "
        "could establish — a passing check with an incomplete verdict is expected, not a failure._"
    )
    lines.append("")


def _render_change(result: dict[str, Any], lines: list[str]) -> None:
    analysis = _get(result, "pr_semantic_analysis", default={})
    summary = _clean(_get(analysis, "change_summary", default=""), limit=700)
    behaviors = _as_list(_get(analysis, "behavior_changes", default=[]))
    if not summary and not behaviors:
        return

    lines.append("### Change")
    lines.append("")
    if summary:
        lines.append(summary)
        lines.append("")
    for item in behaviors[:_MAX_BEHAVIOR_CHANGES]:
        description = _clean(_get(item, "description", default=""))
        if not description:
            continue
        confidence = _get(item, "confidence")
        suffix = ""
        if isinstance(confidence, (int, float)):
            suffix = f" _(confidence {float(confidence):.2f})_"
        lines.append(f"- {description}{suffix}")
    remaining = len(behaviors) - _MAX_BEHAVIOR_CHANGES
    if remaining > 0:
        lines.append(f"- _…and {remaining} more behavior change(s) in the full result._")
    lines.append("")


def _render_verification(result: dict[str, Any], lines: list[str]) -> None:
    counts = _get(result, "summary", "counts", default={})
    test_paths = _test_file_paths(result)
    symbols = _as_list(_get(result, "change", "symbols", default=[]))
    test_symbols = sum(1 for item in symbols if str(_get(item, "file", default="")) in test_paths)
    production_symbols = len(symbols) - test_symbols

    lines.append("### Verification")
    lines.append("")
    lines.append(
        f"- **Changed files:** {counts.get('changed_files', 0)} "
        f"({counts.get('changed_source_files', 0)} source, {counts.get('changed_test_files', 0)} test)"
    )
    # Only show the production/test split when it actually adds up to the
    # reported total, so the line can never render inconsistent arithmetic.
    total_symbols = counts.get("changed_symbols", len(symbols))
    if symbols and total_symbols == len(symbols):
        lines.append(
            f"- **Changed symbols:** {total_symbols} "
            f"({production_symbols} production, {test_symbols} test)"
        )
    else:
        lines.append(f"- **Changed symbols:** {total_symbols}")
    lines.append(
        f"- **Affected behavior:** {counts.get('impacts_proven', 0)} proven · "
        f"{counts.get('impacts_inferred', 0)} inferred"
    )
    lines.append(f"- **Affected flows:** {counts.get('affected_flows', 0)}")

    obligations = counts.get("obligations", 0)
    obligation_line = f"- **Verification obligations:** {obligations}"
    if obligations:
        obligation_line += (
            f" — {counts.get('obligations_passed', 0)} passed · "
            f"{counts.get('obligations_failed', 0)} failed · "
            f"{counts.get('obligations_unverified', 0)} unverified · "
            f"{counts.get('obligations_unknown', 0)} unknown"
        )
    lines.append(obligation_line)
    lines.append(
        f"- **Tests:** {counts.get('mapped_tests', 0)} mapped · "
        f"{counts.get('tests_executed', 0)} executed by Sydes"
    )

    unresolved = counts.get("unresolved_changed_symbols", 0)
    if unresolved:
        lines.append(
            f"- **Unresolved:** {unresolved} changed symbol(s) with no established impact path"
        )
    lines.append("")


def _render_gaps(result: dict[str, Any], lines: list[str]) -> None:
    gaps = _as_list(_get(result, "verification_gaps", default=[]))
    reasons = [_clean(item) for item in _as_list(_get(result, "summary", "risk_reasons", default=[]))]

    lines.append("### Gaps")
    lines.append("")
    if gaps:
        for gap in gaps[:_MAX_GAPS]:
            behavior = _clean(_get(gap, "behavior", default="(unnamed behavior)"))
            status = _get(gap, "status", default="")
            why = _clean(_get(gap, "why", default=""))
            entry = f"- **{behavior}**"
            if status:
                entry += f" — `{status}`"
            lines.append(entry)
            if why:
                lines.append(f"  - {why}")
        remaining = len(gaps) - _MAX_GAPS
        if remaining > 0:
            lines.append(f"- _…and {remaining} more gap(s) in the full result._")
    elif reasons:
        # No enumerated gaps, but the verdict still has drivers worth showing.
        for reason in reasons:
            lines.append(f"- {reason}")
    else:
        lines.append("No verification gaps were recorded.")
    lines.append("")


def _render_review(result: dict[str, Any], lines: list[str]) -> None:
    findings = _as_list(_get(result, "code_findings", default=[]))

    lines.append("### Review")
    lines.append("")
    if not findings:
        lines.append("No code-review findings were reported.")
        lines.append("")
        return

    for finding in findings[:_MAX_FINDINGS]:
        severity = str(_get(finding, "severity", default="P3"))
        icon = _SEVERITY_ICONS.get(severity, "⚪")
        title = _clean(_get(finding, "title", default="(untitled finding)"))
        location = str(_get(finding, "file", default="") or "")
        line_no = _get(finding, "line")
        if location and isinstance(line_no, int):
            location = f"{location}:{line_no}"
        heading = f"- {icon} **[{severity}]** {title}"
        if location:
            heading += f" — `{location}`"
        lines.append(heading)
        for field in ("explanation", "impact", "suggested_fix"):
            text = _clean(_get(finding, field, default=""))
            if text:
                lines.append(f"  - _{field.replace('_', ' ')}:_ {text}")
    remaining = len(findings) - _MAX_FINDINGS
    if remaining > 0:
        lines.append(f"- _…and {remaining} more finding(s) in the full result._")
    lines.append("")


def _render_details(result: dict[str, Any], lines: list[str]) -> None:
    """Secondary context, collapsed so it never crowds the summary."""
    body: list[str] = []

    impacts = _as_list(_get(result, "accepted_impacts", default=[]))
    if impacts:
        body.append("**Affected behavior**")
        body.append("")
        for impact in impacts[:_MAX_IMPACTS]:
            label = _clean(_get(impact, "behavior_label", default="")) or _clean(
                _get(impact, "label", default="(unlabeled)")
            )
            status = str(_get(impact, "status", default="")).upper()
            entry = f"- `{status}` {label}" if status else f"- {label}"
            confidence = _get(impact, "llm_confidence")
            if isinstance(confidence, (int, float)):
                entry += f" _(confidence {float(confidence):.2f})_"
            body.append(entry)
            reason = _clean(_get(impact, "llm_reason", default=""))
            if reason:
                body.append(f"  - {reason}")
        body.append("")

    symbols = _as_list(_get(result, "change", "symbols", default=[]))
    if symbols:
        body.append("**Changed symbols**")
        body.append("")
        for symbol in symbols[:_MAX_SYMBOLS]:
            name = _get(symbol, "name", default="(unnamed)")
            kind = _get(symbol, "kind", default="symbol")
            path = _get(symbol, "file", default="")
            start = _get(symbol, "start_line")
            location = f"{path}:{start}" if path and isinstance(start, int) else str(path)
            body.append(f"- `{name}` ({kind}) — `{location}`")
        remaining = len(symbols) - _MAX_SYMBOLS
        if remaining > 0:
            body.append(f"- _…and {remaining} more symbol(s)._")
        body.append("")

    notes = [_clean(item) for item in _as_list(_get(result, "analysis_notes", default=[]))]
    if notes:
        body.append("**Analysis completeness**")
        body.append("")
        for note in notes[:_MAX_NOTES]:
            body.append(f"- {note}")
        body.append("")

    dependencies = _as_list(_get(result, "runtime_dependencies", default=[]))
    if dependencies:
        body.append("**Runtime requirements** (Sydes does not provision, mock, or contact these)")
        body.append("")
        for dependency in dependencies:
            name = _get(dependency, "name", default="(unnamed)")
            kind = _get(dependency, "kind", default="")
            body.append(f"- {name}" + (f" (`{kind}`)" if kind else ""))
        body.append("")

    if not body:
        return

    lines.append("<details><summary>More details</summary>")
    lines.append("")
    lines.extend(body)
    lines.append("</details>")
    lines.append("")


def render(result: dict[str, Any], run_url: str | None = None) -> str:
    lines: list[str] = [MARKER, ""]
    _render_header(result, lines)
    _render_change(result, lines)
    _render_verification(result, lines)
    _render_gaps(result, lines)
    _render_review(result, lines)
    _render_details(result, lines)

    lines.append("---")
    footer = "🔎 [Sydes](https://github.com/sydes-ai/sydes) · full JSON result in the run artifact"
    if run_url:
        footer += f" · [view run]({run_url})"
    lines.append(footer)
    return "\n".join(lines).rstrip() + "\n"


def render_unavailable(reason: str, run_url: str | None = None) -> str:
    """Fallback body when there is no result to read — a failed or partial run
    should still leave the reviewer with an explanation rather than silence."""
    lines = [
        MARKER,
        "",
        "## Sydes verification",
        "",
        "| | |",
        "| --- | --- |",
        "| **Verdict** | ℹ️ `NOT PRODUCED` |",
        "",
        f"Sydes did not produce a result for this run: {reason}",
        "",
        "This does not indicate a verdict about the change. See the run log for details.",
        "",
        "---",
    ]
    footer = "🔎 [Sydes](https://github.com/sydes-ai/sydes)"
    if run_url:
        footer += f" · [view run]({run_url})"
    lines.append(footer)
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a Sydes result as Markdown.")
    parser.add_argument("result", type=Path, help="Path to sydes-result.json")
    parser.add_argument("--out", type=Path, required=True, help="Markdown output path")
    parser.add_argument("--diagnostics-out", type=Path, help="Write diagnostics to this JSON path")
    parser.add_argument("--run-url", default=None, help="Actions run URL for the footer link")
    args = parser.parse_args()

    result: dict[str, Any] | None = None
    reason = ""
    try:
        loaded = json.loads(args.result.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            result = loaded
        else:
            reason = "the result file did not contain a JSON object"
    except FileNotFoundError:
        reason = "no result file was written"
    except (OSError, json.JSONDecodeError) as exc:
        reason = f"the result file could not be read ({exc})"

    if result is None:
        args.out.write_text(render_unavailable(reason, args.run_url), encoding="utf-8")
        print(f"Sydes result unavailable: {reason}")
        return 0

    args.out.write_text(render(result, args.run_url), encoding="utf-8")
    print(f"Rendered Sydes comment to {args.out}")

    # Diagnostics are split out here rather than in Sydes itself: the result
    # schema still carries them, and this keeps the split to the presentation
    # layer. Real schema separation belongs in Sydes later.
    if args.diagnostics_out:
        payload = {
            "generated_at": result.get("generated_at"),
            "diagnostics": _as_list(result.get("diagnostics")),
            "notes": _as_list(result.get("notes")),
            "analysis_notes": _as_list(result.get("analysis_notes")),
        }
        args.diagnostics_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote diagnostics to {args.diagnostics_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
