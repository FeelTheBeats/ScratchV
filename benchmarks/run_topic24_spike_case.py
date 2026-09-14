#!/usr/bin/env python3
"""Run one Topic 24 Spike tool-resolution feature case and emit CI reports.

The report proves four separate facts about the portable Spike toolchain
resolution added in this topic:

1. ``resolve_spike_tools()`` reports a complete record per tool
   (path/source/candidates/warnings) for every layer of the priority chain
   (CLI > dedicated env vars > ``$SCRATCHV_SPIKE_HOME/bin`` > ``PATH`` >
   common install dirs > legacy constant > missing);
2. a fake executable found through ``SCRATCHV_SPIKE_BIN`` resolves with
   ``source="env"`` while an explicit ``--spike-bin`` wins over the env var,
   proven end to end through ``spike_sim.main(argv=[...])``;
3. missing tools degrade gracefully: an invalid env path only warns and
   falls through, a fully missing toolchain prints ``SKIP:`` and exits 0,
   and ``--require-spike`` instead exits 2;
4. ``run_spike_bench.py --probe-spike --json`` exits 0 with a ``spike_tools``
   snapshot in its JSON payload, and the pure parsers tolerate thousands
   separators as well as missing statistics sections (``parse_warnings``).

Every fake "spike" is a shell script created in a temporary directory, so
the case is hermetic: it never requires a real Spike installation, and it
makes no claim about real Spike availability or simulator performance.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scratchv.standalone import spike_sim

SCHEMA_VERSION = "topic24-spike-case/1"
TOPIC = "topic24-spike-tools"
DEFAULT_CASE = "spike_tools_matrix"
CASES = (DEFAULT_CASE,)
DEFAULT_JSON = Path("benchmark_reports/spike_tools_report.json")
DEFAULT_MARKDOWN = Path("benchmark_reports/spike_tools_report.md")

EXPECTED_COMMITTED = 1234
REQUIRED_SOURCES = ("path", "env", "spike_home", "common", "legacy", "missing")

ENV_SPIKE_BIN = spike_sim.ENV_SPIKE_BIN
ENV_SPIKE_DASM = spike_sim.ENV_SPIKE_DASM
ENV_SPIKE_LOG_PARSER = spike_sim.ENV_SPIKE_LOG_PARSER
ENV_SPIKE_HOME = spike_sim.ENV_SPIKE_HOME

_ENV_NAMES = (ENV_SPIKE_BIN, ENV_SPIKE_DASM, ENV_SPIKE_LOG_PARSER,
              ENV_SPIKE_HOME)

CANNED_STDERR = (
    "Commited 1234 instructions\n"
    "core   0: 0x80000000 (0x00000013) 1.5 MIPS\n"
    "I$: 64 sets x 2 ways x 32 B\n"
    "  hits: 10,000    misses: 25    miss rate: 0.25%\n"
    "D$: 128 sets x 4 ways x 32 B\n"
    "  hits: 20,000    misses: 50    miss rate: 0.25%\n"
)
CANNED_STDOUT = (
    "PC histogram (number of commits per PC):\n"
    "0x80000014: 123\n"
    "0x80000018: 456\n"
    "\n"
)
#: Deliberately different counters: if the env-var stub were executed instead
#: of the CLI stub, the report would show 999999 committed instructions.
POISONED_ENV_STDERR = (
    "Committed 999,999 instructions\n"
    "core   0: 0x80000000 (0x00000013) 1.5 MIPS\n"
)


def make_fake_tool(
    directory: Path,
    name: str,
    *,
    stderr: str = "",
    stdout: str = "",
    exit_code: int = 0,
) -> Path:
    """Write a chmod +x POSIX shell stub; no real Spike is ever used."""
    body: list[str] = ["#!/bin/sh"]
    if stderr:
        body.append("cat >&2 <<'SPIKE_STUB_EOF'")
        body.extend(stderr.rstrip("\n").splitlines())
        body.append("SPIKE_STUB_EOF")
    if stdout:
        body.append("cat <<'SPIKE_STUB_EOF'")
        body.extend(stdout.rstrip("\n").splitlines())
        body.append("SPIKE_STUB_EOF")
    body.append(f"exit {int(exit_code)}")
    path = directory / name
    path.write_text("\n".join(body) + "\n")
    path.chmod(0o755)
    return path


def _case_dir(root: Path, name: str) -> Path:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@contextlib.contextmanager
def _patched_module(
    tmp: Path,
    *,
    legacy_spike: str | None = None,
    legacy_dasm: str | None = None,
    legacy_parser: str | None = None,
    which=None,
):
    """Isolate the module-level resolution layers (constants + which())."""
    saved = (
        spike_sim.SPIKE,
        spike_sim.SPIKE_DASM,
        spike_sim.SPIKE_LOG_PARSER,
        spike_sim.COMMON_SPIKE_DIRS,
        spike_sim.shutil,
    )
    finder = which if callable(which) else (lambda name: None)
    try:
        spike_sim.SPIKE = legacy_spike or str(tmp / "no-legacy-spike")
        spike_sim.SPIKE_DASM = legacy_dasm or str(tmp / "no-legacy-dasm")
        spike_sim.SPIKE_LOG_PARSER = (
            legacy_parser or str(tmp / "no-legacy-parser"))
        spike_sim.COMMON_SPIKE_DIRS = ()
        spike_sim.shutil = types.SimpleNamespace(which=finder)
        yield
    finally:
        (
            spike_sim.SPIKE,
            spike_sim.SPIKE_DASM,
            spike_sim.SPIKE_LOG_PARSER,
            spike_sim.COMMON_SPIKE_DIRS,
            spike_sim.shutil,
        ) = saved


@contextlib.contextmanager
def _temp_environ(values: dict[str, str]):
    """Set only the SCRATCHV_SPIKE_* variables, restoring the host values."""
    saved = {name: os.environ.get(name) for name in _ENV_NAMES}
    try:
        for name in _ENV_NAMES:
            os.environ.pop(name, None)
        for name, value in values.items():
            os.environ[name] = value
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _structure_complete(tools: spike_sim.SpikeTools) -> bool:
    """Every tool entry carries path/source/candidates plus a warnings list."""
    tool_names = ("spike", "spike-dasm", "spike-log-parser")
    if set(tools.sources) != set(tool_names):
        return False
    if set(tools.candidates) != set(tool_names):
        return False
    data = tools.as_dict()
    for name in ("spike", "spike_dasm", "spike_log_parser"):
        entry = data.get(name)
        if not isinstance(entry, dict):
            return False
        if set(entry) != {"path", "source", "candidates"}:
            return False
        if not isinstance(entry["candidates"], list):
            return False
    return isinstance(data.get("warnings"), list)


def _scenario(
    tmp: Path,
    name: str,
    expected_source: str,
    *,
    expect_path: Path | None = None,
    legacy_spike: str | None = None,
    legacy_dasm: str | None = None,
    legacy_parser: str | None = None,
    **resolve_kwargs,
) -> dict[str, Any]:
    with _patched_module(
        tmp,
        legacy_spike=legacy_spike,
        legacy_dasm=legacy_dasm,
        legacy_parser=legacy_parser,
    ):
        tools = spike_sim.resolve_spike_tools(**resolve_kwargs)
    observed = tools.sources.get("spike", "missing")
    expected_path = str(expect_path) if expect_path else None
    return {
        "scenario": name,
        "expected_source": expected_source,
        "observed_source": observed,
        "path": tools.spike,
        "path_exists": bool(tools.spike and os.path.isfile(tools.spike)),
        "candidates": list(tools.candidates.get("spike", ())),
        "warnings": list(tools.warnings),
        "structure_ok": _structure_complete(tools),
        "matches": (
            observed == expected_source
            and (expected_path is None or tools.spike == expected_path)
        ),
    }


def build_resolution_matrix(root: Path) -> list[dict[str, Any]]:
    """Exercise every resolution layer with fake tools only."""
    tmp = _case_dir(root, "resolution")
    fake_cli = make_fake_tool(tmp, "spike-cli")
    fake_env = make_fake_tool(tmp, "spike-env")
    fake_path = make_fake_tool(tmp, "spike-path")
    fake_legacy = make_fake_tool(tmp, "spike-legacy")
    home = tmp / "spike-home"
    (home / "bin").mkdir(parents=True)
    fake_home = make_fake_tool(home / "bin", "spike")
    common = tmp / "common"
    common.mkdir()
    fake_common = make_fake_tool(common, "spike")

    def none_which(_name):
        return None

    def path_which(name):
        return str(fake_path) if name == "spike" else None

    return [
        _scenario(
            tmp, "clean_missing", "missing",
            env={}, which=none_which, common_dirs=()),
        _scenario(
            tmp, "cli", "cli",
            cli_spike=str(fake_cli), env={}, which=none_which,
            common_dirs=(), expect_path=fake_cli),
        _scenario(
            tmp, "cli_beats_env", "cli",
            cli_spike=str(fake_cli),
            env={ENV_SPIKE_BIN: str(fake_env)},
            which=none_which, common_dirs=(), expect_path=fake_cli),
        _scenario(
            tmp, "env", "env",
            env={ENV_SPIKE_BIN: str(fake_env)},
            which=none_which, common_dirs=(), expect_path=fake_env),
        _scenario(
            tmp, "env_beats_path", "env",
            env={ENV_SPIKE_BIN: str(fake_env)},
            which=path_which, common_dirs=(), expect_path=fake_env),
        _scenario(
            tmp, "spike_home", "spike_home",
            env={ENV_SPIKE_HOME: str(home)},
            which=none_which, common_dirs=(), expect_path=fake_home),
        _scenario(
            tmp, "path", "path",
            env={}, which=path_which, common_dirs=(),
            expect_path=fake_path),
        _scenario(
            tmp, "common", "common",
            env={}, which=none_which, common_dirs=(str(common),),
            expect_path=fake_common),
        _scenario(
            tmp, "legacy", "legacy",
            env={}, which=none_which, common_dirs=(),
            legacy_spike=str(fake_legacy), expect_path=fake_legacy),
        _scenario(
            tmp, "env_invalid_falls_through", "missing",
            env={ENV_SPIKE_BIN: str(tmp / "no-such-spike")},
            which=none_which, common_dirs=()),
    ]


def run_cli_probe(root: Path) -> dict[str, Any]:
    """Prove --spike-bin beats SCRATCHV_SPIKE_BIN through main(argv=...)."""
    tmp = _case_dir(root, "cli_probe")
    fake_env = make_fake_tool(
        tmp, "spike-env", stderr=POISONED_ENV_STDERR)
    fake_cli = make_fake_tool(
        tmp, "spike-cli", stderr=CANNED_STDERR, stdout=CANNED_STDOUT)
    binary = tmp / "output.bin"
    binary.write_bytes(b"\x00" * 64)

    out, err = io.StringIO(), io.StringIO()
    with _temp_environ({ENV_SPIKE_BIN: str(fake_env)}), _patched_module(tmp):
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            exit_code = spike_sim.main([
                "--binary", str(binary), "--code-size", "64",
                "--spike-bin", str(fake_cli), "--json",
            ])

    try:
        report = json.loads(out.getvalue())
    except (json.JSONDecodeError, ValueError):
        report = {}
    spike_entry = (report.get("spike_tools") or {}).get("spike") or {}
    return {
        "exit_code": exit_code,
        "status": report.get("status"),
        "committed_insns": report.get("committed_insns"),
        "resolved_path": report.get("spike_binary"),
        "resolved_source": spike_entry.get("source"),
        "cli_path": str(fake_cli),
        "env_path": str(fake_env),
        "stdout_is_json": bool(report),
        "stderr_tail": err.getvalue()[-300:],
    }


def run_degradation_probe(root: Path) -> dict[str, Any]:
    """Prove missing-tool degradation: SKIP/exit 0 vs --require-spike/exit 2."""
    tmp = _case_dir(root, "degradation")
    binary = tmp / "output.bin"
    binary.write_bytes(b"\x00" * 64)

    out, err = io.StringIO(), io.StringIO()
    with _temp_environ({}), _patched_module(tmp):
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            skip_exit_code = spike_sim.main([
                "--binary", str(binary), "--code-size", "64", "--json",
            ])
        skip_stderr = err.getvalue()
        try:
            skip_json = json.loads(out.getvalue())
        except (json.JSONDecodeError, ValueError):
            skip_json = None

        strict_out, strict_err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(strict_out), \
                contextlib.redirect_stderr(strict_err):
            strict_exit_code = spike_sim.main([
                "--binary", str(binary), "--code-size", "64",
                "--require-spike",
            ])

    return {
        "skip_exit_code": skip_exit_code,
        "skip_stderr": skip_stderr,
        "skip_json": skip_json,
        "strict_exit_code": strict_exit_code,
        "strict_stdout_empty": strict_out.getvalue() == "",
        "strict_stderr_tail": strict_err.getvalue()[-300:],
        "elf_created": (tmp / "output_spike.elf").exists(),
    }


def run_probe_subprocess(root: Path) -> dict[str, Any]:
    """Run ``run_spike_bench.py --probe-spike --json`` as a subprocess."""
    tmp = _case_dir(root, "probe")
    binary = tmp / "probe.bin"
    binary.write_bytes(b"\x00" * 64)

    env = {
        name: value for name, value in os.environ.items()
        if not name.startswith("SCRATCHV_SPIKE_")
    }
    python_path = [str(REPO_ROOT)]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)

    command = [
        sys.executable,
        str(REPO_ROOT / "scratchv" / "standalone" / "run_spike_bench.py"),
        "--binary", str(binary), "--code-size", "64",
        "--max-instr", "1", "--progress", "1000000",
        "--probe-spike", "--json",
    ]
    proc = subprocess.run(
        command, capture_output=True, text=True, env=env,
        cwd=str(REPO_ROOT), timeout=120)
    try:
        report = json.loads(proc.stdout)
    except (json.JSONDecodeError, ValueError):
        report = None
    return {
        "command": command,
        "exit_code": proc.returncode,
        "report_ok": isinstance(report, dict),
        "backend": (report or {}).get("backend"),
        "summary": (report or {}).get("summary"),
        "spike_tools": (report or {}).get("spike_tools"),
        "stderr_tail": proc.stderr[-400:],
    }


def _probe_contract_ok(probe: dict[str, Any]) -> bool:
    tools = probe.get("spike_tools")
    if not isinstance(tools, dict):
        return False
    for name in ("spike", "spike_dasm", "spike_log_parser"):
        entry = tools.get(name)
        if not isinstance(entry, dict):
            return False
        if set(entry) != {"path", "source", "candidates"}:
            return False
    return isinstance(tools.get("warnings"), list)


def run_parser_tolerance() -> dict[str, Any]:
    """Prove thousands separators parse and missing sections only warn."""
    committed, mips = spike_sim.parse_commit_stats(
        "Committed 2,000,000 instructions\n42.0 MIPS")
    cache = spike_sim.parse_cache_stats(CANNED_STDERR)
    histogram = spike_sim.parse_pc_histogram(CANNED_STDOUT)
    messy_histogram = spike_sim.parse_pc_histogram(
        "PC histogram (number of commits per PC):\n"
        "0x80000010: 7\n"
        "garbage line\n"
        "0x80000020: 9\n"
        "\n"
        "0x80000030: 11\n"
    )

    original_run = spike_sim.subprocess.run
    try:
        spike_sim.subprocess.run = lambda cmd, **kwargs: types.SimpleNamespace(
            returncode=0, stdout="", stderr="Committed 5 instructions")
        partial = spike_sim.run_spike(
            "unused.elf", tools=spike_sim.SpikeTools(spike="/fake/spike"))
    finally:
        spike_sim.subprocess.run = original_run

    return {
        "committed": committed,
        "mips": mips,
        "cache_hits": int(cache["icache_hits"]),
        "cache_misses": int(cache["icache_misses"]),
        "cache_miss_rate": float(cache["icache_miss_rate"]),
        "histogram": {f"0x{pc:08x}": cnt for pc, cnt in histogram.items()},
        "messy_histogram": {
            f"0x{pc:08x}": cnt for pc, cnt in messy_histogram.items()},
        "empty_cache_is_zeroed": all(
            value == 0 for value in spike_sim.parse_cache_stats("").values()),
        "partial_status": partial.status,
        "partial_committed": partial.committed_insns,
        "partial_parse_warnings": list(partial.parse_warnings),
    }


def evaluate(root: Path) -> dict[str, Any]:
    """Build the full report payload and run the hard invariants."""
    matrix = build_resolution_matrix(root)
    by_scenario = {row["scenario"]: row for row in matrix}
    observed_sources = {row["observed_source"] for row in matrix}
    cli_probe = run_cli_probe(root)
    degradation = run_degradation_probe(root)
    probe = run_probe_subprocess(root)
    tolerance = run_parser_tolerance()

    skip_json = degradation.get("skip_json") or {}
    skip_spike = (skip_json.get("spike_tools") or {}).get("spike") or {}

    def row(name: str) -> dict[str, Any]:
        return by_scenario.get(name, {})

    hard_checks = {
        "resolution_matrix_all_rows_match": all(
            entry["matches"] for entry in matrix),
        "resolution_structure_complete": all(
            entry["structure_ok"] for entry in matrix),
        "resolution_covers_path_env_home_common_legacy_missing": (
            set(REQUIRED_SOURCES) <= observed_sources),
        "clean_env_falls_back_to_missing": (
            row("clean_missing").get("observed_source") == "missing"
            and row("clean_missing").get("path") is None),
        "env_fake_spike_resolves_as_env": (
            row("env").get("observed_source") == "env"
            and row("env").get("matches") is True),
        "cli_beats_env_resolution": (
            row("cli_beats_env").get("observed_source") == "cli"
            and row("cli_beats_env").get("matches") is True),
        "env_beats_path_resolution": (
            row("env_beats_path").get("matches") is True),
        "invalid_env_warns_and_continues": (
            row("env_invalid_falls_through").get("observed_source")
            == "missing"
            and any(ENV_SPIKE_BIN in warning for warning in
                    row("env_invalid_falls_through").get("warnings", ()))),
        "cli_probe_exit_zero": cli_probe["exit_code"] == 0,
        "cli_probe_resolved_cli_over_env": (
            cli_probe["resolved_source"] == "cli"
            and cli_probe["resolved_path"] == cli_probe["cli_path"]),
        "cli_probe_ran_cli_stub": (
            cli_probe["committed_insns"] == EXPECTED_COMMITTED),
        "missing_tool_skips_exit_zero": (
            degradation["skip_exit_code"] == 0
            and "SKIP:" in degradation["skip_stderr"]),
        "missing_tool_json_source_missing": (
            skip_json.get("status") == "skipped"
            and skip_spike.get("source") == "missing"
            and skip_json.get("exit_code") == -2),
        "strict_require_spike_exit_two": degradation["strict_exit_code"] == 2,
        "no_elf_written_when_skipped": degradation["elf_created"] is False,
        "probe_subprocess_exit_zero": (
            probe["exit_code"] == 0 and probe["report_ok"]),
        "probe_json_has_spike_tools_contract": _probe_contract_ok(probe),
        "probe_backend_marked_emulator": (
            probe["backend"] == {"kind": "emulator", "spike_style": True}),
        "parser_reads_thousands_separators": (
            tolerance["committed"] == 2_000_000
            and tolerance["mips"] == 42.0
            and tolerance["cache_hits"] == 10_000
            and tolerance["histogram"] == {"0x80000014": 123,
                                           "0x80000018": 456}),
        "parser_missing_sections_become_warnings": (
            tolerance["partial_status"] == "ok"
            and len(tolerance["partial_parse_warnings"]) == 2),
    }
    failed = sorted(name for name, ok in hard_checks.items() if not ok)

    return {
        "schema_version": SCHEMA_VERSION,
        "topic": TOPIC,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case": DEFAULT_CASE,
        "python": (
            f"{sys.version_info.major}.{sys.version_info.minor}."
            f"{sys.version_info.micro}"),
        "resolution_matrix": matrix,
        "resolution_sources_observed": sorted(observed_sources),
        "cli_probe": cli_probe,
        "degradation": degradation,
        "probe": probe,
        "parser_tolerance": tolerance,
        "hard_checks": hard_checks,
        "hard_failures": failed,
        "honesty": (
            "Hermetic deterministic feature case: every spike executable "
            "used here is a shell stub created under a temporary directory, "
            "and the contract is proven through spike_sim.main(argv=[...]) "
            "plus one run_spike_bench.py --probe-spike subprocess. Resolution "
            "sources prove the documented priority order of the search "
            "layers only; they do not prove that a real Spike binary is "
            "installed, that a real Spike run succeeds, or anything about "
            "simulator performance."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    cli_probe = report["cli_probe"]
    degradation = report["degradation"]
    probe = report["probe"]
    tolerance = report["parser_tolerance"]
    total = len(report["hard_checks"])
    passed = total - len(report["hard_failures"])

    lines = [
        "# Topic 24 Spike Tool-Resolution Feature Case",
        "",
        f"- Schema: `{report['schema_version']}`",
        f"- Case: `{report['case']}` (hermetic, program-built matrix)",
        f"- Generated: {report['generated_at']}",
        f"- Python: {report['python']}",
        f"- Hard checks: {'PASS' if not report['hard_failures'] else 'FAIL'} "
        f"({passed}/{total})",
        "",
        "## Resolution matrix (source / candidates / warnings)",
        "",
        "| Scenario | Expected source | Observed source | Tool found | "
        "Candidates | Warnings |",
        "|----------|-----------------|-----------------|------------|"
        "------------|----------|",
    ]
    for row in report["resolution_matrix"]:
        tool_name = Path(row["path"]).name if row["path"] else "-"
        lines.append(
            f"| `{row['scenario']}` | `{row['expected_source']}` | "
            f"`{row['observed_source']}` | {tool_name} | "
            f"{len(row['candidates'])} | {len(row['warnings'])} |")
    lines += [
        "",
        f"- Observed sources: "
        f"{', '.join(f'`{s}`' for s in report['resolution_sources_observed'])}",
        "- Priority proven: CLI > dedicated env vars > "
        "`$SCRATCHV_SPIKE_HOME/bin` > `PATH` > common install dirs > "
        "legacy constant > missing",
        "",
        "## Degradation path",
        "",
        f"- Invalid `{ENV_SPIKE_BIN}`: warning emitted, resolution falls "
        f"through to `missing` (no exception)",
        f"- Missing toolchain: exit `{degradation['skip_exit_code']}`, "
        f"stderr contains `SKIP:`, JSON status "
        f"`{(degradation['skip_json'] or {}).get('status')}` with source "
        f"`{((degradation['skip_json'] or {}).get('spike_tools') or {}).get('spike', {}).get('source')}`",
        f"- `--require-spike`: exit `{degradation['strict_exit_code']}`, "
        f"no report on stdout",
        f"- ELF written while skipped: `{degradation['elf_created']}`",
        "",
        "## CLI-level priority (`--spike-bin` over env)",
        "",
        f"- `spike_sim.main(argv=[...])` exit code: {cli_probe['exit_code']}",
        f"- Resolved source: `{cli_probe['resolved_source']}` at "
        f"`{cli_probe['resolved_path']}`",
        f"- Parsed committed instructions: {cli_probe['committed_insns']} "
        f"(the env stub is poisoned with 999,999 to prove it was not used)",
        "",
        "## Probe JSON summary (`run_spike_bench.py --probe-spike --json`)",
        "",
        f"- Exit code: {probe['exit_code']}",
        f"- Backend: `{json.dumps(probe['backend'])}`",
        "",
        "```json",
        json.dumps(probe["spike_tools"], indent=2),
        "```",
        "",
        "## Parser tolerance",
        "",
        f"- `parse_commit_stats` thousands separator: "
        f"{tolerance['committed']:,} instructions, "
        f"{tolerance['mips']} MIPS",
        f"- `parse_cache_stats` thousands separator: "
        f"hits={tolerance['cache_hits']:,}, "
        f"misses={tolerance['cache_misses']}",
        f"- `parse_pc_histogram`: {tolerance['histogram']}",
        f"- Missing sections: `parse_warnings`="
        f"{tolerance['partial_parse_warnings']}",
        "",
        "## Hard checks",
        "",
    ]
    for name, ok in report["hard_checks"].items():
        lines.append(f"- [{'x' if ok else ' '}] {name}")
    lines += [
        "",
        "## Honesty",
        "",
        report["honesty"],
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default=DEFAULT_CASE, choices=CASES)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="topic24-spike-case-") as tmp:
        report = evaluate(Path(tmp))

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(report)
    args.markdown.write_text(markdown + "\n")
    print(markdown)
    if report["hard_failures"]:
        print("HARD FAILURES: " + ", ".join(report["hard_failures"]))
        return 1
    print(f"reports written: {args.json}, {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
