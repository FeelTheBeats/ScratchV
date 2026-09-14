"""Tests for the Topic 24 Spike tool-resolution feature case report.

The report is the CI artifact that proves the six-layer Spike toolchain
resolution, CLI-over-env priority, missing-tool degradation, the
``run_spike_bench.py --probe-spike`` JSON contract, and parser tolerance.
All tests are hermetic: fakes are shell stubs in temporary directories, so
no real Spike installation is ever required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.run_topic24_spike_case import (
    SCHEMA_VERSION,
    build_resolution_matrix,
    evaluate,
    main,
    make_fake_tool,
    run_cli_probe,
    run_degradation_probe,
    run_parser_tolerance,
    run_probe_subprocess,
)
from scratchv.standalone import spike_sim

CASE = "spike_tools_matrix"
ALL_SOURCES = {
    "cli", "env", "spike_home", "path", "common", "legacy", "missing",
}


@pytest.fixture(autouse=True)
def hermetic_env(monkeypatch):
    for var in ("SCRATCHV_SPIKE_BIN", "SCRATCHV_SPIKE_DASM",
                "SCRATCHV_SPIKE_LOG_PARSER", "SCRATCHV_SPIKE_HOME"):
        monkeypatch.delenv(var, raising=False)


def test_resolution_matrix_covers_all_sources_and_structure(tmp_path):
    matrix = build_resolution_matrix(tmp_path)
    by_scenario = {row["scenario"]: row for row in matrix}

    assert ALL_SOURCES <= {row["observed_source"] for row in matrix}
    assert all(row["matches"] for row in matrix)
    assert all(row["structure_ok"] for row in matrix)

    assert by_scenario["clean_missing"]["observed_source"] == "missing"
    assert by_scenario["clean_missing"]["path"] is None
    assert by_scenario["cli_beats_env"]["path"].endswith("spike-cli")
    assert by_scenario["env_beats_path"]["path"].endswith("spike-env")
    assert by_scenario["spike_home"]["path"].endswith("bin/spike")
    assert by_scenario["path"]["path"].endswith("spike-path")
    assert by_scenario["legacy"]["path"].endswith("spike-legacy")

    env_row = by_scenario["env"]
    assert env_row["candidates"][0] == env_row["path"]
    invalid = by_scenario["env_invalid_falls_through"]
    assert invalid["observed_source"] == "missing"
    assert any(spike_sim.ENV_SPIKE_BIN in item
               for item in invalid["warnings"])


def test_env_fake_spike_resolves_env_and_cli_wins(tmp_path):
    fake_env = make_fake_tool(tmp_path, "spike-env")
    fake_cli = make_fake_tool(tmp_path, "spike-cli")

    tools_env = spike_sim.resolve_spike_tools(
        env={spike_sim.ENV_SPIKE_BIN: str(fake_env)},
        which=lambda name: None, common_dirs=())
    assert tools_env.spike == str(fake_env)
    assert tools_env.sources["spike"] == "env"

    tools_cli = spike_sim.resolve_spike_tools(
        cli_spike=str(fake_cli),
        env={spike_sim.ENV_SPIKE_BIN: str(fake_env)},
        which=lambda name: None, common_dirs=())
    assert tools_cli.spike == str(fake_cli)
    assert tools_cli.sources["spike"] == "cli"


def test_cli_probe_via_main_executes_cli_tool(tmp_path):
    probe = run_cli_probe(tmp_path)

    assert probe["exit_code"] == 0
    assert probe["status"] == "ok"
    assert probe["resolved_source"] == "cli"
    assert probe["resolved_path"] == probe["cli_path"]
    assert probe["resolved_path"].endswith("spike-cli")
    assert probe["committed_insns"] == 1234
    assert probe["stdout_is_json"]


def test_missing_tool_degrades_to_skip_and_strict_config_error(tmp_path):
    probe = run_degradation_probe(tmp_path)

    assert probe["skip_exit_code"] == 0
    assert "SKIP:" in probe["skip_stderr"]
    skip_json = probe["skip_json"]
    assert skip_json["status"] == "skipped"
    assert skip_json["exit_code"] == -2
    assert skip_json["spike_tools"]["spike"]["source"] == "missing"
    assert skip_json["spike_tools"]["spike"]["path"] is None

    assert probe["strict_exit_code"] == 2
    assert probe["strict_stdout_empty"]
    assert probe["elf_created"] is False


def test_probe_spike_subprocess_reports_spike_tools(tmp_path):
    probe = run_probe_subprocess(tmp_path)

    assert probe["exit_code"] == 0
    assert probe["report_ok"]
    assert probe["backend"] == {"kind": "emulator", "spike_style": True}
    tools = probe["spike_tools"]
    for name in ("spike", "spike_dasm", "spike_log_parser"):
        assert set(tools[name]) == {"path", "source", "candidates"}
    assert isinstance(tools["warnings"], list)


def test_parse_helpers_tolerate_thousands_and_missing_sections():
    tolerance = run_parser_tolerance()

    assert tolerance["committed"] == 2_000_000
    assert tolerance["mips"] == 42.0
    assert tolerance["cache_hits"] == 10_000
    assert tolerance["cache_misses"] == 25
    assert tolerance["cache_miss_rate"] == 0.25
    assert tolerance["histogram"] == {
        "0x80000014": 123, "0x80000018": 456}
    assert tolerance["messy_histogram"] == {
        "0x80000010": 7, "0x80000020": 9}
    assert tolerance["empty_cache_is_zeroed"]
    assert tolerance["partial_status"] == "ok"
    assert tolerance["partial_committed"] == 5
    assert len(tolerance["partial_parse_warnings"]) == 2


def test_main_writes_json_and_markdown(tmp_path, capsys):
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"

    exit_code = main([
        "--case", CASE,
        "--json", str(json_path),
        "--markdown", str(markdown_path),
    ])

    assert exit_code == 0
    data = json.loads(json_path.read_text())
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["topic"] == "topic24-spike-tools"
    assert data["hard_failures"] == []
    assert all(data["hard_checks"].values())
    markdown = markdown_path.read_text()
    assert "Topic 24 Spike Tool-Resolution Feature Case" in markdown
    assert "Resolution matrix" in markdown
    assert "Honesty" in markdown
    assert capsys.readouterr().out

    with pytest.raises(SystemExit) as excinfo:
        main(["--case", "no_such_case"])
    assert excinfo.value.code == 2


def test_hard_check_gate_is_not_vacuous(tmp_path, monkeypatch):
    """A broken matrix must surface as hard failures and exit code 1."""
    def broken_matrix(_root):
        return [{
            "scenario": "clean_missing",
            "expected_source": "missing",
            "observed_source": "path",
            "path": "/usr/bin/spike",
            "path_exists": True,
            "candidates": [],
            "warnings": [],
            "structure_ok": True,
            "matches": False,
        }]

    monkeypatch.setattr(
        "benchmarks.run_topic24_spike_case.build_resolution_matrix",
        broken_matrix)

    report = evaluate(tmp_path)
    assert "resolution_matrix_all_rows_match" in report["hard_failures"]
    assert "resolution_covers_path_env_home_common_legacy_missing" in (
        report["hard_failures"])

    exit_code = main([
        "--case", CASE,
        "--json", str(tmp_path / "broken.json"),
        "--markdown", str(tmp_path / "broken.md"),
    ])
    assert exit_code == 1
    assert json.loads((tmp_path / "broken.json").read_text())["hard_failures"]
