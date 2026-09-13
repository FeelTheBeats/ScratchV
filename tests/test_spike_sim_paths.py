"""Topic 24: portable Spike toolchain resolution and graceful degradation.

All tests are hermetic: they never require a real Spike installation.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

from scratchv.standalone import spike_sim

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_fake_tool(tmp_path: Path, name: str) -> Path:
    p = tmp_path / name
    p.write_text("#!/bin/sh\nexit 0\n")
    p.chmod(0o755)
    return p


@pytest.fixture(autouse=True)
def hermetic_env(monkeypatch, tmp_path):
    for var in ("SCRATCHV_SPIKE_BIN", "SCRATCHV_SPIKE_DASM",
                "SCRATCHV_SPIKE_LOG_PARSER", "SCRATCHV_SPIKE_HOME"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(spike_sim, "COMMON_SPIKE_DIRS", ())
    monkeypatch.setattr(spike_sim, "SPIKE", str(tmp_path / "legacy-spike"))
    monkeypatch.setattr(spike_sim, "SPIKE_DASM", str(tmp_path / "legacy-dasm"))
    monkeypatch.setattr(spike_sim, "SPIKE_LOG_PARSER",
                        str(tmp_path / "legacy-parser"))
    monkeypatch.setattr(shutil, "which", lambda name: None)


CANNED_STDERR = """\
Commited 1234 instructions
core   0: 0x80000000 (0x00000013) 1.5 MIPS
I$: 64 sets × 2 ways × 32 B
  hits: 10,000    misses: 25    miss rate: 0.25%
D$: 128 sets × 4 ways × 32 B
  hits: 20,000    misses: 50    miss rate: 0.25%
"""
CANNED_STDOUT = """\
PC histogram (number of commits per PC):
0x80000014: 123
0x80000018: 456

"""


# ── Import / CLI hygiene ────────────────────────────────────────────────────

def test_import_works_without_spike(tmp_path):
    env = {"PATH": str(tmp_path), "HOME": str(tmp_path),
           "PYTHONPATH": str(REPO_ROOT)}
    proc = subprocess.run(
        [sys.executable, "-c",
         "import scratchv.standalone.spike_sim as s; print(s.SPIKE)"],
        capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as excinfo:
        spike_sim.main(["--help"])

    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--spike-bin", "--spike-dasm", "--spike-log-parser",
                 "--require-spike"):
        assert flag in out


# ── Missing-spike degradation ───────────────────────────────────────────────

def test_missing_spike_skips_with_reason(tmp_path, capsys):
    binary = tmp_path / "output.bin"
    binary.write_bytes(b"\x00" * 64)

    rc = spike_sim.main(["--binary", str(binary), "--code-size", "64"])

    assert rc == spike_sim.EXIT_OK
    err = capsys.readouterr().err
    assert "SKIP:" in err
    assert "spike binary not found" in err
    assert "SCRATCHV_SPIKE_BIN" in err
    assert "SCRATCHV_SPIKE_HOME/bin/spike" in err
    assert "legacy" in err
    assert not (tmp_path / "output_spike.elf").exists()


def test_missing_spike_strict_returns_config_error(tmp_path, capsys):
    binary = tmp_path / "output.bin"
    binary.write_bytes(b"\x00" * 64)

    rc = spike_sim.main(["--binary", str(binary), "--code-size", "64",
                         "--require-spike"])

    assert rc == spike_sim.EXIT_CONFIG
    err = capsys.readouterr().err
    assert "ERROR:" in err
    assert "--require-spike" in err
    assert not (tmp_path / "output_spike.elf").exists()


def test_missing_spike_json_report_fields(tmp_path, capsys):
    binary = tmp_path / "output.bin"
    binary.write_bytes(b"\x00" * 64)

    rc = spike_sim.main(["--binary", str(binary), "--code-size", "64", "--json"])

    assert rc == spike_sim.EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "skipped"
    assert report["skip_reason"] == "spike binary not found"
    assert report["spike_binary"] is None
    for key in ("binary", "code_size", "static_insns", "max_instr",
                "committed_insns", "wall_time_s", "exit_code", "icache",
                "dcache", "top_pcs", "stderr_tail", "parse_warnings",
                "tool_warnings", "spike_tools"):
        assert key in report
    assert report["spike_tools"]["spike"]["source"] == "missing"
    assert report["spike_tools"]["spike"]["path"] is None


# ── Resolution priority chain ───────────────────────────────────────────────

def test_resolution_cli_over_env(tmp_path, monkeypatch):
    fake_env = make_fake_tool(tmp_path, "spike-env")
    fake_cli = make_fake_tool(tmp_path, "spike-cli")
    monkeypatch.setenv("SCRATCHV_SPIKE_BIN", str(fake_env))

    tools_env = spike_sim.resolve_spike_tools()
    assert tools_env.spike == str(fake_env)
    assert tools_env.sources["spike"] == "env"

    tools_cli = spike_sim.resolve_spike_tools(cli_spike=str(fake_cli))
    assert tools_cli.spike == str(fake_cli)
    assert tools_cli.sources["spike"] == "cli"


def test_resolution_env_over_path(tmp_path, monkeypatch):
    fake_env = make_fake_tool(tmp_path, "spike-env")
    monkeypatch.setenv("SCRATCHV_SPIKE_BIN", str(fake_env))

    tools = spike_sim.resolve_spike_tools()
    assert tools.spike == str(fake_env)
    assert tools.sources["spike"] == "env"

    monkeypatch.delenv("SCRATCHV_SPIKE_BIN")
    fake_path = make_fake_tool(tmp_path, "spike-path")
    monkeypatch.setattr(
        shutil, "which",
        lambda name: str(fake_path) if name == "spike" else None)

    tools = spike_sim.resolve_spike_tools()
    assert tools.spike == str(fake_path)
    assert tools.sources["spike"] == "path"


def test_resolution_spike_home_and_common(tmp_path, monkeypatch):
    home = tmp_path / "spike-home"
    (home / "bin").mkdir(parents=True)
    fake_home_spike = make_fake_tool(home / "bin", "spike")
    monkeypatch.setenv("SCRATCHV_SPIKE_HOME", str(home))

    tools = spike_sim.resolve_spike_tools()
    assert tools.spike == str(fake_home_spike)
    assert tools.sources["spike"] == "spike_home"

    monkeypatch.delenv("SCRATCHV_SPIKE_HOME")
    common = tmp_path / "common"
    common.mkdir()
    fake_common_spike = make_fake_tool(common, "spike")
    monkeypatch.setattr(spike_sim, "COMMON_SPIKE_DIRS", (str(common),))

    tools = spike_sim.resolve_spike_tools()
    assert tools.spike == str(fake_common_spike)
    assert tools.sources["spike"] == "common"


def test_resolution_legacy_constant(tmp_path, monkeypatch):
    fake_legacy = make_fake_tool(tmp_path, "legacy-spike")
    monkeypatch.setattr(spike_sim, "SPIKE", str(fake_legacy))

    tools = spike_sim.resolve_spike_tools()

    assert tools.spike == str(fake_legacy)
    assert tools.sources["spike"] == "legacy"


# ── Invalid explicit paths ──────────────────────────────────────────────────

def test_cli_invalid_path_raises(tmp_path):
    with pytest.raises(spike_sim.SpikeConfigError) as ei:
        spike_sim.resolve_spike_tools(cli_spike=str(tmp_path / "nope"))
    assert "--spike-bin" in str(ei.value)

    with pytest.raises(spike_sim.SpikeConfigError) as ei:
        spike_sim.resolve_spike_tools(cli_dasm=str(tmp_path / "nope"))
    assert "--spike-dasm" in str(ei.value)


def test_cli_invalid_path_returns_config_error(tmp_path, capsys):
    binary = tmp_path / "output.bin"
    binary.write_bytes(b"\x00" * 64)

    rc = spike_sim.main(["--binary", str(binary), "--code-size", "64",
                         "--spike-bin", str(tmp_path / "nope")])

    assert rc == spike_sim.EXIT_CONFIG
    assert "ERROR:" in capsys.readouterr().err
    assert not (tmp_path / "output_spike.elf").exists()


def test_env_invalid_path_warns_and_falls_through(tmp_path, monkeypatch):
    monkeypatch.setenv("SCRATCHV_SPIKE_BIN", str(tmp_path / "nope"))

    tools = spike_sim.resolve_spike_tools()

    assert tools.spike is None
    assert tools.sources["spike"] == "missing"
    assert any("SCRATCHV_SPIKE_BIN" in w for w in tools.warnings)


# ── Pure parsing helpers ────────────────────────────────────────────────────

def test_parse_commit_stats():
    committed, mips = spike_sim.parse_commit_stats(CANNED_STDERR)
    assert (committed, mips) == (1234, 1.5)

    corrected, corrected_mips = spike_sim.parse_commit_stats(
        "Committed 2,000,000 instructions\n42.0 MIPS")
    assert (corrected, corrected_mips) == (2_000_000, 42.0)

    assert spike_sim.parse_commit_stats("nothing here") == (0, 0.0)


def test_parse_cache_stats():
    stats = spike_sim.parse_cache_stats(CANNED_STDERR)
    assert stats["icache_hits"] == 10_000
    assert stats["icache_misses"] == 25
    assert stats["icache_miss_rate"] == 0.25
    assert stats["dcache_hits"] == 20_000
    assert stats["dcache_misses"] == 50
    assert stats["dcache_miss_rate"] == 0.25

    empty = spike_sim.parse_cache_stats("no stats here")
    assert set(empty) == {
        "icache_hits", "icache_misses", "icache_miss_rate",
        "dcache_hits", "dcache_misses", "dcache_miss_rate",
    }
    assert all(value == 0 for value in empty.values())


def test_parse_pc_histogram():
    hist = spike_sim.parse_pc_histogram(CANNED_STDOUT)
    assert hist == {0x80000014: 123, 0x80000018: 456}

    messy = (
        "PC histogram (number of commits per PC):\n"
        "0x80000010: 7\n"
        "garbage line\n"
        "0x80000020: 9\n"
        "\n"
        "0x80000030: 11\n"
    )
    assert spike_sim.parse_pc_histogram(messy) == {
        0x80000010: 7, 0x80000020: 9}

    assert spike_sim.parse_pc_histogram("no histogram here") == {}


def test_run_spike_mock_subprocess_records_parse_warnings(tmp_path, monkeypatch):
    tools = spike_sim.SpikeTools(spike="/fake/spike")
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(spike_sim.subprocess, "run", fake_run)

    result = spike_sim.run_spike(str(tmp_path / "x.elf"), tools=tools)

    assert captured["cmd"][0] == "/fake/spike"
    assert result.status == "ok"
    assert result.spike_path == "/fake/spike"
    assert result.committed_insns == 0
    assert result.parse_warnings
    assert any("commit" in w.lower() for w in result.parse_warnings)


def test_run_spike_missing_tool_returns_skipped(tmp_path):
    tools = spike_sim.SpikeTools(
        warnings=("SCRATCHV_SPIKE_BIN=/old/spike is not executable; ignored",))

    result = spike_sim.run_spike(str(tmp_path / "x.elf"), tools=tools)

    assert result.status == "skipped"
    assert result.skip_reason == "spike binary not found"
    assert result.exit_code == -2
    assert list(result.tool_warnings) == list(tools.warnings)


# ── Report fields ───────────────────────────────────────────────────────────

def test_report_status_fields():
    result = spike_sim.SpikeResult(
        status="skipped", skip_reason="spike binary not found")

    text = spike_sim.generate_spike_report(
        result, "output.bin", 64, "64:2:32", "128:4:32", 50_000_000)
    assert "Status:" in text
    assert "skipped" in text
    assert "Skip reason:" in text
    assert "spike binary not found" in text

    tools = spike_sim.SpikeTools()
    report = spike_sim.build_json_report(
        result, "output.bin", 64, "64:2:32", "128:4:32", 50_000_000, tools)
    assert report["status"] == "skipped"
    assert report["skip_reason"] == "spike binary not found"
    assert report["spike_binary"] is None
    for key in ("binary", "code_size", "static_insns", "max_instr",
                "committed_insns", "wall_time_s", "exit_code", "icache",
                "dcache", "top_pcs", "stderr_tail"):
        assert key in report
    assert report["spike_tools"]["spike"]["path"] is None

    plain = spike_sim.build_json_report(
        result, "output.bin", 64, "64:2:32", "128:4:32", 50_000_000)
    assert "spike_tools" not in plain
