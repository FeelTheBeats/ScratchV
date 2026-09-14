"""Wiring tests for compiler structured logging (Topic 07).

Covers the CLI -> CompilerConfig contract, stderr-only console handler
(R1/R2), end-to-end logging activation (D6) and default-off stability.
"""

import re
import sys
from pathlib import Path

import pytest

import scratchv.utils.logger as logger_mod
from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.main import args_to_config, build_arg_parser, main
from scratchv.utils.logger import shutdown

DSL = (Path(__file__).resolve().parent.parent
       / "benchmarks" / "cases" / "001_simple_add.dsl")
ONNX = (Path(__file__).resolve().parent.parent
        / "models" / "graph" / "cnn.onnx")


@pytest.fixture(autouse=True)
def _logger_teardown():
    yield
    shutdown()


def test_args_to_config_logging_fields():
    args = build_arg_parser().parse_args(
        ["input.dsl", "--log-level", "DEBUG", "--log-file", "x.log"])
    config = args_to_config(args)
    assert config.use_logger is True
    assert config.log_level == "DEBUG"
    assert config.log_file == "x.log"
    assert isinstance(config.log_color, bool)


def test_log_file_only_implies_logging():
    args = build_arg_parser().parse_args(["input.dsl", "--log-file", "x.log"])
    config = args_to_config(args)
    assert config.use_logger is True
    assert config.log_level == "INFO"
    assert config.log_file == "x.log"


def test_invalid_log_level_rejected_by_cli():
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(["input.dsl", "--log-level", "VERBOSE"])


def test_console_handler_targets_stderr():
    logger_mod.init_logger(level="INFO", use_color=False)
    assert logger_mod._console_handler is not None
    assert logger_mod._console_handler.stream is sys.stderr
    shutdown()


def test_cli_logging_end_to_end(tmp_path, capsys):
    out = tmp_path / "out.s"
    log_file = tmp_path / "build.log"
    rc = main([str(DSL), "-o", str(out), "--optimize", "all",
               "--log-level", "DEBUG", "--log-file", str(log_file)])
    captured = capsys.readouterr()

    assert rc == 0
    assert out.exists() and out.stat().st_size > 0
    assert captured.out == ""
    assert "[scratchv.compiler.parse]" in captured.err
    assert "done (" in captured.err
    assert "[scratchv.compiler.codegen]" in captured.err
    assert "[scratchv.compiler.passes]" in captured.err
    assert "constant-folding" in captured.err

    text = log_file.read_text()
    assert "DEBUG" in text
    assert "pass constant-folding" in text
    assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", text)


def test_no_logging_by_default_keeps_outputs_stable(tmp_path, capsys):
    out = tmp_path / "out.s"
    rc = main([str(DSL), "-o", str(out)])
    captured = capsys.readouterr()

    assert rc == 0
    assert out.exists()
    assert captured.out == ""
    assert "scratchv.compiler" not in captured.err
    assert captured.err.strip() == f"OK RISCV output written to {out}"


def test_compile_failure_logged(tmp_path, capsys):
    bad = tmp_path / "bad.dsl"
    bad.write_text("add(a, b\n", encoding="utf-8")
    log_file = tmp_path / "fail.log"

    rc = main([str(bad), "-o", str(tmp_path / "out.s"),
               "--log-level", "DEBUG", "--log-file", str(log_file)])
    captured = capsys.readouterr()

    assert rc == 1
    assert captured.out == ""
    text = log_file.read_text()
    assert "ERROR" in text
    assert "compilation failed" in text
    assert "error[E" in text


def test_parse_failure_logged(tmp_path, monkeypatch, capsys):
    def bad_parse(self, input_path, dsl_source=None):
        raise ValueError("parse exploded")

    monkeypatch.setattr(CompilerDriver, "_parse", bad_parse)
    log_file = tmp_path / "parse.log"
    rc = main(["model.onnx", "-o", str(tmp_path / "out.s"),
               "--log-level", "DEBUG", "--log-file", str(log_file)])
    capsys.readouterr()

    assert rc == 1
    text = log_file.read_text()
    assert "ERROR" in text
    assert "compilation failed: parse exploded" in text


def test_codegen_failure_logged(tmp_path, monkeypatch, capsys):
    def bad_codegen(self, program):
        raise ValueError("codegen exploded")

    monkeypatch.setattr(CompilerDriver, "_generate_code", bad_codegen)
    log_file = tmp_path / "codegen.log"
    rc = main([str(DSL), "-o", str(tmp_path / "out.s"),
               "--log-level", "DEBUG", "--log-file", str(log_file)])
    capsys.readouterr()

    assert rc == 1
    text = log_file.read_text()
    assert "ERROR" in text
    assert "compilation failed: codegen exploded" in text
    assert "FAILED" in text


def test_log_file_same_as_input_refused(tmp_path, capsys):
    src = tmp_path / "input.dsl"
    src.write_text(DSL.read_text(encoding="utf-8"), encoding="utf-8")
    original = src.read_text(encoding="utf-8")

    rc = main([str(src), "-o", str(tmp_path / "out.s"),
               "--log-file", str(src)])
    captured = capsys.readouterr()

    assert rc == 2
    assert src.read_text(encoding="utf-8") == original
    assert "log file" in captured.err
    assert "would overwrite input" in captured.err
    assert "Traceback" not in captured.err

    rc2 = main([str(DSL), "-o", str(tmp_path / "out2.s"),
                "--log-file", str(tmp_path / "out2.s")])
    captured2 = capsys.readouterr()

    assert rc2 == 2
    assert "would overwrite output" in captured2.err


def test_log_file_bad_path_error(tmp_path, capsys):
    bad = tmp_path / "missing_dir" / "x.log"

    rc = main([str(DSL), "-o", str(tmp_path / "out.s"),
               "--log-file", str(bad)])
    captured = capsys.readouterr()

    assert rc == 2
    assert "cannot open log file" in captured.err
    assert "Traceback" not in captured.err
    assert logger_mod._initialized is False
    assert logger_mod._root_logger is None

    if ONNX.exists():
        rc2 = main([str(ONNX), "-o", str(tmp_path / "out2.s"),
                    "--log-file", str(bad)])
        captured2 = capsys.readouterr()

        assert rc2 == 2
        assert "cannot open log file" in captured2.err
        assert "Traceback" not in captured2.err


def test_driver_reuse_after_shutdown_keeps_file(tmp_path, capsys):
    log_file = tmp_path / "reuse.log"
    config = CompilerConfig(
        use_logger=True, log_level="INFO",
        log_file=str(log_file), log_color=False,
    )
    driver = CompilerDriver(config)

    first = driver.compile(str(DSL), str(tmp_path / "a.s"))
    assert first.success

    shutdown()
    capsys.readouterr()

    second = driver.compile(str(DSL), str(tmp_path / "b.s"))
    captured = capsys.readouterr()

    assert second.success
    # The second run must honour the driver config instead of silently
    # falling back to the default logger (no file, colored, INFO).
    assert logger_mod._file_handler is not None
    assert logger_mod._config.get("log_file") == str(log_file)
    assert "\033[" not in captured.err
    assert "compilation succeeded" in log_file.read_text()


def test_pass_exception_logs_traceback(tmp_path, monkeypatch, capsys):
    from scratchv.optimizer.constant_folding import ConstantFolder

    def boom(self):
        raise RuntimeError("simulated pass failure")

    monkeypatch.setattr(ConstantFolder, "run", boom)
    log_file = tmp_path / "pass.log"

    rc = main([str(DSL), "-o", str(tmp_path / "out.s"),
               "--optimize", "all", "--log-level", "DEBUG",
               "--log-file", str(log_file)])
    capsys.readouterr()

    assert rc == 0  # E2: the pass failure is swallowed by design
    text = log_file.read_text()
    assert "pass 'constant-folding' failed" in text
    assert "Traceback (most recent call last)" in text
    assert "RuntimeError: simulated pass failure" in text
