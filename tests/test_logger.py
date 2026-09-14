"""Tests for the compiler logger module."""

import logging
import os
import tempfile
from unittest import mock

import pytest
import scratchv.utils.logger as logger_mod
from scratchv.utils.logger import (
    init_logger,
    get_logger,
    set_level,
    shutdown,
    log_progress,
    log_phase,
    log_step,
)


class TestInitLogger:
    """Tests for logger initialization."""

    def test_init_default(self):
        init_logger()
        log = get_logger("test_init")
        assert log is not None
        assert isinstance(log, logging.Logger)
        shutdown()

    def test_init_with_level(self):
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            init_logger(level=level)
            log = get_logger("test_level")
            assert log.level <= getattr(logging, level)
            shutdown()

    def test_init_invalid_level(self):
        with pytest.raises(ValueError):
            init_logger(level="INVALID")

    def test_init_with_log_file(self):
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            log_path = f.name
        try:
            init_logger(level="DEBUG", log_file=log_path)
            log = get_logger("test_file")
            log.info("test message")
            shutdown()

            # Check file was written
            with open(log_path) as f:
                content = f.read()
            assert "test message" in content
        finally:
            if os.path.exists(log_path):
                os.unlink(log_path)

    def test_init_no_color(self):
        init_logger(use_color=False)
        log = get_logger("test_nocolor")
        log.info("no color message")
        # Should not crash
        shutdown()


class TestGetLogger:
    """Tests for logger factory."""

    def test_auto_prefix(self):
        init_logger()
        log = get_logger("parser")
        assert log.name == "scratchv.parser"
        shutdown()

    def test_already_prefixed(self):
        init_logger()
        log = get_logger("scratchv.codegen")
        assert log.name == "scratchv.codegen"
        shutdown()

    def test_hierarchical_names(self):
        init_logger()
        log = get_logger("optimizer.constant_folding")
        assert log.name == "scratchv.optimizer.constant_folding"
        shutdown()

    def test_returns_logger_instance(self):
        init_logger()
        log = get_logger("test")
        assert isinstance(log, logging.Logger)
        shutdown()

    def test_auto_init_on_first_call(self):
        # If init_logger was never called, get_logger should auto-init
        # Reset by clearing handlers first
        root = logging.getLogger("scratchv")
        root.handlers.clear()
        # Force re-init
        import scratchv.utils.logger as logger_mod
        logger_mod._initialized = False
        log = get_logger("auto")
        assert isinstance(log, logging.Logger)
        shutdown()


class TestSetLevel:
    """Tests for runtime level changes."""

    def test_change_level(self):
        init_logger(level="INFO")
        # Child loggers inherit effective level from parent (NOTSET=0)
        log = get_logger("test_setlevel")
        assert log.getEffectiveLevel() == logging.INFO
        set_level("DEBUG")
        root = logging.getLogger("scratchv")
        assert root.level == logging.DEBUG
        shutdown()

    def test_invalid_level(self):
        init_logger()
        with pytest.raises(ValueError):
            set_level("NOPE")
        shutdown()


class TestLogging:
    """Tests for actual log output."""

    def test_info_logging(self):
        init_logger(level="INFO")
        log = get_logger("test_log_info")
        # Just verify no exception
        log.info("test message %d", 42)
        log.warning("warning message")
        shutdown()

    def test_debug_filtering(self):
        init_logger(level="INFO")
        log = get_logger("test_filter")
        # Debug messages at INFO level should not be emitted
        # (just verify no crash)
        log.debug("this should not appear")
        shutdown()

    def test_error_logging(self):
        init_logger()
        log = get_logger("test_error")
        try:
            raise ValueError("test exception")
        except ValueError:
            log.error("caught exception", exc_info=True)
        shutdown()


class TestProgress:
    """Tests for progress indicators."""

    def test_log_progress(self):
        init_logger(level="INFO")
        log_progress("test_prog", current=5, total=10, description="Testing")
        shutdown()

    def test_log_step(self):
        init_logger(level="DEBUG")
        log_step("test_step", "selecting instructions")
        shutdown()


class TestLogPhase:
    """Tests for the log_phase context manager."""

    def test_phase_success(self):
        init_logger(level="INFO")
        with log_phase("test_phase", "Testing phase"):
            pass  # succeed
        shutdown()

    def test_phase_failure(self):
        init_logger(level="INFO")
        try:
            with log_phase("test_phase_fail", "Failing phase"):
                raise RuntimeError("phase failed")
        except RuntimeError:
            pass  # expected
        shutdown()


class TestLoggerDefectRegressions:
    """Regression tests for the known logger defects D1-D5."""

    def teardown_method(self):
        shutdown()

    def test_exc_info_preserved_console_and_file(self, tmp_path, capsys):
        log_path = tmp_path / "exc.log"
        init_logger(level="DEBUG", log_file=str(log_path), use_color=False)
        try:
            raise ValueError("boom")
        except ValueError:
            get_logger("test.exc").error("caught", exc_info=True)
        shutdown()

        err = capsys.readouterr().err
        file_text = log_path.read_text()

        assert "Traceback (most recent call last)" in err
        assert "ValueError: boom" in err
        assert "Traceback (most recent call last)" in file_text
        assert "ValueError: boom" in file_text
        assert "\033[" not in err

    def test_reinit_closes_old_handlers(self, tmp_path):
        init_logger(level="DEBUG", log_file=str(tmp_path / "a.log"))
        old_handlers = list(logging.getLogger("scratchv").handlers)
        spies = [
            mock.patch.object(h, "close", wraps=h.close)
            for h in old_handlers
        ]
        started = [s.start() for s in spies]
        try:
            init_logger(level="INFO", log_file=str(tmp_path / "b.log"))
        finally:
            for s in spies:
                s.stop()

        assert all(m.called for m in started)
        old_file = next(
            h for h in old_handlers
            if isinstance(h, logging.FileHandler)
        )
        assert old_file.stream is None
        new_handlers = list(logging.getLogger("scratchv").handlers)
        assert len(new_handlers) == 2
        # With a file handler the root logger stays at DEBUG so the file
        # receives DEBUG records; the console handler carries the level (F1).
        assert logging.getLogger("scratchv").level == logging.DEBUG
        assert logger_mod._console_handler.level == logging.INFO

    def test_shutdown_resets_state(self):
        init_logger(level="DEBUG")
        shutdown()
        assert logger_mod._initialized is False
        assert logger_mod._root_logger is None
        assert logger_mod._console_handler is None
        assert logger_mod._file_handler is None
        assert logger_mod._config == {}

        log = get_logger("after_shutdown")
        assert isinstance(log, logging.Logger)
        log.info("revived logger")
        shutdown()

    def test_set_level_updates_console_only(self, tmp_path, capsys):
        log_path = tmp_path / "a.log"
        init_logger(level="INFO", log_file=str(log_path))
        root = logging.getLogger("scratchv")
        assert root.level == logging.DEBUG

        set_level("WARNING")
        assert logger_mod._console_handler.level == logging.WARNING
        assert logger_mod._file_handler.level == logging.DEBUG
        assert root.level == logging.DEBUG
        assert logger_mod._config["level"] == "WARNING"

        # Verify actual routing, not just handler levels: INFO is dropped on
        # the console but still recorded in the file.
        log = get_logger("test.setlevel")
        log.info("info-suppressed-on-console")
        log.warning("warning-shown")
        shutdown()

        err = capsys.readouterr().err
        text = log_path.read_text()
        assert "info-suppressed-on-console" not in err
        assert "warning-shown" in err
        assert "info-suppressed-on-console" in text
        assert "warning-shown" in text

    def test_log_file_only_contains_debug(self, tmp_path, capsys):
        log_path = tmp_path / "debug.log"
        init_logger(level="INFO", log_file=str(log_path), use_color=False)
        log = get_logger("test.debug_file")
        log.debug("debug-only-line")
        log.info("info-line")
        shutdown()

        err = capsys.readouterr().err
        text = log_path.read_text()

        assert "debug-only-line" in text
        assert "info-line" in text
        assert "debug-only-line" not in err
        assert "info-line" in err

    def test_log_file_error_resets_state(self, tmp_path, capsys):
        bad_path = tmp_path / "missing_dir" / "x.log"
        with pytest.raises(logger_mod.LogFileError):
            init_logger(level="INFO", log_file=str(bad_path))

        # No half-initialized logger is left behind.
        assert logger_mod._initialized is False
        assert logger_mod._root_logger is None
        assert logger_mod._console_handler is None
        assert logger_mod._file_handler is None
        assert logging.getLogger("scratchv").handlers == []
        assert capsys.readouterr().err == ""
