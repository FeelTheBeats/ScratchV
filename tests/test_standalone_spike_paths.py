"""F1: Spike tool path resolution must be portable (no hardcoded paths)."""

from scratchv.standalone.spike_sim import _resolve_tool


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv("SCRATCHV_SPIKE", "/opt/riscv/bin/spike")
    assert _resolve_tool("spike", "SCRATCHV_SPIKE") == "/opt/riscv/bin/spike"


def test_falls_back_to_path_lookup(monkeypatch):
    monkeypatch.delenv("SCRATCHV_SPIKE", raising=False)
    monkeypatch.setattr(
        "scratchv.standalone.spike_sim.shutil.which",
        lambda name: f"/usr/bin/{name}",
    )
    assert _resolve_tool("spike", "SCRATCHV_SPIKE") == "/usr/bin/spike"


def test_falls_back_to_bare_name(monkeypatch):
    monkeypatch.delenv("SCRATCHV_SPIKE", raising=False)
    monkeypatch.setattr("scratchv.standalone.spike_sim.shutil.which", lambda name: None)
    assert _resolve_tool("spike", "SCRATCHV_SPIKE") == "spike"


def test_no_hardcoded_foreign_path_remains():
    import inspect

    from scratchv.standalone import spike_sim

    source = inspect.getsource(spike_sim)
    assert "kinsomwang" not in source
    assert "/home/" not in source
