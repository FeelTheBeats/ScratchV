"""Pytest gate for the ScratchV DSL benchmark suite (topic 06).

Each stage of each discovered case becomes an independently addressable
test node (``case_id + stage``), so failures pinpoint the exact stage.
Failures already located, owned and declared in ``*.meta.json`` are marked
``xfail`` (non-strict by default; xpasses are visible but do not fail CI).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.dsl_suite import (
    ASSERT_INST_BUDGET,
    ORACLE_EXECUTION,
    ORACLE_NONE,
    STAGE_ASSEMBLE,
    STAGE_BUDGET,
    STAGE_SEMANTIC,
    CaseSpec,
    DSLSuiteRunner,
    SemanticOutcome,
    compare_values,
    discover_cases,
    load_case_spec,
)

ALL_CASES: tuple[CaseSpec, ...] = tuple(discover_cases())
if not ALL_CASES:
    pytest.fail("dsl suite discovery found 0 cases", pytrace=False)


def marks_for(case: CaseSpec, stage: str) -> list[pytest.MarkDecorator]:
    """Return the xfail marker for *stage* when the case declares it."""
    if case.xfail is not None and stage in case.xfail.stages:
        return [
            pytest.mark.xfail(
                reason=case.xfail.reason,
                strict=case.xfail.strict,
            )
        ]
    return []


def params_for(stage: str | None) -> list[pytest.ParameterSet]:
    """Build per-case parameters; marks must be injected per parameter."""
    return [
        pytest.param(
            case,
            id=case.pytest_id,
            marks=marks_for(case, stage) if stage else [],
        )
        for case in ALL_CASES
    ]


@pytest.fixture(scope="session")
def suite_runner() -> DSLSuiteRunner:
    runner = DSLSuiteRunner(verbose=False)
    yield runner
    runner.cleanup()


@pytest.fixture(scope="session")
def compile_results(suite_runner: DSLSuiteRunner) -> dict:
    return {
        case.case_id: suite_runner.compile_case(case) for case in ALL_CASES
    }


@pytest.fixture(scope="session")
def asm_results(
    suite_runner: DSLSuiteRunner, compile_results: dict,
) -> dict:
    return {
        case_id: suite_runner.assemble_asm(outcome.asm_text)
        for case_id, outcome in compile_results.items()
    }


@pytest.fixture(scope="session")
def semantic_results(
    suite_runner: DSLSuiteRunner, compile_results: dict, asm_results: dict,
) -> dict:
    results: dict = {}
    for case in ALL_CASES:
        if case.skip_reason or case.oracle == ORACLE_NONE:
            results[case.case_id] = None
            continue
        if case.oracle == ORACLE_EXECUTION and not asm_results[case.case_id].ok:
            results[case.case_id] = SemanticOutcome(
                ok=None,
                oracle=case.oracle,
                expected=case.expected_return,
                actual=None,
                duration_s=0.0,
                blocked_reason="blocked by assemble stage",
                error=None,
            )
            continue
        results[case.case_id] = suite_runner.evaluate_semantics(
            case, compile_results[case.case_id],
        )
    return results


@pytest.mark.parametrize("case", params_for(None))
def test_meta_contract(case: CaseSpec) -> None:
    assert case.meta_errors == (), (
        f"meta contract violations: {case.meta_errors}"
    )
    if case.xfail is not None:
        assert case.xfail.owner.strip(), "xfail.owner must be non-empty"
        assert case.xfail.reason.strip(), "xfail.reason must be non-empty"


@pytest.mark.parametrize("case", params_for(None))
def test_compile_ok(case: CaseSpec, compile_results: dict) -> None:
    if case.skip_reason:
        pytest.skip(case.skip_reason)
    outcome = compile_results[case.case_id]
    assert outcome.ok, f"compile failed: {outcome.error}"


@pytest.mark.parametrize("case", params_for(STAGE_ASSEMBLE))
def test_asm_encodable(case: CaseSpec, asm_results: dict) -> None:
    if case.skip_reason:
        pytest.skip(case.skip_reason)
    outcome = asm_results[case.case_id]
    assert outcome.ok, f"assemble failed: {outcome.error}"


@pytest.mark.parametrize("case", params_for(STAGE_BUDGET))
def test_instruction_budget(case: CaseSpec, asm_results: dict) -> None:
    if case.skip_reason:
        pytest.skip(case.skip_reason)
    if ASSERT_INST_BUDGET not in case.assertions:
        pytest.skip("inst_budget assertion disabled")
    if case.max_instructions is None:
        pytest.skip("no max_instructions declared")
    outcome = asm_results[case.case_id]
    assert outcome.ok, "blocked by assemble stage"
    assert outcome.instruction_count is not None
    assert outcome.instruction_count <= case.max_instructions, (
        f"{outcome.instruction_count} > {case.max_instructions}"
    )


@pytest.mark.parametrize("case", params_for(STAGE_SEMANTIC))
def test_semantic_golden(
    case: CaseSpec, semantic_results: dict,
) -> None:
    if case.skip_reason:
        pytest.skip(case.skip_reason)
    if case.oracle == ORACLE_NONE:
        pytest.skip("oracle=none")
    outcome = semantic_results[case.case_id]
    assert outcome is not None
    assert outcome.ok, outcome.blocked_reason or outcome.error


def test_compare_values_handles_scalars_vectors_and_text() -> None:
    assert compare_values([2, 4, 6, 8], "[2. 4. 6. 8.]")
    assert compare_values(30.0, "30.")
    assert compare_values(
        [0.032059, 0.087144, 0.236883, 0.643914],
        [0.0320586, 0.08714432, 0.23688282, 0.64391428],
    )
    assert not compare_values([1.0, 2.0], [1.0, 2.0, 3.0])
    assert not compare_values(1.0, 2.0)
    assert not compare_values(None, 1.0)


def _write_pseudo_case(
    directory: Path,
    name: str,
    source: str,
    meta: dict,
) -> Path:
    dsl_path = directory / f"{name}.dsl"
    dsl_path.write_text(source)
    (directory / f"{name}.meta.json").write_text(json.dumps(meta))
    return dsl_path


@pytest.mark.parametrize(
    ("name", "source", "meta", "fragment"),
    [
        (
            "interpreter_control",
            "while (i < 3):\n  i = add(i, 1)\nendwhile\nreturn i\n",
            {
                "description": "pseudo case",
                "oracle": "interpreter",
                "expected_return": 3,
            },
            "oracle/flow conflict",
        ),
        (
            "execution_missing_registers",
            "c = add(a, b)\nreturn c\n",
            {
                "description": "pseudo case",
                "oracle": "execution",
                "inputs": {"a": 1, "b": 2},
                "expected_return": 3,
            },
            "input_registers is required",
        ),
        (
            "none_without_xfail",
            "c = add(a, b)\nreturn c\n",
            {
                "description": "pseudo case",
                "oracle": "none",
            },
            "oracle=none requires",
        ),
    ],
)
def test_meta_contract_rejects_invalid_metadata(
    tmp_path: Path,
    name: str,
    source: str,
    meta: dict,
    fragment: str,
) -> None:
    dsl_path = _write_pseudo_case(tmp_path, name, source, meta)
    spec = load_case_spec(dsl_path, tmp_path)
    assert any(fragment in error for error in spec.meta_errors), (
        f"expected {fragment!r} in {spec.meta_errors}"
    )
