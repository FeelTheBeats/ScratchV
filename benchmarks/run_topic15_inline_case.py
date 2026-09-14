#!/usr/bin/env python3
"""Run the Topic 15 function-inline feature case and emit auditable reports.

The DSL and ONNX frontends never emit ``OpCode.CALL``, so the case is built
programmatically (see ``benchmarks/cases/topic15_inline_feature.py``).  The
report proves four separate facts on that deterministic IR:

1. A/B: with the inliner off the two CALLs stay; with the inliner on (fixed
   :class:`InlinerConfig`) both CALLs disappear;
2. each call site receives one independent clone: distinct ``_inl{k}`` block
   and value namespaces, RETURN rewritten to ``br <caller>_inl{k}_cont``,
   the callee's own RETURN preserved, and no ``IRVerifier`` ERROR left;
3. the transformation is deterministic: repeated runs produce identical IR
   fingerprints;
4. the conservative rejection rules are real: an oversized side-effecting
   callee and a loop-bodied callee keep their CALLs, the IR is unchanged and
   every refusal is recorded as one warning.

This is a deterministic feature/integration case, not a speedup claim.  It
does not execute residual CALLs: the RISC-V CALL ABI is not implemented in
this branch.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from scratchv.analysis.ir_verifier import ErrorLevel, IRVerifier
from scratchv.ir.types import BasicBlock, Function, OpCode, Program
from scratchv.optimizer.inliner import Inliner, InlinerConfig

SCHEMA_VERSION = "topic15-inline-case/1"
DEFAULT_CASE = (
    Path(__file__).parent / "cases" / "topic15_inline_feature.py"
)
DEFAULT_JSON = Path("benchmark_reports/inliner_report.json")
DEFAULT_MARKDOWN = Path("benchmark_reports/inliner_report.md")
ELIGIBLE_CALLEE = "inc"
ELIGIBLE_CALL_SITES = 2
REJECTED_CALL_SITES = 2
REJECTED_MAX_INSTRS = 4
_CLONE_SUFFIX = re.compile(r"_inl(\d+)")
HONESTY = (
    "Structural IR proof only.  The DSL/ONNX frontends never emit CALL, so "
    "the case is built programmatically; no execution-equivalence claim is "
    "made because the RISC-V CALL ABI (prologue/epilogue, argument passing) "
    "is not implemented in this branch -- residual CALLs are refused by the "
    "backend or degraded to non-executable staging.  Instruction counts are "
    "static IR counts, not dynamic instruction totals or speedups, and the "
    "inliner is opt-in and conservative (loops, oversize, recursion and "
    "mixed return shapes are refused).  Note: the IRVerifier label-existence "
    "rule currently treats a residual CALL's function-name target as a block "
    "label, so uninlined programs report one spurious ERROR per CALL; once "
    "inlining removes every CALL the verifier reports no ERROR."
)


def default_inliner_config() -> InlinerConfig:
    """Fixed config for the eligible A/B run (deterministic decisions)."""
    return InlinerConfig(
        max_instrs=32,
        single_site_only=False,
        growth_budget=256,
        reject_loops=True,
        allow_ret_drop=False,
        max_rounds=4,
    )


def rejected_inliner_config() -> InlinerConfig:
    """Config that refuses the oversized side-effecting callee."""
    return InlinerConfig(max_instrs=REJECTED_MAX_INSTRS)


def load_case_module(case_path: Path):
    """Import the programmatic case module from *case_path*."""
    spec = importlib.util.spec_from_file_location(
        "topic15_inline_feature", case_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import feature case: {case_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _iter_instructions(program: Program):
    for func in program.functions:
        for block in func.blocks:
            for ins in block.instructions:
                yield func, block, ins


def count_ir(program: Program) -> int:
    return sum(1 for _ in _iter_instructions(program))


def fingerprint(program: Program) -> str:
    """Stable structural fingerprint (memory addresses never enter dump)."""
    return hashlib.sha256(program.dump().encode("utf-8")).hexdigest()


def _function(program: Program, name: str) -> Function | None:
    return next((f for f in program.functions if f.name == name), None)


def _body_size(func: Function) -> int:
    return sum(len(b.instructions) for b in func.blocks)


def _clone_index(name: str) -> int | None:
    match = _CLONE_SUFFIX.search(name)
    return int(match.group(1)) if match else None


def _is_cont_block(name: str) -> bool:
    return name.endswith("_cont")


def verifier_errors(program: Program) -> list[str]:
    return [
        err.message
        for err in IRVerifier(program).verify()
        if err.level is ErrorLevel.ERROR
    ]


def _duplicate_defined_names(func: Function) -> list[str]:
    names = [p.name for p in func.params]
    for block in func.blocks:
        for ins in block.instructions:
            if ins.dest is not None:
                names.append(ins.dest.name)
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in names:
        if name in seen:
            duplicates.append(name)
        seen.add(name)
    return sorted(set(duplicates))


def _describe(
    program: Program,
    *,
    callee: str | None = None,
    stats: dict[str, int] | None = None,
    warnings: list[str] | None = None,
    pass_time_ms: float | None = None,
) -> dict[str, Any]:
    """Summarize the (possibly transformed) program for the report."""
    blocks_by_index: dict[int, list[BasicBlock]] = {}
    block_names: list[str] = []
    for func in program.functions:
        for block in func.blocks:
            block_names.append(block.name)
            index = _clone_index(block.name)
            if index is not None and not _is_cont_block(block.name):
                blocks_by_index.setdefault(index, []).append(block)

    clones_detail: list[dict[str, Any]] = []
    clone_returns = 0
    for index in sorted(blocks_by_index):
        dest_names: list[str] = []
        redirects: dict[str, str] = {}
        returnless = True
        for block in blocks_by_index[index]:
            for ins in block.instructions:
                if ins.opcode is OpCode.RETURN:
                    clone_returns += 1
                    returnless = False
                if ins.dest is not None:
                    dest_names.append(ins.dest.name)
            last = block.instructions[-1] if block.instructions else None
            if last is not None and last.opcode is OpCode.BR:
                redirects[block.name] = last.target
        clones_detail.append({
            "index": index,
            "blocks": [b.name for b in blocks_by_index[index]],
            "dest_names": dest_names,
            "return_redirects": redirects,
            "returnless": returnless,
        })

    calls = [
        ins for _f, _b, ins in _iter_instructions(program)
        if ins.opcode is OpCode.CALL
    ]
    callee_func = _function(program, callee) if callee else None
    returns_in_callee = 0
    if callee_func is not None:
        returns_in_callee = sum(
            1
            for block in callee_func.blocks
            for ins in block.instructions
            if ins.opcode is OpCode.RETURN
        )
    return {
        "ir_instructions": count_ir(program),
        "call_count": len(calls),
        "call_targets": [ins.target for ins in calls],
        "clone_count": len(blocks_by_index),
        "clones": stats["inlined"] if stats is not None else 0,
        "rejected": stats["rejected"] if stats is not None else 0,
        "rounds": stats["rounds"] if stats is not None else 0,
        "warnings": list(warnings or []),
        "callee_body_size": (
            _body_size(callee_func) if callee_func is not None else None),
        "returns_in_callee": returns_in_callee,
        "clone_returns": clone_returns,
        "clones_detail": clones_detail,
        "block_names": block_names,
        "duplicate_defined_names": {
            func.name: _duplicate_defined_names(func)
            for func in program.functions
        },
        "verifier_errors": verifier_errors(program),
        "pass_time_ms": (
            round(pass_time_ms, 4) if pass_time_ms is not None else None),
        "fingerprint": fingerprint(program),
    }


def _measure_uninlined(build: Callable[[], Program]) -> dict[str, Any]:
    """Build the eligible case and never run the inliner."""
    return _describe(build(), callee=ELIGIBLE_CALLEE)


def _measure_inlined(
        build: Callable[[], Program], repeats: int) -> dict[str, Any]:
    """Build the eligible case and run the inliner *repeats* times."""
    program: Program | None = None
    runner: Inliner | None = None
    times: list[float] = []
    fingerprints: list[str] = []
    for _ in range(repeats):
        program = build()
        runner = Inliner(program, default_inliner_config())
        started = time.perf_counter()
        runner.run()
        times.append((time.perf_counter() - started) * 1000)
        fingerprints.append(fingerprint(program))
    assert program is not None and runner is not None
    described = _describe(
        program,
        callee=ELIGIBLE_CALLEE,
        stats=runner.stats,
        warnings=runner.warnings,
        pass_time_ms=statistics.median(times),
    )
    described["fingerprints"] = fingerprints
    return described


def _measure_rejected(
        build_rejected: Callable[[], Program]) -> dict[str, Any]:
    """Run the inliner on the conservative-rejection case."""
    program = build_rejected()
    before = fingerprint(program)
    runner = Inliner(program, rejected_inliner_config())
    runner.run()
    described = _describe(
        program,
        stats=runner.stats,
        warnings=runner.warnings,
    )
    described["max_instrs"] = REJECTED_MAX_INSTRS
    described["dump_unchanged"] = fingerprint(program) == before
    return described


def evaluate(case_path: Path, repeats: int) -> dict[str, Any]:
    """Build the full report payload and run the hard invariants."""
    module = load_case_module(case_path)
    off = _measure_uninlined(module.build_program)
    on = _measure_inlined(module.build_program, repeats)
    rejected = _measure_rejected(module.build_rejected_program)

    by_index = {d["index"]: d for d in on["clones_detail"]}
    dests0 = set(by_index.get(0, {"dest_names": []})["dest_names"])
    dests1 = set(by_index.get(1, {"dest_names": []})["dest_names"])
    all_block_names = set(on["block_names"])
    clones_independent = (
        bool(dests0) and bool(dests1)
        and not (dests0 & dests1)
        and all(name.endswith("_inl0") for name in dests0)
        and all(name.endswith("_inl1") for name in dests1)
        and not any(on["duplicate_defined_names"].values())
    )
    returns_rewritten = (
        on["clone_returns"] == 0
        and len(on["clones_detail"]) == ELIGIBLE_CALL_SITES
        and all(d["returnless"] for d in on["clones_detail"])
        and all(
            len(d["return_redirects"]) == len(d["blocks"])
            and all(t in all_block_names
                    for t in d["return_redirects"].values())
            for d in on["clones_detail"]
        )
    )
    rejected_call_count = REJECTED_CALL_SITES
    hard_checks = {
        "uninlined_keeps_both_calls": (
            off["call_count"] == ELIGIBLE_CALL_SITES),
        "inlined_removes_all_calls": on["call_count"] == 0,
        "inliner_clones_each_site": (
            on["clones"] == ELIGIBLE_CALL_SITES
            and on["clone_count"] == ELIGIBLE_CALL_SITES
            and on["rejected"] == 0
        ),
        "clone_names_do_not_collide": clones_independent,
        "clone_returns_rewritten_to_branches": returns_rewritten,
        "callee_return_preserved": on["returns_in_callee"] == 1,
        "verifier_clean_after_inline": on["verifier_errors"] == [],
        "ir_grows_by_cloned_bodies": (
            on["clones"] == ELIGIBLE_CALL_SITES
            and off["callee_body_size"] is not None
            and on["ir_instructions"] - off["ir_instructions"]
            == on["clones"] * off["callee_body_size"]
        ),
        "deterministic_across_runs": (
            len(on["fingerprints"]) == repeats
            and len(set(on["fingerprints"])) == 1
        ),
        "rejected_sites_keep_calls": (
            rejected["call_count"] == rejected_call_count
            and rejected["rejected"] == rejected_call_count
            and rejected["clones"] == 0
        ),
        "rejected_warnings_recorded": (
            len(rejected["warnings"]) == rejected_call_count
            and any("body_too_large" in w for w in rejected["warnings"])
            and any("loop_body_unsupported" in w
                    for w in rejected["warnings"])
            and all(w.startswith("inliner: skip")
                    for w in rejected["warnings"])
        ),
        "rejected_program_unchanged": rejected["dump_unchanged"],
    }
    failed = sorted(name for name, ok in hard_checks.items() if not ok)

    return {
        "schema_version": SCHEMA_VERSION,
        "topic": "topic15-function-inline",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case": str(case_path),
        "eligible_call_sites": ELIGIBLE_CALL_SITES,
        "config": dataclasses.asdict(default_inliner_config()),
        "rejected_config": {"max_instrs": REJECTED_MAX_INSTRS},
        "runs": repeats,
        "uninlined": off,
        "inlined": on,
        "rejected": rejected,
        "hard_checks": hard_checks,
        "hard_failures": failed,
        "honesty": HONESTY,
    }


def render_markdown(report: dict[str, Any]) -> str:
    off, on, rejected = (
        report["uninlined"], report["inlined"], report["rejected"])
    cfg = report["config"]
    total = len(report["hard_checks"])
    passed = total - len(report["hard_failures"])
    lines = [
        "# Topic 15 Function-Inline Feature Case",
        "",
        f"- Schema: `{report['schema_version']}`",
        f"- Case: `{report['case']}` (programmatic IR; frontends never emit "
        "CALL)",
        f"- Generated: {report['generated_at']}",
        f"- Fixed config: `max_instrs={cfg['max_instrs']}`, "
        f"`single_site_only={cfg['single_site_only']}`, "
        f"`growth_budget={cfg['growth_budget']}`, "
        f"`reject_loops={cfg['reject_loops']}`, "
        f"`max_rounds={cfg['max_rounds']}`; rejected branch uses "
        f"`max_instrs={report['rejected_config']['max_instrs']}`",
        f"- Hard checks: {'PASS' if not report['hard_failures'] else 'FAIL'} "
        f"({passed}/{total})",
        "",
        "## A/B summary",
        "",
        "| Metric | inliner off | inliner on | delta |",
        "|--------|------------:|-----------:|------:|",
        f"| IR instructions | {off['ir_instructions']} | "
        f"{on['ir_instructions']} | "
        f"{on['ir_instructions'] - off['ir_instructions']:+d} |",
        f"| CALL instructions | {off['call_count']} | {on['call_count']} | "
        f"{on['call_count'] - off['call_count']:+d} |",
        f"| Inlined call sites (clones) | {off['clones']} | {on['clones']} | "
        f"{on['clones'] - off['clones']:+d} |",
        f"| Rejected call sites | {off['rejected']} | {on['rejected']} | "
        f"{on['rejected'] - off['rejected']:+d} |",
        f"| Warnings | {len(off['warnings'])} | {len(on['warnings'])} | "
        f"{len(on['warnings']) - len(off['warnings']):+d} |",
        f"| Verifier ERRORs | {len(off['verifier_errors'])} | "
        f"{len(on['verifier_errors'])} | "
        f"{len(on['verifier_errors']) - len(off['verifier_errors']):+d} |",
        f"| Pass time (ms, median) | n/a | "
        f"{on['pass_time_ms']:.4f} | - |",
        "",
        "## Clone detail (inliner on)",
        "",
    ]
    for detail in on["clones_detail"]:
        rewrite = ", ".join(
            f"`{block}` -> `{target}`"
            for block, target in detail["return_redirects"].items()
        ) or "none"
        definitions = ", ".join(
            f"`{name}`" for name in detail["dest_names"]) or "none"
        blocks = ", ".join(f"`{name}`" for name in detail["blocks"]) or "none"
        lines.append(
            f"- clone `_inl{detail['index']}`: blocks {blocks}; "
            f"renamed defs {definitions}; RETURN rewritten to {rewrite}"
        )
    lines += [
        "",
        "## Rejection detail (conservative v1, "
        f"`max_instrs={rejected['max_instrs']}`)",
        "",
        f"- CALLs kept: {rejected['call_count']}; IR unchanged: "
        f"{rejected['dump_unchanged']}; inlined={rejected['clones']}, "
        f"rejected={rejected['rejected']}",
    ]
    lines += [f"- `{warning}`" for warning in rejected["warnings"]]
    lines += [
        "",
        "## Determinism",
        "",
        f"- inliner off fingerprint: `{off['fingerprint']}`",
        "- inliner on fingerprints: "
        + ", ".join(f"`{fp}`" for fp in on["fingerprints"]),
        f"- rejected fingerprint: `{rejected['fingerprint']}`",
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
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not args.case.is_file():
        parser.error(f"feature case not found: {args.case}")

    report = evaluate(args.case, args.repeats)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(report) + "\n")
    print(render_markdown(report))
    if report["hard_failures"]:
        print("HARD FAILURES: " + ", ".join(report["hard_failures"]))
        return 1
    print(f"reports written: {args.json}, {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
