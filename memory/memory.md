# ScratchV 共享记忆底座

> 所有角色共享。每条知识只写一次，发现重复必须合并。
> 格式: `[日期] 描述。适用范围/前提/风险。`

---

## 架构现状

[2026-08-22] 寄存器分配器三体并存: (1) `register_alloc.RegisterAllocator` — 主线默认(compiler.py `_generate_riscv_linear` / `_generate_riscv_dag` 都用它); (2) `regalloc_linear.LinearScanAllocator` — 仅 `config.reg_alloc == "linear"` 时启用(compiler.py:408); (3) `regalloc_linear_v1_5.py` — PR#37 演进版，**未接入任何编译路径**，仅被 `topic17_bottleneck_scenarios_v1_5.py` 和 `tests/test_pr37_regression.py` 引用。后果: v1.3.1/v1.5 的修复(self-spill clobber、SPILL_ 泄漏、a_temp 误分类、fail-loudly RuntimeError)全部不在实际编译产物上生效。风险: 三套分配器语义漂移，集成时需明确正本。backend/__init__.py 只导出旧版 regalloc_linear 符号。

[2026-08-22] 编译管线两条路径共享前端: A) 原生 Q16.16→RV32IM (InstructionSelector→RegisterAllocator→AsmEmitter→riscv_encoder); B) LLVM 路径 float32→LLVM IR→外部 llc。Pass 顺序见 compiler.py CompilerDriver 与 `/root/Lab/GaoMD/ScratchV/ArcDes/init.md`(项目认知基座，2026-06-28 全量扫描生成)。当前差距: 动态指令数 4.2x (77.7亿 vs 18.5亿)，静态指令反而少 26%——瓶颈在循环嵌套效率而非代码膨胀。

## RA 优化方向数据

[2026-08-22] topic17 场景跑批(topic17_bottleneck_scenarios_v1_5.py, 23 场景)暴露的量化事实: spill slot_reuse=0.0%(每 vreg 独占栈槽不复用)、peak_real_pressure 口径为累计值非瞬时共存值(指标偏大)、compute_live_intervals O(V×N) 且假设 inst.id 连续有序。前提: BB 内线性扫描范围。这些是下一步优化的直接靶点。

[2026-08-22] RA 演进共识: BB 内线性扫描(Poletto & Sarkar)起点正确；下一个里程碑应是 CFG 全局 liveness(依赖课题11)，而非继续打磨 BB 内启发式。对标 LLVM greedy RA 的差距清单按优先级: 全局 liveness > spill weight 加权(uses×loop_depth/span) > slot 复用 > rematerialization > callee-saved 成本模型 > FP register class(machine_types 尚无此概念)。W9(callee-saved prologue/epilogue)是 W10(--regalloc=linear 管线集成)的硬前置。

## 工程卫生债

[2026-08-22] CI 冗余: .github/workflows/ci.yml L71 `pytest tests/` 已收集 test_pr37_regression.py，L75-76 的 "Run PR #37 register-allocation regressions" step 完全重复执行(dd45a27 引入)。该 commit body 还引用了 "PR #49"，与所在分支 pr-37 易混淆。

[2026-08-22] 文档漂移: docs/topics/17-寄存器分配.md 头部写源文件 `regalloc_linear.py | ~500 行`，实际演进产物为 `regalloc_linear_v1_5.py`(818 行)且两者并存; docs/topic17_v1.3开发/设计文档.md 仍被 git 追踪但对应代码文件已在 rename 中消失(孤儿文档)。

[2026-08-22] pr-37 分支 commit log 规范问题(截至 dd45a27): `topic17:`/`topic17(v1.5):` 非 conventional type; 1be4eb9 中文 subject 违反英文 commit 规则; b28b53b (+1062行) 无 body。版本号进文件名(v1_3→v1_4→v1_5 每次 rename)与 git 版本管理职责重复，建议收敛单一正本模块。

## 开发环境事实

[2026-08-22] 本机无 python3.12(CI 用它)；可用解释器: /usr/bin/python3=3.8、/usr/local/bin/python3.11(无 pytest)、`.venv/bin/python`=Python 3.8 带 pytest。v1_5 代码靠 `from __future__ import annotations` 在 3.8 下可跑(PEP 604 注解不求值)。`rg` 不存在，用 grep。测试命令: `.venv/bin/python -m pytest tests/test_pr37_regression.py`。

## 文档输出位置惯例

[2026-08-22] PR/review 类评审文档落位: `/root/Lab/GaoMD/ScratchV/Review/CI-Review/pr-XX.md`(先例 pr-17.md、pr-37.md)；设计类 review 在 `Review/review文档/`。全局默认路径规则见 ~/.config/opencode/AGENTS.md。

## 流程经验

[2026-08-22] PR#37 教训: 新模块首版提交文件名带点号(regalloc_linear_v1.3.py 不可 import)，靠下个 commit 补救——提交前最小冒烟(import + 一条 smoke 测试)应成为硬门槛。回归测试用 `pytest.importorskip("scratchv.backend.regalloc_linear_v1_5")` 实现 main 分支自动跳过、PR 分支激活，此模式可复用。
