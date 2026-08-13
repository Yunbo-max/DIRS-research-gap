# DIRS 与 Research Gap 对话记录

- 保存日期：2026-08-04
- 主题：DIRS 在当前仓库中的作用，以及它是否已经学会寻找 research gap

## User

> can you chat 8in chines and btell me what did teh DIRs doing here and did it leanr the any how to ifjd a readhc gap expeirenece already?

## Assistant

可以。你说的应该是：“DIRS 在这里做什么？它有没有已经学会寻找 research gap（研究空白）的经验？”

结论：**它已经学到了一套寻找 research gap 的结构化套路，但还不能证明它已经能可靠地发现真正新颖的 research gap。**

DIRS 不是在训练新的 LLM 参数。它把专家论文中的经验提取成一个有向技能图：

```text
证据 → 现有方法 → 失败条件 → research gap
     → 方法需求 → 实验验证 → verifier → 修正
```

节点保存可复用技能，边保存推理顺序；生成时再从图中选择合适路径。[DIRS 总体说明](../README.md#overview)

目前已经完成的部分：

- 读取了 20 篇 LLM inference/system 论文，学到了 constraint mismatch、static-policy、exactness-vs-speed、objective misalignment 等 gap 类型。[20 篇论文学习报告](../case1_writing/runs/llm_inference_systems_fullpaper_train20_section_gap_20260722/FULL_PAPER_20_SECTION_GAP_LEARNING_REPORT.md#what-was-learned)
- Case 3 用 20 篇论文生成并排序了 5 个研究问题。最高排名问题是：能否让 7B/8B tool agent 在 token、工具延迟、GPU 时间和状态可靠性的联合预算下进行规划。[Case 3 结果](../case3_research_question_proposal/runs/gpu_agent_efficiency_gap_convergence_20260721/README.md#ranked-questions)
- 在 `p-less Sampling` 单篇案例中，盲模拟恢复了论文的核心 gap，内部 verifier 给出通过；但精确复现没有完成，而且只完成了约 25% 的计划生成，部分原始结果和计时数据缺失。[单篇验证报告](../case1_writing/runs/p_less_onepaper_dag_blind_sim_verify_20260722/ONEPAPER_GAP_SKILL_GRAPH_TWO_LOOP_REPORT.md#final-status)

为什么还不能说“完全学会”：

- Case 3 的状态文件互相有些矛盾：简单测试写着 `converged`，更严格的 long-goal control 则写着 `converged: false / not_ready`。[严格 convergence 报告](../case3_research_question_proposal/runs/gpu_agent_efficiency_gap_convergence_20260721/LONGGOAL_CONVERGENCE_REPORT.md#status)
- 对其余 19 篇论文的专业规模测试是 `0 accepted / 19`，其中多数因模型、数据、算力或完整实验网格缺失而被阻塞，并不等于全部失败，但也没有形成泛化证据。[19 篇严格审计](../case1_writing/runs/remaining19_p_less_style_dirs_20260723/STRICT_DIRS_COMPLETION_AUDIT_20260723.md#strict-dirs-completion-audit)
- Case 4 已经定义了怎样检索反例、检查近似工作并给出 `partial_gap/already_solved/unverified` 等判断，但暂时主要是协议，没有独立的 Case 4 实验结果。
- 当前没有 DIRS 训练或 supervisor 进程正在运行；最新状态文件停在 2026-07-24。

所以最准确的说法是：**DIRS 已经拥有“怎样寻找和收窄 research gap”的经验库与一个初步成功案例；但跨论文、跨领域、外部文献检索下的真实新颖性能力仍未验证完成。**

## User

> can you save our conversation into the /media/scail/HDD_Storage2/Yunbo/negative-preference-optimization/summer2026 as a md file
