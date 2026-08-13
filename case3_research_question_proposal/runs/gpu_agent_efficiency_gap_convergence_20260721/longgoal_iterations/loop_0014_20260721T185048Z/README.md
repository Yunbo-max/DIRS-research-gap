# DIRS Gap-Convergence Run: GPU Agent Efficiency

Date: `2026-07-21`

## Input

- Fit-paper shortlist: `/tf/notebooks/yunbo/fit_papers_for_gpu_simulation_research_gap_20260721.md`
- Selected papers loaded: `20`

## Convergence Verdict

- Status: `converged`
- Loops: `4`
- Verifier verdict: `partial_gap`
- Confidence: `medium_high`
- GPU execution gate: `execution_ready`
- GPUs detected: `4`

## Learned Research-Gap DAG

```text
domain topic -> recent pattern -> unresolved uncertainty -> missing condition
  -> mechanism hypothesis -> target object -> research question
  -> possible evidence -> GPU probe -> repo readiness -> smoke benchmark plan
  -> budget metric schema -> execution verifier -> bounded scope
  -> gap audit -> verdict
```

## Ranked Questions

### Q1_budgeted_state_verifying_agents [pursue_now, score 12/12]

Can a 7B/8B tool agent improve dynamic multi-turn task success by planning under an explicit joint budget for tokens, tool latency, GPU time, and simulator-verified state reliability?

- Supported dimensions: `5/5`
- Experiment: Run tau2-Bench, SimuHome, and Gaia2 subsets with a Qwen/Llama-class local agent. Compare baseline ReAct against a budget-aware planner that rejects plans failing simulator/verifier checks.
- Support papers: Gaia2: Benchmarking LLM Agents on Dynamic and Asynchronous Environments, In-The-Flow Agentic System Optimization for Effective Planning and Tool Use, LLMs Get Lost In Multi-Turn Conversation, OMAC: A Holistic Optimization Framework for LLM-Based Multi-Agent Collaboration, RedTeamCUA: Realistic Adversarial Testing of Computer-Use Agents in Hybrid Web-OS Environments, ScaleCUA: Scaling Open-Source Computer Use Agents with Cross-Platform Data, SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home LLM Agents, Speculative Actions: A Lossless Framework for Faster AI Agents

### Q2_reversible_speculative_tool_actions [pursue_now, score 12/12]

When agent actions are typed by reversibility, can speculative tool-call prefetch reduce p95 latency without increasing wrong or unsafe committed actions?

- Supported dimensions: `4/4`
- Experiment: Use Speculative Actions plus tau2/SimuHome traces. Allow speculation only for reversible reads/prefetches, then measure hit rate, latency, extra token cost, and commit errors.
- Support papers: Gaia2: Benchmarking LLM Agents on Dynamic and Asynchronous Environments, In-The-Flow Agentic System Optimization for Effective Planning and Tool Use, LLMs Get Lost In Multi-Turn Conversation, OMAC: A Holistic Optimization Framework for LLM-Based Multi-Agent Collaboration, RedTeamCUA: Realistic Adversarial Testing of Computer-Use Agents in Hybrid Web-OS Environments, ScaleCUA: Scaling Open-Source Computer Use Agents with Cross-Platform Data, SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home LLM Agents, Speculative Actions: A Lossless Framework for Faster AI Agents

### Q3_token_memory_policy_for_agents [pursue_now, score 12/12]

Can token, cache, and decoding-efficiency methods be converted into agent-level memory policies that preserve task state while reducing GPU memory and generation cost?

- Supported dimensions: `4/4`
- Experiment: Combine p-less/WeDLM-style decoding probes with multi-turn agent benchmarks. Track state-recall accuracy, final task success, token count, KV/cache footprint proxy, and latency.
- Support papers: Differentiable Model Predictive Control on the GPU, Diffusion Language Models Know the Answer Before Decoding, FlashSketch: Sketch-Kernel Co-Design for Fast Sparse Sketching on GPUs, Gaia2: Benchmarking LLM Agents on Dynamic and Asynchronous Environments, In-The-Flow Agentic System Optimization for Effective Planning and Tool Use, LLMs Get Lost In Multi-Turn Conversation, OMAC: A Holistic Optimization Framework for LLM-Based Multi-Agent Collaboration, RedTeamCUA: Realistic Adversarial Testing of Computer-Use Agents in Hybrid Web-OS Environments

### Q4_gpu_headroom_topic_selector [pursue_now, score 12/12]

Can prescriptive scaling and GPU microbenchmarks identify which agent benchmarks still have 4090-reachable headroom for meaningful 7B method improvements?

- Supported dimensions: `4/4`
- Experiment: Fit prescriptive boundaries over public agent/eval results, then run FlashSketch or decoding microbenchmarks to estimate evaluation cost before selecting final benchmark subsets.
- Support papers: FlashSketch: Sketch-Kernel Co-Design for Fast Sparse Sketching on GPUs, Gaia2: Benchmarking LLM Agents on Dynamic and Asynchronous Environments, In-The-Flow Agentic System Optimization for Effective Planning and Tool Use, LLMs Get Lost In Multi-Turn Conversation, OMAC: A Holistic Optimization Framework for LLM-Based Multi-Agent Collaboration, RedTeamCUA: Realistic Adversarial Testing of Computer-Use Agents in Hybrid Web-OS Environments, ScaleCUA: Scaling Open-Source Computer Use Agents with Cross-Platform Data, SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home LLM Agents

### Q5_coordination_overhead_in_multi_agent_7b_systems [pursue_now, score 12/12]

For 7B multi-agent systems, when does coordination improve accuracy enough to offset extra calls, latency, and context overhead?

- Supported dimensions: `4/4`
- Experiment: Run OMAC-lite and AgentFlow-style controllers on HumanEval/MATH/tool tasks, logging marginal accuracy per extra call, token, and second.
- Support papers: Gaia2: Benchmarking LLM Agents on Dynamic and Asynchronous Environments, In-The-Flow Agentic System Optimization for Effective Planning and Tool Use, LLMs Get Lost In Multi-Turn Conversation, OMAC: A Holistic Optimization Framework for LLM-Based Multi-Agent Collaboration, RedTeamCUA: Realistic Adversarial Testing of Computer-Use Agents in Hybrid Web-OS Environments, SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home LLM Agents, Speculative Actions: A Lossless Framework for Faster AI Agents, VS-Bench: Evaluating VLMs for Strategic Abilities in Multi-Agent Environments

## Reframed Gap

Current work has strong pieces for agent benchmarks, simulator verification, speculative execution, decoding efficiency, and GPU microbenchmarking, but these are usually optimized separately. The supported gap is a 4090-feasible framework for 7B/8B agents that jointly measures and optimizes task success, tool latency, token/KV cost, speculative-action safety, and state-verification reliability in dynamic multi-turn environments.

## GPU / Experiment Gate

- Execution status: `execution_ready`
- Detected GPUs: `4`
- GPU 0: NVIDIA GeForce RTX 4090, total 24564 MiB, free 24198 MiB, driver 535.183.01
- GPU 1: NVIDIA GeForce RTX 4090, total 24564 MiB, free 24198 MiB, driver 535.183.01
- GPU 2: NVIDIA GeForce RTX 4090, total 24564 MiB, free 23187 MiB, driver 535.183.01
- GPU 3: NVIDIA GeForce RTX 4090, total 24564 MiB, free 24198 MiB, driver 535.183.01

First smoke order:

- SimuHome: run minimal simulator/verifier episode without model call
- tau2-Bench: run CLI/import smoke and one tiny environment task
- Speculative Actions: run reversible/read-only toy speculation task
- Gaia2: run local scenario/parser/verifier smoke
- AgentFlow: run import/config smoke before any model serving

Required budget metrics:

- `task_success`
- `state_verifier_pass_rate`
- `tool_calls`
- `tokens_in`
- `tokens_out`
- `wall_clock_latency`
- `gpu_memory_peak_mb`
- `gpu_utilization_snapshot`
- `wrong_commit_count`
- `speculation_hit_rate`

## Output Files

- `domain_skill_library.json`
- `skill_graph.yaml`
- `node_library.json`
- `edge_library.json`
- `ranked_research_questions.json`
- `gpu_probe.json`
- `gpu_execution_plan.json`
- `verifier_result.json`
- `training_trace.jsonl`
