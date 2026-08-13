# Research Loop Persistent Operating Memory

Date: 2026-07-22

This project treats research-loop work as strict conference-level author/reviewer simulation, not as a small demo.

## Non-Negotiable Policy

- Do not treat reduced, toy, tiny, smoke, one-prompt, sampler-proxy, or miniature runs as convergence evidence.
- Reduced runs are allowed only for environment checks, harness validation, debugging, or quick failure localization.
- A verifier may accept "close results" only when they come from a professional paper-shaped experiment: the right model family, datasets, temperatures, baselines, seeds, scoring, timing, hardware logs, and artifact packaging.
- If the run is reduced, label it `smoke_only`, `debug_only`, or `preflight_only`; do not label it as gap convergence, paper convergence, or NeurIPS/ICLR-level evidence.
- The loop continues until the verifier says the relevant performance is good enough under the professional-scale gate, or until an external blocker is explicit and actionable.
- The DAG must tell the blind author-agent how to do the real work: clone/download code, resolve models, resolve datasets, build/run scripts, collect raw generations, score outputs, measure GPU/CPU/RAM/timing, compare to tables/figures/paragraphs/appendix, and revise claims.
- For research-gap learning, "close" means close to the paper's real gap and result shape under a serious experiment. It does not mean a small reduced run happens to point in the same direction.
- Always keep the reviewer standard high: NeurIPS/ICLR-level experiments, clear artifact provenance, no hidden proxy success, no premature convergence.

## Loop Semantics

- Loop 1: repair or expand the DAG from verifier failures.
- Loop 2: blind author simulation from the DAG only.
- Verifier: compare simulated gap, method reasoning, and result evidence against the paper's actual tables, figures, paragraphs, appendix, and code artifacts.
- Continue until verifier acceptance under the correct scale gate.
