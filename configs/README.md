# Agent Configurations

- `demo.json` runs a dependency-free synthetic recommendation benchmark and verifies the complete agent control loop.
- `demo-llm.json` runs the same benchmark with an API-key Gemini 3.7 Flash research planner.
- `competition.example.json` is intentionally non-runnable until the organizer assets, checksums, official scores, commands, and approved experiment catalog are supplied.
- `kuairand-public-research.json` runs on the real public KuaiRand-Pure rows using a clearly non-competition observed-impression protocol. It is a scalability/integration check, not an official score.

The competition profile enforces KuaiRand-Pure, `is_click`, NDCG@10, Recall@50, 50 iterations, a 21,600-second wall-clock limit, and epsilon/patience `0.002/3`. Do not weaken the profile to make preflight pass.

All commands are argument arrays and execute without a shell. Supported placeholders are:

- `{python}`
- `{workspace}`
- `{config_dir}`
- `{run_dir}`
- `{result_path}`
- `{submission_path}`
- `{experiment_id}`
- `{strategy}`
- `{evidence_path}`
- `{decision_path}`

Each benchmark command must write a result JSON object:

```json
{
  "status": "succeeded",
  "metrics": {
    "NDCG@10": 0.0,
    "Recall@50": 0.0
  },
  "official_selection_score": 0.0,
  "resources": {
    "gpu_hours": 0.0,
    "llm_tokens": 0
  }
}
```

Only include `official_selection_score` when the organizer defines it. Otherwise configure an official metric as `metrics.selection`.

An external research planner receives the EvidencePack path and must write a decision containing an approved `experiment_id`, a reason, supporting evidence, and its LLM-token usage. It cannot supply an arbitrary command; execution remains restricted to the reviewed experiment catalog.

## LLM planner

Use `planner.mode: "llm"` to call Gemini or OpenAI with structured output. The default LLM configuration uses Gemini 3.7 Flash. The API key is loaded from the ignored `.env` file and is never placed in JSON configuration, command arguments, the EvidencePack, or run reports.

```json
{
  "planner": {
    "mode": "llm",
    "provider": "gemini",
    "model": "gemini-3.7-flash",
    "api_key_env": "GEMINI_API_KEY",
    "env_file": ".env",
    "api_timeout_seconds": 60,
    "max_output_tokens": 1200,
    "timeout_seconds": 75,
    "fallback_to_catalog": false
  }
}
```

The model may select exactly one currently available experiment ID and provide a concise reason plus evidence. A dynamic JSON Schema restricts the ID to the current catalog. The orchestrator validates the response again and retains control of commands, metrics, split policy, promotion, convergence, and finalization.

Put the key in the root `.env` file:

```dotenv
GEMINI_API_KEY=your_real_key_here
```

Then run:

```powershell
python -m automl_agent --config configs/demo-llm.json preflight
python -m automl_agent --config configs/demo-llm.json run --run-id llm-demo-001
```

The loader supports comments, `KEY=value`, and single- or double-quoted values. Existing process environment variables take precedence over `.env`. Set `fallback_to_catalog` to `true` only when continuing with the deterministic planner is preferable to failing the run. The run record distinguishes `llm`, `command`, and `catalog_fallback` decisions. Offline tests exercise both Gemini and OpenAI parsing without a key or network request.
