# Competition Readiness

## Current status

The autonomous control plane is executable and verified on the included demo benchmark. A competition-valid KuaiRand-Pure run is intentionally blocked because the organizer package is not present in this workspace.

| Requirement | Current evidence | Status |
|---|---|---|
| Contract enforcement | Competition profile fixes `is_click`, NDCG@10, Recall@50, 50 iterations, 6 hours, and epsilon/patience `0.002/3` | Ready |
| Hidden-test isolation | Forbidden-path preflight and regression test | Ready at control-plane boundary |
| Official baseline reproduction | No organizer baseline implementation or reported score is present | Blocked externally |
| Official evaluator | Existing `evaluate.py` is GAUC/nDCG@5 for `long_view`, not the required evaluator | Blocked externally |
| Official train/validation split | Public KuaiRand files and the legacy date split are not proof of the competition split | Blocked externally |
| Submission schema/validator | Existing legacy schema is not confirmed by the organizer package | Blocked externally |
| Autonomous planning | External EvidencePack/decision protocol with catalog safety and fallback | Ready |
| Iteration/wall-time controls | Persistent counters and subprocess deadlines | Ready |
| Recovery | Transient retry, timeout termination, interrupted-run recovery, failure records | Ready |
| Convergence | Persistent epsilon/patience tracker tested end to end | Ready |
| Final designation | Validation-best submission copied, revalidated, hashed, and manifested | Ready |
| GPU/token/intervention reporting | Aggregated in state and final manifest | Ready |

## Organizer assets needed

Place verified assets under a non-agent-owned directory such as `official/`:

1. official KuaiRand-Pure baseline code and exact validation score(s);
2. official evaluator and its single model-selection rule, if one exists;
3. official train and public-validation split or deterministic split manifest;
4. official submission example, schema, and validator;
5. environment/dependency specification;
6. any organizer definition of what consumes one of the 50 iterations.

Then update `configs/competition.example.json` with real paths, checksums, commands, scores, and an approved experiment catalog. Rename it to a local competition config if desired. Do not weaken preflight.

## Verification commands

```powershell
python -m automl_agent --config configs/competition.example.json preflight
python -m unittest discover -s tests -v
```

Preflight must remain red until every organizer asset is authentic and complete. Once green, the next gate is official baseline reproduction; research iterations still cannot begin before that score matches.

## Public dataset preparation

The official public KuaiRand-Pure archive can be validated separately without treating it as the competition split:

```powershell
./scripts/fetch_kuairand_pure.ps1

# Or regenerate only the manifest:
python -m automl_agent.kuairand_manifest `
  --data-dir KuaiRand-Pure/data `
  --archive-md5 0820331067a3784d9691136f772b35a7 `
  --output data_manifests/kuairand-pure-public.json
```

The resulting manifest records hashes, schemas, dates, click rates, row counts, and the explicit status `unverified_not_for_competition_selection`.

The `kuairand-public-001` integration run also proves that the control plane can process the real public data and produce a validated 295,497-row submission. Its score must not be compared with the official baseline because it ranks observed standard-log impressions rather than a confirmed organizer candidate set.
