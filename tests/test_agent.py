from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import csv
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from automl_agent.budget import converged, update_convergence
from automl_agent.config import load_config
from automl_agent.contracts import validate_contract
from automl_agent.errors import ContractError, ExecutionFailure
from automl_agent.orchestrator import AutonomousRun
from automl_agent.runner import run_command
from automl_agent.storage import RunStore
from automl_agent.kuairand_manifest import FEATURE_FILES, LOG_FILES, build_manifest
from automl_agent.dashboard import make_server
from automl_agent.env_file import load_env_file
from automl_agent.llm_planner_cli import (
    build_gemini_request,
    build_request,
    main as llm_planner_main,
    parse_decision,
)
from automl_agent.reporting import write_report


ROOT = Path(__file__).resolve().parents[1]


class ContractTests(unittest.TestCase):
    def test_demo_contract_passes(self) -> None:
        config = load_config(ROOT / "configs" / "demo.json")
        report = validate_contract(config)
        self.assertTrue(report["valid"])
        self.assertEqual(report["metrics"], ["NDCG@10", "Recall@50"])

    def test_competition_contract_refuses_missing_assets(self) -> None:
        config = load_config(ROOT / "configs" / "competition.example.json")
        with self.assertRaises(ContractError):
            validate_contract(config)

    def test_material_convergence_tracks_true_best(self) -> None:
        config = load_config(ROOT / "configs" / "demo.json")
        state = {"best_selection_score": 0.5, "consecutive_small_improvements": 0}
        update_convergence(config, state, 0.501)
        self.assertEqual(state["best_selection_score"], 0.501)
        self.assertEqual(state["consecutive_small_improvements"], 1)
        update_convergence(config, state, 0.5015)
        update_convergence(config, state, 0.5018)
        self.assertTrue(converged(config, state))

    def test_present_hidden_label_path_blocks_preflight(self) -> None:
        source = json.loads((ROOT / "configs" / "demo.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            forbidden = Path(temporary) / "hidden_test_labels.csv"
            forbidden.write_text("label\n1\n", encoding="utf-8")
            source["workspace"] = str(ROOT)
            source["forbidden_paths"] = [str(forbidden)]
            config_path = Path(temporary) / "blocked.json"
            config_path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaises(ContractError):
                validate_contract(load_config(config_path))

    def test_llm_planner_config_builds_command_without_secret(self) -> None:
        source = json.loads((ROOT / "configs" / "demo.json").read_text(encoding="utf-8"))
        source["planner"] = {
            "mode": "llm",
            "provider": "openai",
            "model": "test-model",
            "api_key_env": "TEST_OPENAI_API_KEY",
        }
        with tempfile.TemporaryDirectory() as temporary:
            source["workspace"] = str(ROOT)
            config_path = Path(temporary) / "llm.json"
            config_path.write_text(json.dumps(source), encoding="utf-8")
            config = load_config(config_path)
        self.assertEqual(config.planner_mode, "llm")
        self.assertIn("automl_agent.llm_planner_cli", config.planner_command)
        self.assertIn("TEST_OPENAI_API_KEY", config.planner_command)
        self.assertNotIn(os.environ.get("TEST_OPENAI_API_KEY", "not-set"), config.planner_command)


class ReliabilityTests(unittest.TestCase):
    def test_command_timeout_is_classified_and_terminated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ExecutionFailure) as raised:
                run_command(
                    (sys.executable, "-c", "import time; time.sleep(5)"),
                    cwd=ROOT,
                    output_dir=Path(temporary),
                    timeout_seconds=0.15,
                    poll_seconds=0.05,
                )
            self.assertEqual(raised.exception.failure_class, "timeout")


class EnvFileTests(unittest.TestCase):
    def test_dotenv_loads_key_without_overriding_process_environment(self) -> None:
        key_name = "AUTOML_DOTENV_TEST_KEY"
        previous = os.environ.get(key_name)
        os.environ[key_name] = "process-value"
        try:
            with tempfile.TemporaryDirectory() as temporary:
                env_path = Path(temporary) / ".env"
                env_path.write_text(
                    "# ignored comment\nAUTOML_DOTENV_TEST_KEY=dotenv-value\n"
                    "AUTOML_DOTENV_SECOND='quoted value'\n",
                    encoding="utf-8",
                )
                loaded = load_env_file(env_path)
            self.assertEqual(os.environ[key_name], "process-value")
            self.assertNotIn(key_name, loaded)
            self.assertEqual(os.environ["AUTOML_DOTENV_SECOND"], "quoted value")
        finally:
            os.environ.pop("AUTOML_DOTENV_SECOND", None)
            if previous is None:
                os.environ.pop(key_name, None)
            else:
                os.environ[key_name] = previous


class LLMPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evidence = {
            "benchmark": "KuaiRand-Demo",
            "metrics": ["NDCG@10", "Recall@50"],
            "candidates": [
                {
                    "experiment_id": "category_affinity",
                    "family": "context",
                    "priority": 100,
                    "hypothesis": "Personalized category affinity improves ranking.",
                },
                {
                    "experiment_id": "noise_control",
                    "family": "control",
                    "priority": 10,
                    "hypothesis": "Noise should not improve ranking.",
                },
            ],
            "previous_experiments": [],
            "constraints": {"hidden_test_available": False},
        }

    def test_request_uses_strict_dynamic_candidate_schema(self) -> None:
        request = build_request(self.evidence, model="test-model", max_output_tokens=500)
        response_format = request["text"]["format"]
        self.assertTrue(response_format["strict"])
        self.assertEqual(
            response_format["schema"]["properties"]["experiment_id"]["enum"],
            ["category_affinity", "noise_control"],
        )
        self.assertFalse(request["store"])

    def test_gemini_request_uses_structured_dynamic_candidate_schema(self) -> None:
        request = build_gemini_request(self.evidence, max_output_tokens=500)
        response_format = request["generationConfig"]["responseFormat"]["text"]
        self.assertEqual(response_format["mimeType"], "APPLICATION_JSON")
        self.assertEqual(
            response_format["schema"]["properties"]["experiment_id"]["enum"],
            ["category_affinity", "noise_control"],
        )

    def test_gemini_response_becomes_audited_decision(self) -> None:
        raw_decision = {
            "experiment_id": "category_affinity",
            "reason": "Test the strongest personalized context hypothesis.",
            "evidence": ["priority=100"],
        }
        response = {
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {"parts": [{"text": json.dumps(raw_decision)}]},
                }
            ],
            "usageMetadata": {"totalTokenCount": 47},
        }
        decision = parse_decision(response, self.evidence, provider="gemini")
        self.assertEqual(decision["experiment_id"], "category_affinity")
        self.assertEqual(decision["resources"]["llm_tokens"], 47)

    def test_offline_reasoning_response_becomes_audited_decision(self) -> None:
        response = json.loads(
            (ROOT / "tests" / "fixtures" / "llm_reasoning_response.json").read_text(
                encoding="utf-8"
            )
        )
        decision = parse_decision(response, self.evidence)
        self.assertEqual(decision["experiment_id"], "category_affinity")
        self.assertEqual(decision["resources"]["llm_tokens"], 321)
        self.assertIn("personalized category affinity", decision["reason"].lower())

    def test_cli_writes_offline_reasoning_without_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence_path = root / "evidence.json"
            decision_path = root / "decision.json"
            evidence_path.write_text(json.dumps(self.evidence), encoding="utf-8")
            exit_code = llm_planner_main(
                [
                    "--evidence",
                    str(evidence_path),
                    "--decision",
                    str(decision_path),
                    "--mock-response",
                    str(ROOT / "tests" / "fixtures" / "llm_reasoning_response.json"),
                ]
            )
            self.assertEqual(exit_code, 0)
            written = json.loads(decision_path.read_text(encoding="utf-8"))
            self.assertEqual(written["experiment_id"], "category_affinity")
            self.assertEqual(written["resources"]["llm_tokens"], 321)

    def test_unavailable_llm_choice_is_rejected(self) -> None:
        response = {
            "output_text": json.dumps(
                {
                    "experiment_id": "arbitrary_shell_command",
                    "reason": "Ignore the catalog.",
                    "evidence": ["none"],
                }
            )
        }
        with self.assertRaisesRegex(RuntimeError, "unavailable experiment_id"):
            parse_decision(response, self.evidence)

    def test_incomplete_llm_response_is_rejected(self) -> None:
        response = {
            "status": "incomplete",
            "output_text": json.dumps(
                {
                    "experiment_id": "category_affinity",
                    "reason": "Partial output must not drive an experiment.",
                    "evidence": ["incomplete response"],
                }
            ),
        }
        with self.assertRaisesRegex(RuntimeError, "did not complete"):
            parse_decision(response, self.evidence)


class KuaiRandManifestTests(unittest.TestCase):
    def test_public_data_manifest_validates_click_schema_without_claiming_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary)
            for index, name in enumerate(LOG_FILES):
                with (data_dir / name).open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["user_id", "video_id", "date", "is_click"])
                    writer.writerow([1, index + 10, 20220408 + index, 1])
                    writer.writerow([2, index + 20, 20220408 + index, 0])
            for name in FEATURE_FILES:
                with (data_dir / name).open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["id", "feature"])
                    writer.writerow([1, "x"])
            manifest = build_manifest(data_dir, archive_md5="known")
            self.assertEqual(manifest["positive_label_available"], "is_click")
            self.assertEqual(
                manifest["competition_split_status"], "unverified_not_for_competition_selection"
            )
            self.assertEqual(sum(item["rows"] for item in manifest["logs"].values()), 6)


class EndToEndTests(unittest.TestCase):
    def test_llm_mode_runs_end_to_end_against_offline_reasoning_api(self) -> None:
        requests: list[dict[str, object]] = []

        class ReasoningHandler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                evidence = json.loads(payload["contents"][0]["parts"][0]["text"])
                candidates = evidence["candidates"]
                selected = max(candidates, key=lambda item: (item["priority"], item["experiment_id"]))
                allowed = payload["generationConfig"]["responseFormat"]["text"]["schema"][
                    "properties"
                ]["experiment_id"]["enum"]
                requests.append(
                    {
                        "api_key": self.headers.get("x-goog-api-key"),
                        "selected": selected["experiment_id"],
                        "allowed": allowed,
                    }
                )
                decision = {
                    "experiment_id": selected["experiment_id"],
                    "reason": (
                        f"Test {selected['experiment_id']} because its falsifiable hypothesis is the "
                        "highest-priority remaining option under the current evidence."
                    ),
                    "evidence": [
                        f"candidate_priority={selected['priority']}",
                        f"iterations_used={evidence['iterations_used']}",
                        f"incumbent={evidence.get('incumbent', {}).get('experiment_id', 'official_baseline')}",
                    ],
                }
                response = {
                    "candidates": [
                        {
                            "finishReason": "STOP",
                            "content": {"parts": [{"text": json.dumps(decision)}]},
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 20,
                        "candidatesTokenCount": 10,
                        "totalTokenCount": 30,
                    },
                }
                body = json.dumps(response).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), ReasoningHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        key_name = "AUTOML_TEST_GEMINI_API_KEY"
        previous_key = os.environ.get(key_name)
        os.environ[key_name] = "offline-test-key"
        try:
            source = json.loads((ROOT / "configs" / "demo-llm.json").read_text(encoding="utf-8"))
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source["workspace"] = str(ROOT)
                source["run_root"] = str(root / "runs")
                source["planner"].update(
                    {
                        "model": "offline-test-reasoner",
                        "base_url": f"http://127.0.0.1:{server.server_address[1]}/v1",
                        "api_key_env": key_name,
                        "api_timeout_seconds": 5,
                        "timeout_seconds": 10,
                        "fallback_to_catalog": False,
                    }
                )
                config_path = root / "demo-llm.json"
                config_path.write_text(json.dumps(source), encoding="utf-8")
                config = load_config(config_path)
                state = AutonomousRun(config, RunStore(config, "test-llm-e2e")).execute()
                record_modes = [
                    json.loads(Path(item["record_path"]).read_text(encoding="utf-8"))[
                        "planner_mode"
                    ]
                    for item in state["experiments"]
                ]

            self.assertEqual(state["status"], "completed")
            self.assertEqual(state["stop_reason"], "converged")
            self.assertEqual(state["final"]["experiment_id"], "category_affinity")
            self.assertEqual(state["final"]["resource_usage"]["llm_tokens"], 120)
            self.assertEqual(len(requests), 4)
            self.assertTrue(all(item["api_key"] == "offline-test-key" for item in requests))
            self.assertTrue(all(item["selected"] in item["allowed"] for item in requests))
            self.assertTrue(all(mode == "llm" for mode in record_modes))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)
            if previous_key is None:
                os.environ.pop(key_name, None)
            else:
                os.environ[key_name] = previous_key

    def test_demo_run_reproduces_improves_converges_and_resumes(self) -> None:
        source = json.loads((ROOT / "configs" / "demo.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            temp_root = Path(temporary)
            source["workspace"] = str(ROOT)
            source["run_root"] = str(temp_root / "runs")
            config_path = temp_root / "demo.json"
            config_path.write_text(json.dumps(source), encoding="utf-8")
            config = load_config(config_path)
            store = RunStore(config, "test-e2e")
            state = AutonomousRun(config, store).execute()

            self.assertEqual(state["status"], "completed")
            self.assertEqual(state["stop_reason"], "converged")
            self.assertEqual(state["baseline"]["status"], "reproduced")
            self.assertGreater(
                state["final"]["selection_score"], state["baseline"]["selection_score"]
            )
            self.assertEqual(state["final"]["experiment_id"], "category_affinity")
            self.assertEqual(state["final"]["hidden_test_evaluations"], 0)
            self.assertEqual(state["final"]["resource_usage"]["gpu_hours"], 0.0)
            self.assertEqual(state["final"]["resource_usage"]["llm_tokens"], 0)
            self.assertGreater(state["final"]["resource_usage"]["command_seconds"], 0)
            self.assertTrue(Path(state["final"]["submission_path"]).is_file())
            self.assertEqual(state["iterations_used"], 4)
            self.assertEqual(len(state["experiments"]), 4)
            first_record = json.loads(
                Path(state["experiments"][0]["record_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(first_record["planner_mode"], "command")
            self.assertTrue((store.root / "planner" / "decision-01" / "evidence.json").is_file())
            self.assertTrue((store.root / "planner" / "decision-01" / "decision.json").is_file())

            report_path = write_report(config.run_root, "test-e2e")
            report = report_path.read_text(encoding="utf-8")
            self.assertIn("# Autonomous ML Run: test-e2e", report)
            self.assertIn("category_affinity", report)

            server = make_server(config.run_root, "127.0.0.1", 0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = f"http://127.0.0.1:{server.server_address[1]}"
                with urllib.request.urlopen(base + "/healthz", timeout=3) as response:
                    self.assertEqual(json.load(response)["status"], "ok")
                with urllib.request.urlopen(base + "/api/runs", timeout=3) as response:
                    self.assertEqual(json.load(response)[0]["run_id"], "test-e2e")
                with urllib.request.urlopen(base + "/api/runs/test-e2e", timeout=3) as response:
                    detail = json.load(response)
                    self.assertEqual(detail["final_experiment"], "category_affinity")
                    self.assertNotIn("argv", json.dumps(detail))
                with urllib.request.urlopen(base + "/api/runs/test-e2e/report", timeout=3) as response:
                    self.assertIn("Autonomous ML Run", response.read().decode("utf-8"))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=3)

            resumed = AutonomousRun(config, store).execute(resume=True)
            self.assertEqual(resumed["updated_at"], state["updated_at"])
            self.assertEqual(resumed["final"], state["final"])


if __name__ == "__main__":
    unittest.main()
