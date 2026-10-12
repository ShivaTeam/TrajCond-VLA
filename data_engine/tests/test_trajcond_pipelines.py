"""Source, geometry, serialization, filtering, and runner integration tests."""

from copy import deepcopy
from contextlib import redirect_stdout
import importlib.util
import io
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import ModuleType, SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data_engine" / "src"))
from ir.builder import build_sample_ir
from ir.schema import EgoPoint, PseudoLabelIR
from datasets.registry import register_dataset
from pipelines.config import load_yaml, prepare_config
from pipelines.runner import run_pipeline, base_cache_key, instantiate_enrichers, preflight
from prompts.trajcond_stage1 import TrajcondStage1Builder
from prompts.trajcond_stage2 import TrajcondStage2Builder
from utils.action_search import build_searcher, integrate_actions, local_action, SearchResult
from utils.action_search_adapter import validate_result
from utils.trajcond_contract import RELEASE, TEMPLATES
from utils.trajectory import compute_sample_trajectory
from enrichers.encode_trajcond import EncodeTrajcond, init_search_worker
from enrichers.filter_action_error import FilterActionError
from enrichers.validate_trajcond_input import ValidateTrajcondInput
from enrichers.merge_pseudo_labels import MergePseudoLabelsEnricher
from enrichers.merge_gt_trajectories import MergeGtTrajectories
from enrichers.normalize_trajectory import NormalizeTrajectoryEnricher
from writers.json_writer import write_output


def raw_sample(key, image):
    frames = []
    for i in range(12):
        current = i == 3
        frames.append(dict(token=f"{key}-{i}", timestamp=(i+10)*500000,
                           position=[float(i-3), 0, 0],
                           rotation=[1, 0, 0, 0], velocity=[2, 0], acceleration=[0, 0],
                           command="go straight", images={"cam_f0": str(image)} if current else {}))
    return dict(token=key, metadata=dict(log_name="test", scene_token="scene", map_name="map",
                                        num_frames=12, current_frame_index=3), frames=frames)


@register_dataset("synthetic")
class SyntheticAdapter:
    def __init__(self, config):
        self.config = config

    def get_sample_count(self):
        return 3

    def extract_sample(self, index):
        return raw_sample(["c", "a", "b"][index], self.config["image"])


def cot():
    return dict(critical_objects={k: "no" for k in RELEASE["critical_objects"]},
                meta_behaviour={"speed": "keep", "command": "straight"},
                explanation="<thinking><thinking>Keep the lane. 引号 \"test\"\nNext.</thinking></thinking>")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "tmp")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.image = self.path / "front.jpg"
        self.image.write_bytes(b"fixture image; no image decoder is used in data construction")
        self.variance = self.write_json("variances.json", {"variances": [1, 1, 1]})
        self.joint = self.write_json("joint.json", {"k": 3, "codebook": [[0, 0, 0], [1, 0, 0], [2, 0, 0]]})
        self.brohan = self.write_json("brohan.json", {"k": [3, 1, 2], "codebook": [[0, 1, 2], [0], [0, 0.2]]})
        self.stats = self.write_json("stats.json", {"mean": np.zeros((8, 3)).tolist(), "std": (np.ones((8, 3))*2).tolist()})
        self.labels = self.path / "cot.jsonl"
        self.labels.write_text("\n".join(json.dumps({"id": k, "predict": json.dumps(cot(), ensure_ascii=False)}, ensure_ascii=False)
                                         for k in ["a", "b", "c"]), encoding="utf-8")
        self.pkl = self.path / "trajectory.pkl"
        self.pkl.write_bytes(pickle.dumps([
            dict(id="a", gt=[[2*i, 1, 0.4] for i in range(1, 9)]),
            dict(id="c", gt=[[4*i, 3, 0.8] for i in range(1, 9)]),
        ]))

    def write_json(self, name, value):
        path = self.path / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return str(path)

    def sample(self, key="a"):
        sample = compute_sample_trajectory(build_sample_ir(raw_sample(key, self.image)))
        sample.pseudo_label = PseudoLabelIR(**cot())
        return sample

    def config(self, method, stage):
        cfg = load_yaml(ROOT / f"data_engine/configs/pipelines/navsim_{method}_stage{stage}.yaml")
        cfg["dataset"] = dict(name="synthetic", split="train", splits={}, image=str(self.image))
        cfg["cache"] = {"enabled": False}
        cfg["output"]["path"] = str(self.path / f"{method}-{stage}.json")
        for ec in cfg["enrichers"]:
            params, name = ec["params"], ec["name"]
            if name == "replace_image_base":
                params.update(old_base=str(self.path), new_base="training/images")
            elif name == "merge_gt_trajectories":
                params["source_path"] = str(self.pkl)
            elif name == "merge_pseudo_labels":
                params["source_path"] = str(self.labels)
            elif name == "encode_trajcond":
                params.update(codebook_path=self.brohan if method == "brohan" else self.joint, variances_path=self.variance)
        return prepare_config(cfg)

    def test_cli_help_imports(self):
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "data_engine" / "main.py"), "--help"],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--check-config", result.stdout)

    def test_six_full_pipelines_sources_and_formats(self):
        for method in ("kmeans", "kdisks", "brohan"):
            answers = {}
            for stage, builder in [(1, TrajcondStage1Builder()), (2, TrajcondStage2Builder())]:
                cfg = self.config(method, stage)
                search_output = io.StringIO()
                with redirect_stdout(search_output):
                    run_pipeline(cfg, builder)
                log = search_output.getvalue()
                self.assertEqual(log.count("--- Reconstruction error statistics"), 1)
                self.assertIn(f"Valid searched samples: {3 if stage == 1 else 2}", log)
                self.assertLess(log.index("--- Reconstruction error statistics"), log.index("Action error filter:"))
                records = json.loads(Path(cfg["output"]["path"]).read_text(encoding="utf-8"))
                serial_bytes = Path(cfg["output"]["path"]).read_bytes()
                cfg["num_workers"] = 2
                run_pipeline(cfg, builder)
                self.assertEqual(Path(cfg["output"]["path"]).read_bytes(), serial_bytes)
                self.assertEqual([r["id"] for r in records], ["a", "b", "c"] if stage == 1 else ["a", "c"])
                if stage == 2:
                    self.assertEqual(list(records[0]), ["trajectory_stats", "id", "image", "system", "conversations"])
                    stats = records[0]["trajectory_stats"]
                    np.testing.assert_allclose(stats["mean"], [[3*i, 2, 0.6] for i in range(1, 9)])
                    np.testing.assert_allclose(stats["std"], [[i, 1, 0.2] for i in range(1, 9)])
                    self.assertTrue(all("trajectory_stats" not in r for r in records[1:]))
                else:
                    self.assertTrue(all("trajectory_stats" not in r for r in records))
                answer = json.loads(records[0]["conversations"][1]["value"])
                answers[stage] = answer
                self.assertEqual(list(answer), builder.spec["answer_fields"])
                self.assertEqual(answer["explanation"].count("<thinking>"), 1)
                self.assertIn('引号 "test"\nNext.', answer["explanation"])
                self.assertTrue(records[0]["conversations"][0]["value"].endswith("\n"))
                action = answer["future_trajectory" if stage == 1 else "discrete_action_trajectory"]
                self.assertEqual(len(re.findall(r"<action_\d+>", action)), 24 if method == "brohan" else 8)
                if method == "brohan":
                    self.assertEqual(action, "<answer>[AT, " + ", ".join(["(<action_1>,<action_3>,<action_4>)"]*8) + "]</answer>")
                else:
                    self.assertEqual(action, "<answer>[AT, " + "<action_1>"*8 + "]</answer>")
                if stage == 2:
                    self.assertEqual(answer["future_trajectory"],
                                     "<answer>[PT, " + ", ".join(["(-1.00, -1.00, -1.00)"]*8) + "]</answer>")
                    other_answer = json.loads(records[1]["conversations"][1]["value"])
                    self.assertEqual(other_answer["future_trajectory"],
                                     "<answer>[PT, " + ", ".join(["(+1.00, +1.00, +1.00)"]*8) + "]</answer>")
            self.assertEqual(answers[1]["future_trajectory"], answers[2]["discrete_action_trajectory"])
            for key in ("critical_objects", "meta_behaviour", "explanation"):
                self.assertEqual(answers[1][key], answers[2][key])

    def test_stage2_normalization_without_stats_file(self):
        cfg = self.config("kmeans", 2)
        params = next(e["params"] for e in cfg["enrichers"] if e["name"] == "normalize_trajectory")
        self.assertFalse(params["require_existing_stats"])
        self.assertNotIn("stats_file", params)
        preflight(cfg, TrajcondStage2Builder(), instantiate_enrichers(cfg["enrichers"]))
        samples = MergeGtTrajectories(str(self.pkl)).enrich_all([self.sample("a"), self.sample("c")])
        normalizer = NormalizeTrajectoryEnricher(source_field="target_trajectory",
                                                output_field="normalized_target_trajectory")
        files_before = set(self.path.iterdir())
        normalizer.enrich_all(samples)
        expected_mean = [[3*i, 2, 0.6] for i in range(1, 9)]
        expected_std = [[i, 1, 0.2] for i in range(1, 9)]
        np.testing.assert_allclose(normalizer._mean, expected_mean)
        np.testing.assert_allclose(normalizer._std, expected_std)
        self.assertEqual(set(self.path.iterdir()), files_before)
        for sample, value in zip(samples, [-1, 1]):
            points = sample.normalized_target_trajectory
            np.testing.assert_allclose([[p.x, p.y, p.heading] for p in points], value)
            self.assertEqual([p.t for p in points], [i*0.5 for i in range(1, 9)])
        # A new retained set recomputes in-memory stats, including the std floor.
        normalizer.enrich_all(samples[:1])
        np.testing.assert_allclose(normalizer._std, 1e-6)
        points = samples[0].normalized_target_trajectory
        np.testing.assert_allclose([[p.x, p.y, p.heading] for p in points], 0)
        self.assertEqual(set(self.path.iterdir()), files_before)
        filter_params = next(e["params"] for e in cfg["enrichers"] if e["name"] == "filter_action_error")
        filter_params["drop_fraction"] = 0.5
        run_pipeline(cfg, TrajcondStage2Builder())
        records = json.loads(Path(cfg["output"]["path"]).read_text(encoding="utf-8"))
        self.assertEqual([r["id"] for r in records], ["c"])
        np.testing.assert_allclose(records[0]["trajectory_stats"]["mean"], [[4*i, 3, 0.8] for i in range(1, 9)])
        np.testing.assert_allclose(records[0]["trajectory_stats"]["std"], 1e-6)
        answer = json.loads(records[0]["conversations"][1]["value"])
        self.assertEqual(answer["future_trajectory"],
                         "<answer>[PT, " + ", ".join(["(0.00, 0.00, 0.00)"]*8) + "]</answer>")

    def test_normalization_optional_stats_file_modes(self):
        cfg = self.config("kmeans", 2)
        params = next(e["params"] for e in cfg["enrichers"] if e["name"] == "normalize_trajectory")
        params["stats_file"] = str(self.path / "new_stats.json")
        preflight(cfg, TrajcondStage2Builder(), instantiate_enrichers(cfg["enrichers"]))
        samples = MergeGtTrajectories(str(self.pkl)).enrich_all([self.sample("a"), self.sample("c")])
        options = dict(source_field="target_trajectory", output_field="normalized_target_trajectory")
        normalizer = NormalizeTrajectoryEnricher(**params)
        normalizer.enrich_all(samples)
        self.assertTrue(Path(params["stats_file"]).is_file())
        cached = NormalizeTrajectoryEnricher(stats_file=params["stats_file"], **options)
        with patch.object(cached, "_collect_stats", side_effect=AssertionError("Existing stats must be reused")):
            cached.enrich_all(samples)
        np.testing.assert_allclose(cached._mean, normalizer._mean)
        fixed = NormalizeTrajectoryEnricher(stats_file=self.stats, require_existing_stats=True, **options)
        fixed.enrich_all(samples)
        first = samples[0].normalized_target_trajectory[0]
        np.testing.assert_allclose([first.x, first.y, first.heading], [1, 0.5, 0.2])
        params.update(stats_file=self.stats, require_existing_stats=True)
        run_pipeline(cfg, TrajcondStage2Builder())
        records = json.loads(Path(cfg["output"]["path"]).read_text(encoding="utf-8"))
        self.assertEqual(records[0]["trajectory_stats"], json.loads(Path(self.stats).read_text(encoding="utf-8")))
        with self.assertRaisesRegex(ValueError, "stats_file is required"):
            NormalizeTrajectoryEnricher(require_existing_stats=True, **options)
        with self.assertRaises(FileNotFoundError):
            NormalizeTrajectoryEnricher(stats_file=str(self.path / "missing.json"),
                                       require_existing_stats=True, **options)

    def test_search_geometry_wrap_and_actual_size(self):
        previous = np.array([2., 3., np.pi-0.05])
        action = np.array([1., 0.2, 0.1])
        target = integrate_actions(previous, action)
        np.testing.assert_allclose(local_action(previous, target), action, atol=1e-14)
        searcher = build_searcher(method="brohan", codebook=[[1], [0], [0.1, -0.1]], variances=[1, 1, 1])
        pose, targets = np.zeros(3), []
        for _ in range(8):
            pose = integrate_actions(pose, [1, 0, 0.1])
            targets.append(pose)
        result = searcher.encode(targets)
        self.assertEqual(searcher.sizes, [1, 1, 2])
        np.testing.assert_array_equal(result.token_ids, [[0, 1, 2]]*8)
        np.testing.assert_allclose(result.step_errors, 0, atol=1e-14)
        for kwargs in ({"beam_size": 2}, {"per_axis_top_k": 0}, {"candidate_top_k": 4}):
            with self.assertRaises(ValueError):
                build_searcher(method="brohan", codebook=[[1], [0], [0]], variances=[1, 1, 1], **kwargs)

    def test_invalid_search_outputs(self):
        for tokens in [np.ones((8, 1)), np.full((8, 1), 3, dtype=int), np.zeros((7, 1), dtype=int)]:
            with self.assertRaises(ValueError):
                validate_result(SearchResult(tokens, np.zeros((8, 3)), np.zeros(8)), "kmeans", [3], "bad")
        result = SearchResult(np.zeros((8, 3), dtype=int), None, np.zeros(8), "per_axis")
        plan = validate_result(result, "brohan", [2, 3, 1], "good")
        self.assertEqual(plan.token_ids, [[0, 2, 5]]*8)

    def test_joint_variance_shortlist_and_brohan_variance_independence(self):
        poses = np.column_stack([np.arange(1, 9), np.zeros((8, 2))])
        book = [[1, 0.2, 0], [0.8, 0, 0]]
        first = build_searcher(method="kmeans", codebook=book, variances=[0.01, 1, 1], candidate_top_k=1).encode(poses)
        second = build_searcher(method="kmeans", codebook=book, variances=[1, 0.01, 1], candidate_top_k=1).encode(poses)
        self.assertEqual(first.token_ids[0, 0], 0)
        self.assertEqual(second.token_ids[0, 0], 1)
        axes = [[0.8, 1], [0, 0.2], [0, 0.1]]
        a = build_searcher(method="brohan", codebook=axes, variances=[0.01, 1, 1]).encode(poses)
        b = build_searcher(method="brohan", codebook=axes, variances=[1, 0.01, 1]).encode(poses)
        np.testing.assert_array_equal(a.token_ids, b.token_ids)

    def test_independent_error_filter(self):
        samples = []
        for i in range(200):
            s = self.sample(f"{i:03}")
            s.action_plan = validate_result(SearchResult(np.zeros((8, 1), dtype=int), None, np.ones(8)*i), "kmeans", [3], s.sample_id)
            samples.append(s)
        self.assertEqual(len(FilterActionError().enrich_all(samples)), 198)
        self.assertEqual(len(FilterActionError().enrich_all(samples[:100])), 99)
        self.assertEqual(len(FilterActionError().enrich_all(samples[:99])), 99)
        self.assertEqual(len(FilterActionError(0).enrich_all(samples)), 200)
        for s in samples:
            s.action_plan.step_errors = [1]*8
        kept = FilterActionError().enrich_all(samples)
        self.assertEqual(kept[0].sample_id, "002")
        for fraction in [float("nan"), -0.01, 1, True]:
            with self.assertRaises(ValueError):
                FilterActionError(fraction)

    def test_input_current_frame_and_point_checks(self):
        valid = self.sample()
        self.assertEqual(len(ValidateTrajcondInput().enrich_all([valid])), 1)
        for failure in ("image", "intent", "missing_point"):
            sample = self.sample()
            if failure == "image":
                sample.frames[3].images = {}
            elif failure == "intent":
                sample.metadata.command = "unknown"
            else:
                sample.trajectory.pop()
            self.assertEqual(ValidateTrajcondInput().enrich_all([sample]), [])
        raw = raw_sample("extra", self.image)
        extra = deepcopy(raw["frames"][0])
        extra["timestamp"] -= 500000
        raw["frames"].insert(0, extra)
        raw["metadata"]["current_frame_index"] = 4
        sample = compute_sample_trajectory(build_sample_ir(raw))
        self.assertEqual(len(ValidateTrajcondInput().enrich_all([sample])[0].trajectory), 12)

    def test_frame_sampling_ignores_capture_timestamps(self):
        for mode in ("jitter", "milliseconds", "zero"):
            with self.subTest(mode=mode):
                raw = raw_sample(mode, self.image)
                for i, frame in enumerate(raw["frames"]):
                    if mode == "jitter":
                        frame["timestamp"] += (i % 3) * 20000
                    elif mode == "milliseconds":
                        frame["timestamp"] //= 1000
                    else:
                        frame["timestamp"] = 0
                sample = compute_sample_trajectory(build_sample_ir(raw))
                self.assertEqual(ValidateTrajcondInput().enrich_all([sample]), [sample])
                self.assertEqual([p.t for p in sample.trajectory], [i * 0.5 for i in range(-3, 9)])
                self.assertEqual([p.x for p in sample.trajectory], list(range(-3, 9)))
                self.assertEqual([p.y for p in sample.trajectory], [0] * 12)
                self.assertEqual([p.heading for p in sample.trajectory], [0] * 12)

    def test_duplicate_resources_and_missing_targets(self):
        self.labels.write_text(self.labels.read_text(encoding="utf-8") + "\n" + json.dumps({"id": "a", "predict": "{}"}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate ID a"):
            MergePseudoLabelsEnricher(str(self.labels), strict_ids=True).enrich_all([self.sample()])
        self.pkl.write_bytes(pickle.dumps([{"id": "a", "gt": []}, {"id": "a", "gt": []}]))
        with self.assertRaisesRegex(ValueError, "duplicate ID a"):
            MergeGtTrajectories(str(self.pkl)).enrich_all([self.sample()])
        self.pkl.write_bytes(pickle.dumps([{"id": "a", "gt": [[1, 2, 3]]}]))
        self.assertEqual(MergeGtTrajectories(str(self.pkl)).enrich_all([self.sample(), self.sample("b")]), [])

    def test_template_mismatch_and_chain_preflight(self):
        with self.assertRaisesRegex(ValueError, "TXT does not match"):
            TrajcondStage1Builder(template_path=TEMPLATES / "navsim_trajcond_stage2_release.txt")
        contract = deepcopy(RELEASE)
        contract["profile"] = "unsupported"
        invalid_contract = self.write_json("invalid_contract.json", contract)
        with self.assertRaisesRegex(ValueError, "Only the release format"):
            TrajcondStage1Builder(format_contract_path=invalid_contract)
        cfg = self.config("kmeans", 2)
        cfg["enrichers"] = [e for e in cfg["enrichers"] if e["name"] != "merge_gt_trajectories"]
        with self.assertRaises(ValueError):
            preflight(cfg, TrajcondStage2Builder(), instantiate_enrichers(cfg["enrichers"]))
        cfg = self.config("kmeans", 1)
        for ec in cfg["enrichers"]:
            if ec["name"] == "filter_incomplete_pseudo":
                ec["params"]["require_meta_behaviour"] = False
        with self.assertRaisesRegex(ValueError, "complete CoT"):
            preflight(cfg, TrajcondStage1Builder(), instantiate_enrichers(cfg["enrichers"]))

    def test_navsim_adapter_loader_versions_and_current_frame(self):
        modules = {name: ModuleType(name) for name in
                   ["navsim", "navsim.common", "navsim.common.dataloader", "navsim.common.dataclasses"]}
        frames = []
        for i in range(12):
            frames.append(dict(token=f"frame-{i}", timestamp=5000000+i*500000,
                               ego2global_translation=[i-3, 0, 0], ego2global_rotation=[1,0,0,0],
                               ego_dynamic_state=[2,0,0,0], driving_command=[0,1,0,0],
                               cams={"CAM_F0":{"data_path":"front.jpg"}}, scene_token="scene"))
        class LoaderV20:
            def __init__(self, data_path, sensor_blobs_path, synthetic_scenes_path, scene_filter, sensor_config):
                self.scene_frames_dicts = {"current-token": frames}
                self.sensor_path = sensor_blobs_path
        class LoaderLaterV2:
            def __init__(self, data_path, original_sensor_path, synthetic_sensor_path, synthetic_scenes_path, scene_filter, sensor_config):
                self.scene_frames_dicts = {"current-token": frames}
                self.sensor_path = original_sensor_path
        modules["navsim.common.dataloader"].SceneLoader = LoaderV20
        modules["navsim.common.dataclasses"].SceneFilter = lambda **kw: SimpleNamespace(**kw, include_synthetic_scenes=False)
        modules["navsim.common.dataclasses"].SensorConfig = SimpleNamespace(build_no_sensors=lambda: None)
        config = dict(split="train", splits={"train":dict(data_path=str(self.path), sensor_blobs_path=str(self.path))})
        with patch.dict(sys.modules, modules):
            spec = importlib.util.spec_from_file_location("datasets.navsim.adapter", ROOT / "data_engine/src/datasets/navsim/adapter.py")
            adapter_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(adapter_module)
            for loader in (LoaderV20, LoaderLaterV2):
                with patch.object(adapter_module, "SceneLoader", loader):
                    adapter = adapter_module.NavsimAdapter(config)
                    raw = adapter.extract_sample(0)
                    self.assertEqual(raw["token"], "current-token")
                    self.assertEqual(raw["metadata"]["current_frame_index"], 3)
                    self.assertEqual(raw["frames"][3]["timestamp"], 6500000)
                    self.assertEqual(adapter.get_sample_count(), 1)
                    ir = compute_sample_trajectory(build_sample_ir(raw))
                    self.assertEqual(ir.images, [str(self.image)])
                    self.assertEqual(len(ValidateTrajcondInput().enrich_all([ir])), 1)

    def test_atomic_iterator_writer(self):
        path = self.path / "output.json"
        records = [{"text": "你好\nhello"}, {"value": [1, 2]}]
        write_output(iter(records), {"path": str(path)})
        self.assertEqual(path.read_text(encoding="utf-8"), json.dumps(records, ensure_ascii=False, indent=2))
        def broken():
            yield records[0]
            raise RuntimeError("interrupted")
        old = path.read_bytes()
        with self.assertRaises(RuntimeError):
            write_output(broken(), {"path": str(path)})
        self.assertEqual(path.read_bytes(), old)
        self.assertEqual(list(self.path.glob("*.tmp")), [])

    def test_first_record_stats_json_jsonl_and_shards(self):
        records = [{"id": str(i), "conversations": []} for i in range(3)]
        original = deepcopy(records)
        stats = {"mean": [[1.2345678901234567, 0, 0]]*8, "std": [[1, 1, 1]]*8}
        fields = {"trajectory_stats": stats}
        for fmt in ("json", "jsonl"):
            for chunk_size in (None, 2):
                with self.subTest(fmt=fmt, chunk_size=chunk_size):
                    path = self.path / f"metadata-{chunk_size}.{fmt}"
                    count = write_output(iter(records), {"path": str(path), "format": fmt,
                                                        "chunk_size": chunk_size}, fields)
                    self.assertEqual(count, len(records))
                    paths = [path] if chunk_size is None else [path.with_name(f"{path.stem}_{i}.{fmt}") for i in (1, 2)]
                    loaded_ids = []
                    for output in paths:
                        text = output.read_text(encoding="utf-8")
                        loaded = json.loads(text) if fmt == "json" else [json.loads(line) for line in text.splitlines()]
                        self.assertEqual(next(iter(loaded[0])), "trajectory_stats")
                        self.assertEqual(loaded[0]["trajectory_stats"], stats)
                        self.assertTrue(all("trajectory_stats" not in r for r in loaded[1:]))
                        loaded_ids.extend(r["id"] for r in loaded)
                    self.assertEqual(loaded_ids, [r["id"] for r in records])
        self.assertEqual(records, original)

    def test_cache_filter_content_and_paths(self):
        scene_filter = self.path / "filter.yaml"
        scene_filter.write_text("log_names: [a]\n", encoding="utf-8")
        cfg = dict(name="navsim", split="train", splits={"train": {"data_path": "one", "log_names_file": str(scene_filter)}})
        first = base_cache_key(cfg, True)
        scene_filter.write_text("log_names: [b]\n", encoding="utf-8")
        self.assertNotEqual(first, base_cache_key(cfg, True))
        self.assertNotEqual(base_cache_key(cfg, True), base_cache_key(cfg, False))
        other = deepcopy(cfg)
        other["splits"]["train"]["data_path"] = "two"
        self.assertNotEqual(base_cache_key(cfg, True), base_cache_key(other, True))

    def test_base_cache_excludes_enrichment_and_is_reusable_between_stages(self):
        first = self.config("kmeans", 1)
        first["cache"] = {"enabled": True}
        first["paths"] = {"cache_dir": str(self.path / "cache")}
        run_pipeline(first, TrajcondStage1Builder())
        cache_files = list((self.path / "cache").glob("*.pkl"))
        self.assertEqual(len(cache_files), 1)
        cached = pickle.loads(cache_files[0].read_bytes())
        for sample in cached:
            self.assertIsNone(sample.action_plan)
            self.assertIsNone(sample.pseudo_label)
            self.assertIsNone(sample.target_trajectory)
            self.assertEqual(sample.images, [str(self.image)])
        second = self.config("kmeans", 2)
        second["cache"], second["paths"] = first["cache"], first["paths"]
        with patch("pipelines.runner.get_adapter_class", side_effect=AssertionError("Base cache must be reused")):
            self.assertEqual(run_pipeline(second, TrajcondStage2Builder()), 2)

    def test_spawn_search_matches_serial(self):
        params = dict(method="brohan", codebook_path=self.brohan, variances_path=self.variance, search={"beam_size": 1})
        serial_output, parallel_output = io.StringIO(), io.StringIO()
        with redirect_stdout(serial_output):
            serial = EncodeTrajcond(**params).enrich_all([self.sample("a"), self.sample("b")])
        with mp.get_context("spawn").Pool(2, initializer=init_search_worker, initargs=(params,)) as pool:
            with redirect_stdout(parallel_output):
                parallel = EncodeTrajcond(**params).enrich_all([self.sample("a"), self.sample("b")], pool)
        self.assertEqual([s.action_plan.model_dump() for s in serial], [s.action_plan.model_dump() for s in parallel])
        self.assertEqual(serial_output.getvalue(), parallel_output.getvalue())
        self.assertEqual(parallel_output.getvalue().count("--- Reconstruction error statistics"), 1)

if __name__ == "__main__":
    unittest.main()
