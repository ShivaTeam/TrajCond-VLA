"""Check GT geometry, joint clustering, total token sizes, and JSON output."""

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import math
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "build_codebook_from_gt.py"
SPEC = importlib.util.spec_from_file_location("gt_codebook_pipeline", SCRIPT)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)
HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None
HAS_SCIPY = importlib.util.find_spec("scipy") is not None


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.actions = np.array([
            [0.0, -2.0, -0.5], [0.2, -1.8, -0.4],
            [4.0, 2.0, 0.4], [4.2, 2.2, 0.5],
        ])
        self.records = [
            {"id": f"sample_{index}", "gt": [action.tolist()]}
            for index, action in enumerate(self.actions)
        ]

    def test_local_heading_yaw_wrap_and_variable_lengths(self):
        records = [
            {"id": "turn", "gt": [[1, 0, math.pi/2], [1, 2, math.pi/2]]},
            {"id": "restart", "gt": [[3, -2, -0.3]]},
        ]
        actions, counts = builder.extract_local_actions(records)
        np.testing.assert_allclose(
            actions, [[1, 0, math.pi/2], [2, 0, 0], [3, -2, -0.3]], atol=1e-12
        )
        self.assertEqual(counts, {
            "trajectory_count": 2, "action_count": 3, "trajectory_lengths": {1: 1, 2: 1}
        })
        wrapped = builder.trajectory_to_local_actions(
            [[0, 0, math.pi-0.1], [0, 0, -math.pi+0.1]]
        )
        self.assertAlmostEqual(wrapped[1, 2], 0.2)

    def test_invalid_gt_reports_record_id(self):
        for gt in ([], [[1, 2]], [[1, float("nan"), 0]], [[1, 2, float("inf")]]):
            with self.subTest(gt=gt), self.assertRaisesRegex(ValueError, "id=bad"):
                builder.extract_local_actions([{"id": "bad", "gt": gt}])
        with self.assertRaisesRegex(ValueError, "id=missing"):
            builder.extract_local_actions([{"id": "missing"}])

    def test_brohan_size_is_total_tokens(self):
        stats = {"min": [1, -2, 0], "max": [3, 2, 0]}
        book = builder.build_codebook(stats, 9)
        self.assertEqual(book, {
            "k": [3, 3, 3], "codebook": [[1, 2, 3], [-2, 0, 2], [0, 0, 0]],
            "boundaries": stats,
        })
        self.assertEqual(sum(book["k"]), 9)
        for size in (1, 2, 5, 7, 8, 10):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "divisible by 3"):
                builder.build_codebook(stats, size)

    @unittest.skipUnless(HAS_SKLEARN, "KMeans requires scikit-learn")
    def test_joint_kmeans_uses_all_samples_and_is_repeatable(self):
        weighted = np.vstack([self.actions[[0]], self.actions[[0]], self.actions])
        book = builder.build_clustered_codebook(weighted, "kmeans", 2, seed=42)
        centers = np.asarray(book["codebook"])
        order = np.argsort(centers[:, 0])
        self.assertEqual(book["k"], 2)
        self.assertEqual(centers.shape, (2, 3))
        np.testing.assert_allclose(
            centers[order], [[0.05, -1.95, -0.475], [4.1, 2.1, 0.45]], atol=1e-12
        )
        np.testing.assert_array_equal(np.asarray(book["cluster_sizes"])[order], [4, 2])
        self.assertEqual(book, builder.build_clustered_codebook(weighted, "kmeans", 2, seed=42))
        self.assertEqual(set(book), {"k", "codebook", "cluster_sizes"})

    @unittest.skipUnless(HAS_SCIPY, "KDisks assignment counts require SciPy")
    def test_joint_kdisks_centers_preserve_input_vectors_precision_and_order(self):
        rng = np.random.default_rng(12)
        actions = rng.normal(size=(80, 3))
        book = builder.build_clustered_codebook(actions, "kdisks", 7, seed=42, epsilon=0.4)
        centers = np.asarray(book["codebook"])
        self.assertEqual(book["k"], 7)
        self.assertEqual(centers.shape, (7, 3))
        for center in centers:
            self.assertTrue(np.any(np.all(actions == center, axis=1)))
        for first in range(len(centers)):
            for second in range(first):
                self.assertGreater(np.linalg.norm(centers[first] - centers[second]), 0.4)
        first_index = np.random.default_rng(42).integers(0, len(actions))
        np.testing.assert_array_equal(centers[0], actions[first_index])
        self.assertFalse(np.array_equal(centers, np.round(centers, 6)))
        self.assertEqual(sum(book["cluster_sizes"]), len(actions))
        self.assertEqual(book, builder.build_clustered_codebook(actions, "kdisks", 7, seed=42, epsilon=0.4))

    @unittest.skipUnless(HAS_SCIPY, "KDisks assignment counts require SciPy")
    def test_kdisks_zero_epsilon_removes_duplicate_vectors(self):
        actions = np.repeat(self.actions, 5, axis=0)
        book = builder.build_clustered_codebook(actions, "kdisks", 4, epsilon=0)
        self.assertEqual({tuple(row) for row in book["codebook"]}, {tuple(row) for row in self.actions})
        self.assertEqual(book["cluster_sizes"], [5, 5, 5, 5])

    @unittest.skipUnless(HAS_SCIPY, "KDisks assignment counts require SciPy")
    def test_kdisks_uses_joint_distance_with_constant_axis(self):
        actions = np.array([[0, 0, 0], [0.08, 0.08, 0]])
        book = builder.build_clustered_codebook(actions, "kdisks", 2, epsilon=0.1)
        self.assertEqual(np.asarray(book["codebook"]).shape, (2, 3))
        self.assertEqual(book["cluster_sizes"], [1, 1])

    def test_kdisks_includes_epsilon_boundary_in_exclusion(self):
        actions = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        with self.assertRaisesRegex(ValueError, r"KDisks: found [12]/3 centers"):
            builder.build_clustered_codebook(actions, "kdisks", 3, epsilon=1)

    @unittest.skipUnless(HAS_SCIPY, "Nearest-center assignment requires SciPy")
    def test_chunked_assignment_counts_preserve_center_order_and_ties(self):
        actions = np.array([[0, 0, 0], [0.5, 0, 0], [1, 0, 0], [2, 0, 0]])
        centers = np.array([[0, 0, 0], [1, 0, 0]])
        with patch.object(builder, "DISTANCE_BUFFER_ENTRIES", 2):
            counts = builder.count_cluster_assignments(actions, centers)
        np.testing.assert_array_equal(counts, [2, 2])

    @unittest.skipUnless(HAS_SKLEARN, "KMeans requires scikit-learn")
    def test_kmeans_accepts_constant_axis_in_distinct_3d_vectors(self):
        actions = self.actions.copy()
        actions[:, 2] = 0
        book = builder.build_clustered_codebook(actions, "kmeans", 2)
        self.assertEqual(book["k"], 2)
        np.testing.assert_array_equal(np.asarray(book["codebook"])[:, 2], [0, 0])
        self.assertEqual(sum(book["cluster_sizes"]), len(actions))

    @unittest.skipUnless(HAS_SKLEARN, "KMeans requires scikit-learn")
    def test_kmeans_rejects_insufficient_distinct_vectors(self):
        actions = np.repeat(self.actions[[0]], 4, axis=0)
        with self.assertRaisesRegex(ValueError, "only 1 distinct action vectors"):
            builder.build_clustered_codebook(actions, "kmeans", 2)

    def test_invalid_clustering_configuration(self):
        for epsilon in (-1, float("nan"), float("inf")):
            with self.subTest(epsilon=epsilon), self.assertRaisesRegex(ValueError, "epsilon"):
                builder.build_clustered_codebook(self.actions, "kdisks", 2, epsilon=epsilon)
        for size in (0, -2, 2.5, True):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "size"):
                builder.build_clustered_codebook(self.actions, "kdisks", size)
        with self.assertRaisesRegex(ValueError, "Seed"):
            builder.build_clustered_codebook(self.actions, "kdisks", 2, seed=-1)

    def test_single_token_codebooks_are_valid(self):
        methods = []
        if HAS_SKLEARN:
            methods.append("kmeans")
        if HAS_SCIPY:
            methods.append("kdisks")
        for method in methods:
            with self.subTest(method=method):
                book = builder.build_clustered_codebook(self.actions, method, 1)
                self.assertEqual(book["k"], 1)
                self.assertEqual(np.asarray(book["codebook"]).shape, (1, 3))
                self.assertEqual(book["cluster_sizes"], [len(self.actions)])

    def test_cli_each_method_writes_codebook_and_action_variances(self):
        methods = ["brohan"]
        if HAS_SKLEARN:
            methods.append("kmeans")
        if HAS_SCIPY:
            methods.append("kdisks")
        for method in methods:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "gt.pkl"
                custom_records = [{"id": r["id"], "poses": r["gt"]} for r in self.records]
                with source.open("wb") as stream:
                    pickle.dump(custom_records, stream)
                terminal = io.StringIO()
                size = 6 if method == "brohan" else 4
                with redirect_stdout(terminal):
                    status = builder.main([
                        "--pkl", str(source), "--gt-key", "poses", "--method", method, "--size", str(size)
                    ])
                self.assertEqual(status, 0)
                filename = (
                    f"brohan_codebook_1d_size_{size}.json" if method == "brohan"
                    else f"{method}_codebook_{size}.json"
                )
                output = root / filename
                book = json.loads(output.read_text(encoding="utf-8"))
                if method == "brohan":
                    self.assertEqual(sum(book["k"]), size)
                    self.assertEqual(np.asarray(book["codebook"]).shape, (3, size // 3))
                else:
                    self.assertEqual(book["k"], size)
                    self.assertEqual(np.asarray(book["codebook"]).shape, (size, 3))
                    self.assertEqual(len(book["cluster_sizes"]), size)
                    self.assertEqual(sum(book["cluster_sizes"]), len(self.records))
                variance_output = root / "action_variances.json"
                variances = json.loads(variance_output.read_text(encoding="utf-8"))
                self.assertEqual(set(variances), {"variances"})
                np.testing.assert_allclose(variances["variances"], np.var(self.actions, axis=0))
                self.assertEqual(
                    {p.name for p in root.iterdir()}, {source.name, output.name, variance_output.name}
                )
                self.assertIn("Trajectories: 4; actions: 4", terminal.getvalue())
                self.assertIn(f"vocabulary: {size}", terminal.getvalue())
                self.assertIn(f"Action variances: {variance_output.resolve()}", terminal.getvalue())

    @unittest.skipUnless(HAS_SKLEARN, "KMeans requires scikit-learn")
    def test_cli_default_size_creates_exactly_768_joint_tokens(self):
        rng = np.random.default_rng(13)
        records = [
            {"id": f"sample_{index}", "gt": [pose.tolist()]}
            for index, pose in enumerate(rng.uniform(-0.5, 0.5, size=(768, 3)))
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "gt.pkl"
            with source.open("wb") as stream:
                pickle.dump(records, stream)
            with redirect_stdout(io.StringIO()):
                status = builder.main(["--input", str(source), "--method", "kmeans"])
            self.assertEqual(status, 0)
            output = root / "kmeans_codebook_768.json"
            book = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(book["k"], 768)
            self.assertEqual(np.asarray(book["codebook"]).shape, (768, 3))
            self.assertEqual(book["cluster_sizes"], [1] * 768)
            self.assertEqual(
                {p.name for p in root.iterdir()}, {source.name, output.name, "action_variances.json"}
            )

    def test_cli_variances_use_all_actions_apply_std_floor_and_follow_custom_output(self):
        records = [
            {"id": "two_steps", "gt": [[0, 0, 0], [2, 0, 5e-7]]},
            {"id": "one_step", "gt": [[4, 0, 0]]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "gt.pkl"
            source.write_bytes(pickle.dumps(records))
            output = root / "custom" / "codebook.json"
            with redirect_stdout(io.StringIO()):
                status = builder.main(["--input", str(source), "--size", "6", "--output", str(output)])
            self.assertEqual(status, 0)
            variance_output = output.parent / "action_variances.json"
            variances = json.loads(variance_output.read_text(encoding="utf-8"))
            np.testing.assert_allclose(variances["variances"], [8 / 3, 1e-12, 1e-12], rtol=1e-14)
            self.assertFalse((root / "action_variances.json").exists())
            self.assertEqual({p.name for p in output.parent.iterdir()}, {output.name, variance_output.name})

    def test_cli_rejects_variance_output_collision_with_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "action_variances.json"
            before = pickle.dumps(self.records)
            source.write_bytes(before)
            output = root / "codebook.json"
            stderr = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
                status = builder.main(["--input", str(source), "--size", "6", "--output", str(output)])
            self.assertEqual(status, 2)
            self.assertIn("Variance output must not overwrite the input PKL", stderr.getvalue())
            self.assertEqual(source.read_bytes(), before)
            self.assertFalse(output.exists())

    def test_cli_failure_preserves_existing_output_and_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "gt.pkl", root / "existing.json"
            with source.open("wb") as stream:
                pickle.dump(self.records, stream)
            before = source.read_bytes()
            output.write_text('{"existing": true}\n', encoding="utf-8")
            variance_output = root / "action_variances.json"
            variance_before = '{"variances": [1, 2, 3]}\n'
            variance_output.write_text(variance_before, encoding="utf-8")
            stderr = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
                status = builder.main([
                    "--input", str(source), "--method", "kdisks", "--size", "2",
                    "--epsilon", "100", "--output", str(output),
                ])
            self.assertEqual(status, 2)
            self.assertIn("all candidates were exhausted", stderr.getvalue())
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"existing": True})
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(variance_output.read_text(encoding="utf-8"), variance_before)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(builder.main(["--input", str(source), "--output", str(source)]), 2)
                self.assertEqual(builder.main(["--input", str(source), "--method", "brohan", "--size", "8"]), 2)
                self.assertEqual(builder.main(["--input", str(source), "--output", str(variance_output)]), 2)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(variance_output.read_text(encoding="utf-8"), variance_before)


if __name__ == "__main__":
    unittest.main()
