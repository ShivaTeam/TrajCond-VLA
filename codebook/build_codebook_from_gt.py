"""Extract local actions from GT trajectories and build an action codebook.

Examples (run from the repository root):
    python codebook/build_codebook_from_gt.py --method brohan --size 768
    python codebook/build_codebook_from_gt.py --method kmeans --size 768
    python codebook/build_codebook_from_gt.py --method kdisks --size 768

Input: a pickle containing [{"id": str, "gt": [[x, y, yaw], ...]}, ...].
GT poses are in the ego frame at t=0, with x/y in meters and yaw in radians.
Only future poses belong in the input; each trajectory starts at (0, 0, 0).
Actions are computed in the previous true pose's frame, with yaw differences
wrapped to [-pi, pi). All trajectories contribute to all three methods.

Size always denotes the total number of action tokens. KMeans and KDisks
jointly process [dx, dy, dyaw] vectors, producing (size, 3) codebooks. Each
center is one token; center order and full precision are preserved.
Brohan uses inclusive per-axis min/max grids with size/3 values per axis.
Its size must be divisible by three, with at least two values per axis.
The codebook and action_variances.json are saved in the same directory.
Variances use all extracted actions, with population standard deviations
clamped to at least 1e-6 before squaring. Statistics are printed to the terminal.
NumPy is required. KMeans also needs scikit-learn and threadpoolctl;
KDisks needs SciPy for nearest-center assignment counts.
Exit codes: 0 = success; 2 = invalid input, configuration, or missing dependency.
"""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import pickle
import sys
import tempfile

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "trajectory.pkl"
DIMENSIONS = ("dx", "dy", "dyaw")
METHODS = ("brohan", "kmeans", "kdisks")
DISTANCE_BUFFER_ENTRIES = 1_048_576


def load_gt_pickle(path):
    with path.open("rb") as stream:
        records = pickle.load(stream)
    if not isinstance(records, list) or not records:
        raise ValueError("GT pickle must contain a nonempty list of records.")
    return records


def trajectory_to_local_actions(trajectory):
    """Return (T, 3) local [dx, dy, dyaw], starting from (0, 0, 0)."""
    poses = np.asarray(trajectory, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 3 or poses.shape[0] == 0:
        raise ValueError("GT trajectory must have shape (T, 3), with T > 0.")
    if not np.isfinite(poses).all():
        raise ValueError("GT trajectory contains a non-finite value.")
    previous = np.zeros(3, dtype=np.float64)
    actions = np.empty_like(poses)
    for step, current in enumerate(poses):
        world_dx = current[0] - previous[0]
        world_dy = current[1] - previous[1]
        cos_yaw, sin_yaw = math.cos(previous[2]), math.sin(previous[2])
        actions[step] = (
            world_dx * cos_yaw + world_dy * sin_yaw,
            -world_dx * sin_yaw + world_dy * cos_yaw,
            (current[2] - previous[2] + math.pi) % (2 * math.pi) - math.pi,
        )
        previous = current
    if not np.isfinite(actions).all():
        raise ValueError("Local actions contain a non-finite value.")
    return actions


def extract_local_actions(records, gt_key="gt"):
    """Extract all trajectories once, preserving record order and error context."""
    batches = []
    trajectory_lengths = Counter()
    for index, record in enumerate(records, start=1):
        sample_id = record.get("id", "<missing>") if isinstance(record, dict) else "<invalid>"
        try:
            if not isinstance(record, dict) or gt_key not in record:
                raise ValueError(f"Record must be a dict containing {gt_key!r}.")
            actions = trajectory_to_local_actions(record[gt_key])
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f"Record {index}, id={sample_id}: {exc}") from exc
        batches.append(actions)
        trajectory_lengths[len(actions)] += 1
    if not batches:
        raise ValueError("No GT trajectories were provided.")
    all_actions = np.concatenate(batches, axis=0)
    counts = {
        "trajectory_count": len(batches),
        "action_count": len(all_actions),
        "trajectory_lengths": dict(sorted(trajectory_lengths.items())),
    }
    return all_actions, counts


def validate_actions(actions):
    array = np.asarray(actions, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or len(array) == 0:
        raise ValueError("Local actions must have nonempty shape (M, 3).")
    if not np.isfinite(array).all():
        raise ValueError("Local actions contain a non-finite value.")
    return array


def action_statistics(actions):
    actions = validate_actions(actions)
    stats = {
        "min": np.min(actions, axis=0).tolist(),
        "max": np.max(actions, axis=0).tolist(),
        "std": [float(np.std(actions[:, dim], ddof=0)) for dim in range(3)],
        "mean": [float(np.mean(actions[:, dim])) for dim in range(3)],
    }
    if not all(math.isfinite(value) for values in stats.values() for value in values):
        raise ValueError("Action statistics contain a non-finite value.")
    return stats


def compute_action_statistics(records, gt_key="gt"):
    """Retain the statistics helper for callers that do not need the action array."""
    actions, counts = extract_local_actions(records, gt_key)
    return action_statistics(actions), counts


def validate_codebook_size(codebook_size):
    if not isinstance(codebook_size, int) or isinstance(codebook_size, bool) or codebook_size < 1:
        raise ValueError("Codebook size must be a positive integer representing total tokens.")


def brohan_axis_size(codebook_size):
    validate_codebook_size(codebook_size)
    if codebook_size < 6 or codebook_size % 3:
        raise ValueError("Brohan total token size must be divisible by 3 and >= 6.")
    return codebook_size // 3


def validate_triplet(values, label):
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must contain three finite numbers.")
    return array.tolist()


def build_codebook(stats, codebook_size=768):
    """Build a Brohan grid with codebook_size total tokens across three axes."""
    axis_size = brohan_axis_size(codebook_size)
    minimum = validate_triplet(stats.get("min"), "stats.min")
    maximum = validate_triplet(stats.get("max"), "stats.max")
    if any(lower > upper for lower, upper in zip(minimum, maximum)):
        raise ValueError("Each action minimum must be <= its maximum.")
    codebook = [
        np.round(np.linspace(lower, upper, axis_size), 6).tolist()
        for lower, upper in zip(minimum, maximum)
    ]
    return {
        "k": [axis_size] * 3,
        "codebook": codebook,
        "boundaries": {"min": minimum, "max": maximum},
    }


def kmeans_centers(actions, codebook_size, seed):
    """Fit joint 3D KMeans and return centers and assignment counts."""
    try:
        from sklearn.cluster import KMeans
        from threadpoolctl import threadpool_limits
    except ImportError as exc:
        raise ValueError(
            "KMeans requires scikit-learn and threadpoolctl. "
            "Install dependencies with: python -m pip install -r codebook/requirements.txt"
        ) from exc

    unique_count = len(np.unique(actions, axis=0))
    if unique_count < codebook_size:
        raise ValueError(
            f"KMeans: only {unique_count} distinct action vectors for "
            f"{codebook_size} centers. Use a smaller --codebook-size."
        )
    with threadpool_limits(limits=1):
        model = KMeans(n_clusters=codebook_size, random_state=seed, n_init="auto").fit(actions)
    centers = model.cluster_centers_
    if len(np.unique(centers, axis=0)) != codebook_size:
        raise ValueError(f"KMeans: fewer than {codebook_size} distinct centers were fitted.")
    return centers, np.bincount(model.labels_, minlength=codebook_size)


def kdisks_centers(actions, codebook_size, seed, epsilon):
    """Select joint 3D centers using the Euclidean rule in kdisks_single.py."""
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("KDisks epsilon must be finite and nonnegative.")
    rng = np.random.default_rng(seed)
    remaining = actions.copy()
    centers = []
    while len(centers) < codebook_size and len(remaining):
        # Copy the row so it does not retain the entire candidate array.
        center = remaining[rng.integers(0, len(remaining))].copy()
        centers.append(center)
        if len(centers) == codebook_size:
            break
        distances = np.linalg.norm(remaining - center, axis=1)
        remaining = remaining[distances > epsilon]
    if len(centers) < codebook_size:
        raise ValueError(
            f"KDisks: found {len(centers)}/{codebook_size} centers "
            f"with epsilon={epsilon:g}; all candidates were exhausted. "
            "Use a smaller --epsilon or --codebook-size."
        )
    return np.asarray(centers)


def count_cluster_assignments(actions, centers):
    """Count nearest centers with bounded memory and reference argmin tie handling."""
    try:
        from scipy.spatial.distance import cdist
    except ImportError as exc:
        raise ValueError(
            "KDisks assignment counts require SciPy. "
            "Install dependencies with: python -m pip install -r codebook/requirements.txt"
        ) from exc
    actions, centers = validate_actions(actions), validate_actions(centers)
    counts = np.zeros(len(centers), dtype=np.int64)
    chunk_rows = max(1, DISTANCE_BUFFER_ENTRIES // len(centers))
    for start in range(0, len(actions), chunk_rows):
        distances = cdist(actions[start:start + chunk_rows], centers)
        labels = np.argmin(distances, axis=1)
        counts += np.bincount(labels, minlength=len(centers))
    return counts


def build_clustered_codebook(actions, method, codebook_size=768, seed=42, epsilon=0.03):
    """Return a single-token 3D codebook in the original center order and precision."""
    validate_codebook_size(codebook_size)
    actions = validate_actions(actions)
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer between 0 and 2**32 - 1.")
    if method == "kmeans":
        centers, cluster_sizes = kmeans_centers(actions, codebook_size, seed)
    elif method == "kdisks":
        centers = kdisks_centers(actions, codebook_size, seed, epsilon)
        cluster_sizes = count_cluster_assignments(actions, centers)
    else:
        raise ValueError(f"Unknown clustering method: {method!r}.")
    if centers.shape != (codebook_size, 3) or not np.isfinite(centers).all():
        raise ValueError("Clustering produced an invalid codebook.")
    if cluster_sizes.shape != (codebook_size,) or np.any(cluster_sizes < 0) or cluster_sizes.sum() != len(actions):
        raise ValueError("Cluster assignment counts do not cover all actions.")
    return {
        "k": codebook_size,
        "codebook": centers.tolist(),
        "cluster_sizes": cluster_sizes.tolist(),
    }


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def print_statistics(stats, counts):
    print(f"Trajectories: {counts['trajectory_count']:,}; actions: {counts['action_count']:,}")
    print(f"Trajectory lengths: {counts['trajectory_lengths']}")
    print(f"{'dimension':<10} {'min':>22} {'max':>22} {'mean':>22} {'std':>22}")
    for dimension, lower, upper, mean, std in zip(
        DIMENSIONS, stats["min"], stats["max"], stats["mean"], stats["std"]
    ):
        print(f"{dimension:<10} {lower:>22.16g} {upper:>22.16g} {mean:>22.16g} {std:>22.16g}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", "--pkl", type=Path, default=DEFAULT_INPUT, help="GT trajectory pickle (default: codebook/trajectory.pkl).")
    parser.add_argument("--gt-key", default="gt", help="Trajectory key in each record (default: gt).")
    parser.add_argument("--method", choices=METHODS, default="brohan", help="Codebook algorithm (default: brohan).")
    parser.add_argument("--codebook-size", "--size", type=int, default=768, help="Total action token count (default: 768). Brohan requires a multiple of 3 and >= 6.")
    parser.add_argument("--seed", type=int, default=42, help="KMeans/KDisks random seed (default: 42).")
    parser.add_argument("--epsilon", type=float, default=0.03, help="KDisks exclusion radius in raw [dx, dy, dyaw] Euclidean space (default: 0.03).")
    parser.add_argument("--output", type=Path, help="Default beside input: <method>_codebook_<size>.json; Brohan: brohan_codebook_1d_size_<size>.json. Also writes action_variances.json in the same directory.")
    args = parser.parse_args(argv)
    filename = (
        f"brohan_codebook_1d_size_{args.codebook_size}.json" if args.method == "brohan"
        else f"{args.method}_codebook_{args.codebook_size}.json"
    )
    output = args.output or args.input.parent / filename
    variance_output = output.parent / "action_variances.json"
    try:
        validate_codebook_size(args.codebook_size)
        if args.method == "brohan":
            brohan_axis_size(args.codebook_size)
        if not args.gt_key:
            raise ValueError("GT key must be nonempty.")
        if output.resolve() == args.input.resolve():
            raise ValueError("Output must not overwrite the input PKL.")
        if variance_output.resolve() == args.input.resolve():
            raise ValueError("Variance output must not overwrite the input PKL.")
        if output.resolve() == variance_output.resolve():
            raise ValueError("Codebook output must differ from action_variances.json.")
        if not 0 <= args.seed < 2**32:
            raise ValueError("Seed must be between 0 and 2**32 - 1.")
        if args.method == "kdisks" and (not math.isfinite(args.epsilon) or args.epsilon < 0):
            raise ValueError("KDisks epsilon must be finite and nonnegative.")
        print(f"Extracting local actions from: {args.input.resolve()}", flush=True)
        records = load_gt_pickle(args.input)
        actions, counts = extract_local_actions(records, args.gt_key)
        del records
        stats = action_statistics(actions)
        # Clamp population standard deviations before squaring to avoid zero variances.
        variances = {"variances": np.square(np.maximum(stats["std"], 1e-6)).tolist()}
        print_statistics(stats, counts)
        print(f"Building {args.method} codebook: vocabulary={args.codebook_size}.", flush=True)
        if args.method == "brohan":
            codebook = build_codebook(stats, args.codebook_size)
        else:
            codebook = build_clustered_codebook(
                actions, args.method, args.codebook_size, args.seed, args.epsilon
            )
        write_json(output, codebook)
        write_json(variance_output, variances)
    except (OSError, ValueError, TypeError, OverflowError, EOFError, pickle.PickleError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Codebook shape: {np.asarray(codebook['codebook']).shape}; vocabulary: {args.codebook_size}")
    print(f"Codebook: {output.resolve()}")
    print(f"Action variances: {variance_output.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
