"""Deterministic local-motion search with vehicle corner distance; no file I/O."""

from dataclasses import dataclass
from itertools import product
import numpy as np


def wrap_angle(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def local_action(previous, target):
    delta = np.asarray(target) - previous
    c, s = np.cos(previous[2]), np.sin(previous[2])
    return np.array([c * delta[0] + s * delta[1],
                     -s * delta[0] + c * delta[1], wrap_angle(delta[2])])


def integrate_actions(previous, actions):
    """Render one or more local actions in the current t=0 ego frame."""
    actions = np.asarray(actions, dtype=np.float64)
    c, s = np.cos(previous[2]), np.sin(previous[2])
    result = np.empty_like(actions)
    result[..., 0] = previous[0] + c * actions[..., 0] - s * actions[..., 1]
    result[..., 1] = previous[1] + s * actions[..., 0] + c * actions[..., 1]
    result[..., 2] = wrap_angle(previous[2] + actions[..., 2])
    return result


def box_corners(poses, *, width=2.297, front_bumper_to_axle=4.049,
                rear_bumper_to_axle=-1.127):
    poses = np.asarray(poses, dtype=np.float64)
    corners = np.array([[front_bumper_to_axle, width / 2],
                        [front_bumper_to_axle, -width / 2],
                        [rear_bumper_to_axle, -width / 2],
                        [rear_bumper_to_axle, width / 2]])
    c, s = np.cos(poses[..., 2])[..., None], np.sin(poses[..., 2])[..., None]
    x = poses[..., 0, None] + c * corners[:, 0] - s * corners[:, 1]
    y = poses[..., 1, None] + s * corners[:, 0] + c * corners[:, 1]
    return np.stack([x, y], axis=-1)


@dataclass
class SearchResult:
    token_ids: np.ndarray
    reconstructed_trajectory: np.ndarray
    step_errors: np.ndarray
    index_space: str = "global"


class ActionSearcher:
    def __init__(self, *, method, codebook, variances, beam_size=None,
                 candidate_top_k=None, per_axis_top_k=None, vehicle=None):
        if method not in {"kmeans", "kdisks", "brohan"}:
            raise ValueError(f"Unknown action method: {method}")
        if beam_size is None:
            beam_size = 1 if method == "brohan" else 4
        if type(beam_size) is not int or beam_size <= 0:
            raise ValueError("Beam size must be a positive integer")
        if method == "brohan" and beam_size != 1:
            raise ValueError("Brohan only supports beam_size=1")
        self.method = method
        self.beam_size = beam_size
        self.variances = np.asarray(variances, dtype=np.float64)
        if self.variances.shape != (3,) or not np.isfinite(self.variances).all() or (self.variances <= 0).any():
            raise ValueError("Action variances must contain three finite positive values")
        self.vehicle = dict(vehicle or {})
        geometry = {"width": 2.297, "front_bumper_to_axle": 4.049, "rear_bumper_to_axle": -1.127}
        if set(self.vehicle) - set(geometry):
            raise ValueError("Unknown vehicle geometry fields")
        geometry.update(self.vehicle)
        if not all(np.isfinite(x) for x in geometry.values()) or geometry["width"] <= 0 or geometry["front_bumper_to_axle"] <= geometry["rear_bumper_to_axle"]:
            raise ValueError("Invalid vehicle geometry")
        self.vehicle = geometry
        if method == "brohan":
            if candidate_top_k is not None or len(codebook) != 3:
                raise ValueError("Brohan requires three axis codebooks and per_axis_top_k")
            self.codebook = [np.asarray(axis, dtype=np.float64) for axis in codebook]
            if any(a.ndim != 1 or len(a) == 0 or not np.isfinite(a).all() for a in self.codebook):
                raise ValueError("Each Brohan axis must be a nonempty finite vector")
            self.sizes = [len(a) for a in self.codebook]
            self.top_k = 4 if per_axis_top_k is None else per_axis_top_k
        else:
            if per_axis_top_k is not None:
                raise ValueError("Joint search requires candidate_top_k")
            self.codebook = np.asarray(codebook, dtype=np.float64)
            if self.codebook.ndim != 2 or self.codebook.shape[1] != 3 or len(self.codebook) == 0 or not np.isfinite(self.codebook).all():
                raise ValueError("Joint codebook must be a finite nonempty (K,3) matrix")
            self.sizes = [len(self.codebook)]
            self.top_k = 80 if candidate_top_k is None else candidate_top_k
        if type(self.top_k) is not int or self.top_k <= 0:
            raise ValueError("Candidate count must be a positive integer")

    def _candidates(self, previous, target):
        ideal = local_action(previous, target)
        if self.method == "brohan":
            shortlist = []
            for dimension, axis in enumerate(self.codebook):
                residual = axis - ideal[dimension]
                if dimension == 2:
                    residual = wrap_angle(residual)
                order = np.lexsort((np.arange(len(axis)), np.abs(residual)))
                shortlist.append(order[:min(self.top_k, len(axis))])
            ids = np.array(sorted(product(*shortlist)), dtype=np.int64)
            actions = np.column_stack([self.codebook[d][ids[:, d]] for d in range(3)])
            ids += np.array([0, self.sizes[0], sum(self.sizes[:2])])
        else:
            residual = self.codebook - ideal
            residual[:, 2] = wrap_angle(residual[:, 2])
            distance = np.sum(residual ** 2 / self.variances, axis=1)
            order = np.lexsort((np.arange(len(distance)), distance))[:min(self.top_k, len(distance))]
            ids, actions = order[:, None], self.codebook[order]
        return ids, integrate_actions(previous, actions)

    def encode(self, poses):
        poses = np.asarray(poses, dtype=np.float64)
        if poses.shape != (8, 3) or not np.isfinite(poses).all():
            raise ValueError("Search requires eight finite physical future poses")
        # Each state stores the true error sum, token rows, poses, and step errors.
        beam = [(0.0, (), (), ())]
        for target in poses:
            target_corners = box_corners(target, **self.vehicle)
            expanded = []
            for total_error, tokens, reconstructed, errors in beam:
                previous = reconstructed[-1] if reconstructed else np.zeros(3)
                ids, candidates = self._candidates(previous, target)
                losses = np.linalg.norm(box_corners(candidates, **self.vehicle) - target_corners,
                                        axis=-1).mean(axis=-1)
                for token_row, candidate, loss in zip(ids, candidates, losses):
                    expanded.append((total_error + float(loss),
                                     tokens + (tuple(token_row),),
                                     reconstructed + (candidate,), errors + (float(loss),)))
            # At equal path lengths, error sums and means give the same ranking.
            # Full token sequences break ties independently of candidate ranking.
            # For one path, use the next error directly to preserve greedy ties.
            expanded.sort(key=lambda state: (state[3][-1] if self.beam_size == 1 else state[0], state[1]))
            beam = expanded[:self.beam_size]
        _, tokens, reconstructed, errors = beam[0]
        return SearchResult(np.asarray(tokens), np.asarray(reconstructed), np.asarray(errors))


def build_searcher(*, method, codebook, variances, **search_params):
    return ActionSearcher(method=method, codebook=codebook, variances=variances, **search_params)
