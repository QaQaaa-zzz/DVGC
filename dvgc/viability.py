"""Phase-explicit Physical-Belief Viability Ensemble.

Formal Tube labels always come from frozen-policy branch certification.  This
ensemble only interpolates inside candidate support and exposes epistemic
uncertainty for budgeted re-labelling.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass
class _Member:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray


def physical_belief_feature(record: dict[str, Any]) -> np.ndarray:
    ps = dict(record.get("policy_state", {}))
    physical = np.asarray(record["physical_feature"], np.float32).reshape(-1)
    phase_probs = np.asarray(ps.get("phase_probs", np.eye(4, dtype=np.float32)[int(record.get("oracle_phase", 0))]), np.float32).reshape(-1)
    contact_probs = np.asarray(ps.get("contact_probs", np.zeros(2, np.float32)), np.float32).reshape(-1)
    progress = np.asarray([float(ps.get("phase_progress", 0.0))], np.float32)
    confidence = np.asarray([float(ps.get("phase_confidence", float(phase_probs.max(initial=0.0))))], np.float32)
    last_action = np.asarray(ps.get("last_action", np.zeros(4, np.float32)), np.float32).reshape(-1)
    history = np.asarray(ps.get("obs_history", np.zeros((0, 1), np.float32)), np.float32)
    # Bounded history summary keeps the model compact while preserving that the
    # same physical state with different recent observations is a different z.
    if history.size:
        hmean = history.mean(axis=0)
        hlast = history[-1]
        history_summary = np.concatenate([hmean[:12], hlast[:12]]).astype(np.float32)
    else:
        history_summary = np.zeros(24, np.float32)
    return np.concatenate([physical, phase_probs, contact_probs, progress, confidence, last_action, history_summary]).astype(np.float32)


class ViabilityEnsemble:
    VERSION = 2

    def __init__(self, members: int = 5, hidden: int = 64, seed: int = 0):
        self.members_n = int(members)
        self.hidden = int(hidden)
        self.seed = int(seed)
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.train_x: np.ndarray | None = None
        self.members: list[_Member] = []

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))

    @staticmethod
    def _forward(member: _Member, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = np.tanh(x @ member.w1 + member.b1)
        return h, (h @ member.w2 + member.b2)[:, 0]

    def _transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Viability ensemble has not been fitted")
        return ((np.asarray(x, np.float32) - self.mean) / self.std).astype(np.float32)

    def fit(self, records: Sequence[dict[str, Any]], *, epochs: int = 400, batch_size: int = 96, learning_rate: float = 2e-3, sample_weights: np.ndarray | None = None) -> dict[str, float]:
        rows = [r for r in records if int(r.get("final", {}).get("branches", 0)) > 0 and not bool(r.get("training_only", False))]
        if len(rows) < 8:
            raise ValueError("Need at least 8 Final-certified snapshots")
        x = np.stack([physical_belief_feature(r) for r in rows]).astype(np.float32)
        y = np.asarray([float(r["final"]["posterior"]["mean"]) for r in rows], np.float32)
        weights = np.ones(len(rows), np.float64) if sample_weights is None else np.asarray(sample_weights, np.float64)
        if weights.shape != (len(rows),) or not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("sample_weights must be positive finite state-level weights")
        weights /= weights.sum()
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0) + 1e-5
        xx = self._transform(x)
        self.train_x = xx.copy()
        rng = np.random.default_rng(self.seed)
        self.members = []
        losses = []
        for _member_id in range(self.members_n):
            boot = rng.choice(len(xx), size=len(xx), replace=True, p=weights)
            xb, yb = xx[boot], y[boot]
            member = _Member(
                (rng.normal(size=(xx.shape[1], self.hidden)) / np.sqrt(xx.shape[1])).astype(np.float32),
                np.zeros(self.hidden, np.float32),
                (rng.normal(size=(self.hidden, 1)) / np.sqrt(self.hidden)).astype(np.float32),
                np.zeros(1, np.float32),
            )
            for _ in range(int(epochs)):
                ids = rng.integers(0, len(xb), size=min(int(batch_size), len(xb)))
                bx, by = xb[ids], yb[ids]
                h, logits = self._forward(member, bx)
                pred = self._sigmoid(logits)
                dl = (pred - by)[:, None] / len(ids)
                gw2, gb2 = h.T @ dl, dl.sum(axis=0)
                dh = (dl @ member.w2.T) * (1.0 - h * h)
                member.w2 -= learning_rate * gw2.astype(np.float32)
                member.b2 -= learning_rate * gb2.astype(np.float32)
                member.w1 -= learning_rate * (bx.T @ dh).astype(np.float32)
                member.b1 -= learning_rate * dh.sum(axis=0).astype(np.float32)
            _, logits = self._forward(member, xx)
            pred = self._sigmoid(logits)
            losses.append(float(-np.mean(y * np.log(pred + 1e-6) + (1-y) * np.log(1-pred + 1e-6))))
            self.members.append(member)
        mean, std, support = self.predict_records(rows)
        for row, sigma, sup in zip(rows, std, support):
            row["model_uncertainty"] = float(sigma)
            row["support_score"] = float(sup)
        return {"snapshots": len(rows), "members": len(self.members), "bce": float(np.mean(losses)), "mae": float(np.mean(np.abs(mean-y))), "mean_uncertainty": float(std.mean()), "mean_support": float(support.mean())}

    def predict_features(self, x: np.ndarray, *, support_k: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.members or self.train_x is None:
            raise RuntimeError("Viability ensemble has not been fitted")
        xx = self._transform(x)
        preds = np.stack([self._sigmoid(self._forward(m, xx)[1]) for m in self.members])
        mean = preds.mean(axis=0)
        std = preds.std(axis=0, ddof=1) if len(self.members) > 1 else np.zeros_like(mean)
        support = np.empty(len(xx), np.float32)
        for i, q in enumerate(xx):
            d = np.linalg.norm(self.train_x - q[None, :], axis=1)
            k = min(max(int(support_k)-1, 0), len(d)-1)
            kth = np.partition(d, k)[k]
            support[i] = np.exp(-(kth / max(1.0, np.sqrt(xx.shape[1])))**2)
        return mean.astype(np.float32), std.astype(np.float32), support

    def predict_records(self, records: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.predict_features(np.stack([physical_belief_feature(r) for r in records]))

    def save(self, path: str | Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"version": self.VERSION, "state": self.__dict__}, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "ViabilityEnsemble":
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        if int(payload.get("version", -1)) != cls.VERSION:
            raise ValueError("Incompatible ViabilityEnsemble file")
        obj = cls(); obj.__dict__.update(payload["state"]); return obj
