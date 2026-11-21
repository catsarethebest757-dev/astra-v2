"""Listwise reranker over pair features.

Trained with ApproxNDCG plus a pairwise margin, with the collaborative-graph block dropped on half of
the updates. Dropout is what lets the same ranker score TV, movies, and brand-new anime that have no
fan edges: it has to be right from content alone, and it still uses the graph when the graph is there.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ..index import Index
from ..learn.features import FEATURES, GRAPH_FEATURES, pair_features
from ..learn.pairs import adjacency, relevant
from ..util import log


HEURISTIC = np.zeros(len(FEATURES), dtype=np.float32)
HEURISTIC[FEATURES.index("cos_plot")] = 0.16
HEURISTIC[FEATURES.index("cos_themes")] = 0.14
HEURISTIC[FEATURES.index("cos_tone")] = 0.12
HEURISTIC[FEATURES.index("cos_whole")] = 0.20
HEURISTIC[FEATURES.index("cos_adapter")] = 0.28
HEURISTIC[FEATURES.index("bm25")] = 0.08
HEURISTIC[FEATURES.index("tag_cos")] = 0.08
HEURISTIC[FEATURES.index("soft_tag_cos")] = 0.10
HEURISTIC[FEATURES.index("creator_sim")] = 0.08
HEURISTIC[FEATURES.index("studio_sim")] = 0.04
HEURISTIC[FEATURES.index("year_gap")] = -0.03
HEURISTIC[FEATURES.index("form_match")] = 0.02
HEURISTIC[FEATURES.index("length_ratio")] = -0.02
HEURISTIC[FEATURES.index("source_match")] = 0.02
HEURISTIC[FEATURES.index("graph_direct")] = 0.18
HEURISTIC[FEATURES.index("graph_2hop")] = 0.08
HEURISTIC[FEATURES.index("graph_spectral")] = 0.05
HEURISTIC[FEATURES.index("cand_quality")] = 0.04


# Only titles with fan recommendations can be a training positive, and those are the popular, well
# documented ones. A ranker that sees these columns learns "popular means relevant" and then buries
# the obscure titles it is asked about; the engine applies popularity and quality as explicit priors.
SELECTION_BIASED = ("has_graph", "cand_pop", "cand_quality", "seed_pop", "cand_has_synopsis", "source_match")


def heuristic_scores(features: np.ndarray) -> np.ndarray:
    return np.asarray(features, dtype=np.float32) @ HEURISTIC


def approx_ndcg_loss(scores: torch.Tensor, gains: torch.Tensor, alpha: float = 12.0) -> torch.Tensor:
    """1 - ApproxNDCG. Soft ranks use a sigmoid pairwise comparison with the diagonal removed."""
    pairwise = scores.unsqueeze(0) - scores.unsqueeze(1)
    eye = torch.eye(scores.shape[0], dtype=torch.bool, device=scores.device)
    ranks = 1 + torch.sigmoid(alpha * pairwise).masked_fill(eye, 0).sum(dim=1)
    dcg = (gains / torch.log2(1 + ranks)).sum()
    ideal = torch.sort(gains, descending=True).values
    positions = torch.arange(1, scores.shape[0] + 1, device=scores.device, dtype=scores.dtype)
    idcg = (ideal / torch.log2(1 + positions)).sum().clamp(min=1e-6)
    return 1 - dcg / idcg


def _ndcg(scores: np.ndarray, gains: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    ranked = gains[order]
    ideal = np.sort(gains)[::-1]
    positions = np.arange(2, len(gains) + 2)
    def _dcg(values):
        return float(np.sum(values / np.log2(positions)))
    ideal_dcg = _dcg(ideal)
    return _dcg(ranked) / ideal_dcg if ideal_dcg > 1e-8 else 0.0


class RankerNet(nn.Module):
    """Small on purpose: a few thousand sparse lists overfit anything larger."""

    def __init__(self, input_dim: int, hidden: int = 32, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        keep = torch.ones(input_dim)
        if input_dim == len(FEATURES):
            keep[[FEATURES.index(name) for name in SELECTION_BIASED]] = 0
        self.register_buffer("keep", keep)

    def forward(self, features):
        return self.net(features * self.keep).squeeze(-1)


class Ranker:
    def __init__(self, model: RankerNet, mean: np.ndarray, std: np.ndarray):
        self.model = model.eval()
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _tensor(self, features: np.ndarray, grad: bool = False) -> torch.Tensor:
        scaled = (np.asarray(features, dtype=np.float32) - self.mean) / self.std
        tensor = torch.from_numpy(np.ascontiguousarray(scaled)).to(self.device)
        return tensor.requires_grad_(grad) if grad else tensor

    def score(self, features: np.ndarray) -> np.ndarray:
        if len(features) == 0:
            return np.zeros(0, dtype=np.float32)
        with torch.no_grad():
            return self.model(self._tensor(features)).float().cpu().numpy()

    def attribute(self, features: np.ndarray) -> np.ndarray:
        """Gradient-times-input contribution of each pair feature to the score."""
        if len(features) == 0:
            return np.zeros_like(features, dtype=np.float32)
        self.model.zero_grad(set_to_none=True)
        inputs = self._tensor(features, grad=True)
        scores = self.model(inputs)
        scores.sum().backward()
        return (inputs.grad * inputs).detach().float().cpu().numpy()


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _training_lists(index: Index, seeds: np.ndarray, label_edges: dict, list_size: int, seed: int,
                    progress=log) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    if len(seeds) > 2500:
        seeds = np.sort(rng.choice(seeds, 2500, replace=False))
    progress(f"Ranker: preparing training lists for {len(seeds):,} seeds")
    graph = adjacency(label_edges)
    lists = []
    for row, item in enumerate(seeds):
        item = int(item)
        grades = relevant(index, item, graph.get(item, {}))
        if not grades:
            continue
        def usable(values):
            values = np.unique(np.asarray(values, dtype=np.int64))
            values = values[(index.franchise[values] != index.franchise[item]) & ~index.adult[values]]
            return values[[int(value) not in grades for value in values]]

        positive = rng.permutation(np.array(list(grades), dtype=np.int64))[:list_size // 2]
        room = list_size - len(positive)
        # Fan recommendations are sparse, so a close neighbor without one is usually a good match nobody
        # has written down yet. Lists built from near neighbors teach the ranker that similarity is a bad
        # sign (measured on a 118k-title catalog), so every negative is a random title.
        negative = usable(rng.integers(0, index.size, size=room * 2))[:room]
        if not len(negative):
            continue
        keep = np.concatenate([positive, negative])
        gains = np.array([grades.get(int(value), 0.0) for value in keep], dtype=np.float32)
        lists.append((pair_features(index, item, keep), gains))
        if (row + 1) % 100 == 0 or row + 1 == len(seeds):
            progress(f"Ranker: built lists for {row + 1:,}/{len(seeds):,} seeds")
    return lists


def train_ranker(index: Index, seeds: np.ndarray, edges: dict, epochs: int = 30, list_size: int = 200,
                 seed: int = 13, graph_dropout: float = 0.5, progress=log,
                 label_edges: dict | None = None) -> Ranker:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    # Labels can include hidden edges; index.graph contains only the edges visible
    # to the model. Keeping these separate is essential for honest cold-start training.
    lists = _training_lists(index, seeds, edges if label_edges is None else label_edges,
                            list_size, seed, progress=progress)
    if len(lists) < 2:
        raise ValueError("Not enough recommendation lists to train the ranker.")
    stacked = np.concatenate([features for features, _ in lists])
    mean = stacked.mean(axis=0).astype(np.float32)
    std = np.maximum(stacked.std(axis=0), 1e-3).astype(np.float32)
    device = _device()
    model = RankerNet(len(FEATURES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    mean_t = torch.from_numpy(mean).to(device)
    std_t = torch.from_numpy(std).to(device)
    graph_cols = torch.tensor(GRAPH_FEATURES, device=device)
    missing_graph = -mean_t[graph_cols] / std_t[graph_cols]
    split = max(1, len(lists) // 10)
    validation, training = lists[:split], lists[split:]
    best_state, best_score, patience = None, -1e18, 5
    progress(f"Ranker: {len(training):,} lists, {epochs} epochs on {device}")
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(training))
        total = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, row in enumerate(order, start=1):
            features, gains = training[int(row)]
            inputs = (torch.from_numpy(features).to(device) - mean_t) / std_t
            if rng.random() < graph_dropout:
                inputs = inputs.clone()
                # Graph dropout must look like *raw* zero graph features after
                # normalization, which is what unseen titles have at serving time.
                inputs[:, graph_cols] = missing_graph
            scores = model(inputs)
            target = torch.from_numpy(gains).to(device)
            loss = approx_ndcg_loss(scores, target)
            positive, negative = scores[target > 0], scores[target == 0]
            if len(positive) and len(negative):
                take_p = positive if len(positive) <= 16 else positive[torch.randperm(len(positive), device=device)[:16]]
                take_n = negative if len(negative) <= 16 else negative[torch.randperm(len(negative), device=device)[:16]]
                loss = loss + 0.25 * F.softplus(-(take_p[:, None] - take_n[None, :])).mean()
            (loss / 16).backward()
            if step % 16 == 0 or step == len(order):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach())
            if step % max(1, len(order) // 4) == 0 or step == len(order):
                progress(f"Ranker: epoch {epoch + 1}/{epochs}, list {step:,}/{len(order):,}")
        model.eval()
        with torch.no_grad():
            checked = [validation[i] for i in range(0, len(validation), max(1, len(validation) // 32))]
            # Judge each list with and without its graph features, so the checkpoint that is kept is
            # also good for titles nobody has recommended yet.
            scores = []
            for features, gains in checked:
                inputs = (torch.from_numpy(features).to(device) - mean_t) / std_t
                masked = inputs.clone()
                masked[:, graph_cols] = missing_graph
                scores.append(0.5 * (_ndcg(model(inputs).cpu().numpy(), gains)
                                     + _ndcg(model(masked).cpu().numpy(), gains)))
            score = float(np.mean(scores)) if scores else -total / max(len(training), 1)
        progress(f"Ranker: epoch {epoch + 1}/{epochs} complete, loss {total / max(len(training), 1):.4f}, validation nDCG {score:.4f}")
        if score > best_score + 1e-4:
            best_score = score
            patience = 5
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            patience -= 1
            if patience <= 0 and epoch >= 4:
                progress("Ranker: early stop")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return Ranker(model, mean, std)


def save_ranker(path: Path, ranker: Ranker) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key.replace(".", "__"): value.detach().cpu().numpy() for key, value in ranker.model.state_dict().items()}
    payload["__mean"] = ranker.mean
    payload["__std"] = ranker.std
    temporary = path.with_suffix(".part.npz")
    np.savez(temporary, **payload)
    temporary.replace(path)


def load_ranker(path: Path) -> Ranker:
    data = np.load(path, allow_pickle=False)
    model = RankerNet(len(FEATURES))
    state = {key.replace("__", "."): torch.from_numpy(data[key]) for key in data.files if not key.startswith("__")}
    try:
        model.load_state_dict(state)
    except RuntimeError as error:
        raise ValueError("This ranker was saved by an older version; run `python -m astranami train` again.") from error
    return Ranker(model, data["__mean"], data["__std"])
