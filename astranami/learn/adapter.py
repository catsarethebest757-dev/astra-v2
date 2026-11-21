"""Taste Adapter: distills crowd-perceived similarity ("fans of A recommend B") into content space.

A small residual network maps the four frozen aspect embeddings of a title to a 256-d unit vector.
It is trained with a vote-weighted, franchise-masked InfoNCE loss on recommendation pairs plus mined hard
negatives (titles that look alike on paper but that fans do not recommend together). Because it only
reads content, it scores brand-new titles and titles from other formats (TV, movies) that have no
recommendation data. An anchoring loss keeps the projected geometry correlated with the base embedding
space across the whole catalog so those out-of-distribution titles land sensibly.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ..config import ASPECTS
from ..index import Index
from ..util import log


class AdapterNet(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 1024, output: int = 256, dropout: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(nn.Dropout(dropout), nn.Linear(input_dim, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, output))
        self.skip = nn.Linear(input_dim, output, bias=False)
        self.log_temperature = nn.Parameter(torch.tensor(np.log(0.05), dtype=torch.float32))

    def forward(self, features):
        return F.normalize(self.mlp(self.norm(features)) + self.skip(features), dim=-1)


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _aspect_dropout(batch, blocks: int, probability: float):
    if probability <= 0:
        return batch
    keep = (torch.rand(batch.shape[0], blocks, device=batch.device) > probability).float()
    keep[keep.sum(dim=1) == 0, -1] = 1.0
    width = batch.shape[1] // blocks
    return batch * keep.repeat_interleave(width, dim=1) * blocks / keep.sum(dim=1, keepdim=True)


def mine_hard_negatives(index: Index, nodes: np.ndarray, edges: dict, per_node: int = 12, pool: int = 60,
                        progress=log) -> dict[int, np.ndarray]:
    """For each node: close neighbors in the base 'whole' space that are neither linked nor same-franchise."""
    device = _device()
    linked: dict[int, set[int]] = {}
    for left, right in edges:
        linked.setdefault(left, set()).add(right)
        linked.setdefault(right, set()).add(left)
    matrix = torch.from_numpy(np.asarray(index.aspects["whole"][nodes], dtype=np.float16)).to(device)
    if device.type == "cpu":
        matrix = matrix.float()
    output = {}
    for start in range(0, len(nodes), 2048):
        scores = matrix[start:start + 2048] @ matrix.T
        top = torch.topk(scores, min(pool + 1, len(nodes)), dim=1).indices.cpu().numpy()
        for row, neighbors in enumerate(top):
            node = int(nodes[start + row])
            known = linked.get(node, set())
            chosen = [int(nodes[position]) for position in neighbors
                      if nodes[position] != node and int(nodes[position]) not in known
                      and index.franchise[nodes[position]] != index.franchise[node]]
            output[node] = np.array(chosen[:per_node], dtype=np.int64)
        progress(f"Taste Adapter: mined negatives for {min(start + 2048, len(nodes)):,}/{len(nodes):,} titles")
    return output


def train_adapter(index: Index, edges: dict, epochs: int = 12, output_dim: int = 256, batch_size: int = 512,
                  seed: int = 13, validation_fraction: float = 0.05, progress=log) -> AdapterNet:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = _device()
    pairs = np.array(list(edges.keys()), dtype=np.int64)
    votes = np.array(list(edges.values()), dtype=np.float32)
    if len(pairs) < 2:
        raise ValueError("The taste adapter needs at least two recommendation pairs.")
    order = rng.permutation(len(pairs))
    if validation_fraction <= 0 or len(pairs) < 8:
        validation, training = order[:0], order
    else:
        validation_count = max(1, int(len(pairs) * validation_fraction))
        validation, training = order[:validation_count], order[validation_count:]
    nodes = np.unique(pairs)
    progress(f"Taste Adapter: preparing {len(pairs):,} pairs across {len(nodes):,} titles")
    hard = mine_hard_negatives(index, nodes, edges, progress=progress)
    progress("Taste Adapter: loading aspect vectors")
    features = torch.from_numpy(index.adapter_input(np.arange(index.size))).to(device, dtype=torch.float16)
    franchise = torch.from_numpy(index.franchise).to(device)
    model = AdapterNet(features.shape[1], output=output_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.02)
    step = min(batch_size, len(training))
    batches_per_epoch = max(1, len(training) // step)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, total_steps=max(epochs * batches_per_epoch, 1), pct_start=0.1)
    weights = torch.from_numpy(np.log1p(votes) / np.log1p(votes).mean()).to(device)
    blocks = len(ASPECTS)
    best_state, best_recall = None, -1e18
    progress(f"Taste Adapter: {len(training):,} training pairs, {epochs} epochs on {device}")
    for epoch in range(epochs):
        model.train()
        permutation = rng.permutation(training)
        total = 0.0
        for start in range(0, batches_per_epoch * step, step):
            batch = permutation[start:start + step]
            left = torch.from_numpy(pairs[batch, 0]).to(device)
            right = torch.from_numpy(pairs[batch, 1]).to(device)
            flip = torch.rand(len(batch), device=device) < 0.5
            left, right = torch.where(flip, right, left), torch.where(flip, left, right)
            negatives = torch.from_numpy(np.stack([
                rng.choice(hard.get(int(node), ()), 4, replace=len(hard.get(int(node), ())) < 4)
                if len(hard.get(int(node), ())) else rng.integers(0, index.size, 4)
                for node in left.cpu().numpy()])).to(device)
            anchor = model(_aspect_dropout(features[left].float(), blocks, 0.15))
            positive = model(_aspect_dropout(features[right].float(), blocks, 0.15))
            negative = model(features[negatives.reshape(-1)].float()).reshape(len(batch), 4, -1)
            temperature = model.log_temperature.exp().clamp(0.01, 0.2)
            logits = anchor @ positive.T / temperature
            collision = ((franchise[left][:, None] == franchise[right][None, :])
                         | (left[:, None] == left[None, :]) | (right[:, None] == right[None, :]))
            collision.fill_diagonal_(False)
            logits = logits.masked_fill(collision, -1e4)
            hard_logits = torch.einsum("bd,bkd->bk", anchor, negative) / temperature
            forward = F.cross_entropy(torch.cat([logits, hard_logits], dim=1),
                                      torch.arange(len(batch), device=device), reduction="none")
            backward = F.cross_entropy(logits.T, torch.arange(len(batch), device=device), reduction="none")
            contrastive = ((forward + backward) * 0.5 * weights[batch]).mean()
            sample = torch.randint(0, index.size, (256,), device=device)
            projected = model(features[sample].float())
            base = F.normalize(features[sample].float()[:, -features.shape[1] // blocks:], dim=-1)
            anchoring = 1 - _correlation(projected @ projected.T, base @ base.T)
            loss = contrastive + 0.3 * anchoring
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += float(loss)
            batch_number = start // step + 1
            if batch_number % max(1, batches_per_epoch // 4) == 0 or batch_number == batches_per_epoch:
                progress(f"Taste Adapter: epoch {epoch + 1}/{epochs}, batch {batch_number}/{batches_per_epoch}")
        recall = (_validation_recall(model, features, pairs[validation], nodes, device)
                  if len(validation) else -total)
        progress(f"Taste Adapter: epoch {epoch + 1}/{epochs} complete, loss {total / max(batches_per_epoch, 1):.4f}"
            + (f", validation recall@50 {recall:.4f}" if len(validation) else ""))
        if recall > best_recall:
            best_recall = recall
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def _correlation(left, right):
    mask = ~torch.eye(left.shape[0], dtype=torch.bool, device=left.device)
    x, y = left[mask], right[mask]
    x, y = x - x.mean(), y - y.mean()
    return (x * y).sum() / (x.norm() * y.norm() + 1e-8)


@torch.no_grad()
def _validation_recall(model, features, pairs, nodes, device, cutoff: int = 50) -> float:
    model.eval()
    node_tensor = torch.from_numpy(nodes).to(device)
    vectors = torch.cat([model(features[node_tensor[start:start + 8192]].float())
                         for start in range(0, len(nodes), 8192)])
    position = {int(node): offset for offset, node in enumerate(nodes)}
    sample = pairs[:2000]
    left = torch.tensor([position[int(node)] for node in sample[:, 0]], device=device)
    right = torch.tensor([position[int(node)] for node in sample[:, 1]], device=device)
    scores = vectors[left] @ vectors.T
    scores[torch.arange(len(left)), left] = -1e4
    target = scores[torch.arange(len(left)), right]
    rank = (scores > target[:, None]).sum(dim=1)
    model.train()
    return float((rank < cutoff).float().mean())


@torch.no_grad()
def project(model: AdapterNet, index: Index) -> np.ndarray:
    device = next(model.parameters()).device
    output = []
    for start in range(0, index.size, 8192):
        rows = np.arange(start, min(start + 8192, index.size))
        batch = torch.from_numpy(index.adapter_input(rows)).to(device)
        output.append(model(batch).float().cpu().numpy())
    return np.concatenate(output).astype(np.float32)
