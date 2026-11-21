"""The importer, the judged-file benchmark, and the ranker's training lists, on generated data."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from astranami.benchmark import benchmark_file
from astranami.importer import import_catalog
from astranami.index import Index
from astranami.learn.ranker import _training_lists
from astranami.store import load_json
from tests.synthetic import synthetic_index


def export(folder: Path, clusters: int = 6, per_cluster: int = 12, width: int = 16) -> dict:
    rng = np.random.default_rng(3)
    centers = rng.normal(size=(clusters, width))
    words = [f"word{number}" for number in range(clusters * 4)]
    titles, keys, vectors = [], [], []
    for cluster in range(clusters):
        for member in range(per_cluster):
            identifier = f"anilist:{cluster * 100 + member}" if cluster % 2 else f"tmdb:{cluster * 100 + member}"
            own = words[cluster * 4:cluster * 4 + 4]
            titles.append({
                "id": identifier, "media": "anime" if cluster % 2 else "movie", "name": f"Title {cluster}-{member}",
                "year": 2000 + member, "format": "TV", "genres": [f"genre{cluster}"],
                "tags": {own[0].upper(): 0.9, own[0]: 0.1, own[1]: 0.6}, "synopsis": " ".join(own * 3) + f" unique{member}",
                "score": 60 + member, "popularity": 100 * (member + 1), "episodes": 12,
            })
            keys.append(identifier)
            vectors.append(centers[cluster] + 0.3 * rng.normal(size=width))
    titles[0]["media"] = "series"
    titles.append({"id": "tmdb:9999", "media": "movie", "name": "No Vector"})
    matrix = np.asarray(vectors, dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    (folder / "titles.jsonl").write_text("\n".join(json.dumps(row) for row in titles), encoding="utf-8")
    (folder / "keys.json").write_text(json.dumps(keys), encoding="utf-8")
    np.save(folder / "vectors.npy", matrix.astype(np.float16))
    training = folder / "training"
    training.mkdir()
    prefixed = [f"film:{key}" if key.startswith("tmdb") else key for key in keys]
    rows = [{"key": key, "fam": "family-a" if position < 2 else key, "text": "Tone: quiet\nAppeal: slow burn"}
            for position, key in enumerate(prefixed)]
    (training / "titles.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    pairs = [{"a": prefixed[start], "b": prefixed[start + 3], "w": 0.8} for start in range(0, len(keys) - 3, 2)]
    (training / "pairs.jsonl").write_text("\n".join(json.dumps(row) for row in pairs), encoding="utf-8")
    return {"keys": keys, "prefixed": prefixed, "per_cluster": per_cluster}


class ImportTests(unittest.TestCase):
    def test_import_round_trip_and_benchmark(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            made = export(folder)
            data = folder / "data"
            quiet = lambda message: None
            built = import_catalog(data, folder / "titles.jsonl", folder / "keys.json", folder / "vectors.npy",
                                   folder / "training", progress=quiet)
            (data / "model").mkdir(parents=True)
            (data / "model" / "ranker.npz").write_bytes(b"stale")
            built = import_catalog(data, folder / "titles.jsonl", folder / "keys.json", folder / "vectors.npy",
                                   folder / "training", progress=quiet)
            self.assertFalse((data / "model" / "ranker.npz").exists())
            self.assertEqual(built.size, len(made["keys"]))
            self.assertEqual(built.records[built.ids["tmdb:0"]]["url"], "https://www.themoviedb.org/tv/0")
            self.assertEqual(built.records[built.ids["tmdb:1"]]["tags"]["word0"], 0.9)
            self.assertEqual(len({built.aspects[aspect].shape[1] for aspect in built.aspects}), 1)
            index = Index.load(load_json(data, "catalog"), data / "index")
            self.assertEqual(index.size, built.size)
            self.assertNotIn("tmdb:9999", index.ids)
            first, second = index.ids[made["keys"][0]], index.ids[made["keys"][1]]
            self.assertEqual(index.franchise[first], index.franchise[second])
            self.assertTrue(index.all_edges())
            for aspect in ("plot", "themes", "tone", "whole"):
                self.assertTrue(np.isfinite(np.asarray(index.aspects[aspect], dtype=np.float32)).all())
            whole = index.dense32("whole")
            nearest = np.argsort(-(whole @ whole[index.ids[made["keys"][40]]]))[1:6]
            self.assertTrue(all(index.records[row]["id"].startswith("anilist:3") for row in nearest))

            per = made["per_cluster"]
            judged = {"movies": [{"seed": made["prefixed"][0], "relevant": made["prefixed"][4:per]},
                                 {"seed": "film:tmdb:777777", "relevant": made["prefixed"][4:per]}],
                      "pools": {"movies": [key for key in made["prefixed"] if key.startswith("film:")]}}
            (folder / "judged.json").write_text(json.dumps(judged), encoding="utf-8")
            report = benchmark_file(index, folder / "judged.json", progress=quiet)
            self.assertEqual(report["movies"]["seeds"], 1)
            self.assertGreater(report["movies"]["whole"]["recall@10"], 0.8)
            self.assertGreater(report["movies"]["heuristic"]["ndcg@10"], 0.5)

    def test_import_without_pairs_still_serves(self):
        from astranami.pipeline import prepare
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            export(folder)
            quiet = lambda message: None
            import_catalog(folder / "data", folder / "titles.jsonl", folder / "keys.json", folder / "vectors.npy",
                           progress=quiet)
            engine = prepare(folder / "data", progress=quiet)
            self.assertIsNone(engine.ranker)
            results = engine.recommend({"liked": ["tmdb:1"], "exclude_tags": ["WORD1"], "limit": 5})["results"]
            self.assertTrue(results)
            self.assertTrue(all("word1" not in item["tags"] for item in results))

    def test_training_negatives_are_not_near_neighbors(self):
        index = synthetic_index()
        edges = index.all_edges()
        seeds = np.array(sorted({node for edge in edges for node in edge}), dtype=np.int64)[:40]
        lists = _training_lists(index, seeds, edges, 60, 1, progress=lambda message: None)
        self.assertTrue(lists)
        from astranami.learn.features import FEATURES
        column = FEATURES.index("cos_whole")
        positive = np.mean([features[gains > 0, column].mean() for features, gains in lists])
        negative = np.mean([features[gains == 0, column].mean() for features, gains in lists])
        self.assertGreater(positive, negative + 0.2)
        self.assertTrue(all((gains > 0).any() and (gains == 0).any() for _, gains in lists))


if __name__ == "__main__":
    unittest.main()
