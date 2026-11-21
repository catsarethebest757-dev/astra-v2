"""Fixture checks for the ranker, engine, split, and local server."""
from __future__ import annotations

import json
import threading
import time
import unittest
from http.client import HTTPConnection

import numpy as np

from astranami.catalog import blank, cluster_franchises, derive_priors, drop_anime_duplicates
from astranami.config import ASPECTS
from astranami.engine import Engine
from astranami.features.graph import Graph
from astranami.index import Index
from astranami.learn.adapter import train_adapter
from astranami.learn.pairs import make_split
from astranami.learn.ranker import RankerNet, approx_ndcg_loss, train_ranker
from astranami.server import serve
from astranami.compare import compare_formats
from tests.synthetic import cluster_of, synthetic_index


def toy_index() -> Index:
    titles = [
        "Night Inquiry", "Night Inquiry Season 2", "Quiet Case", "Gray Verdict",
        "Iron Regiment", "Steel March", "Gear Front", "Pilot Oath",
        "Lunch Club", "Soft Punchline", "Club Room", "After School",
    ]
    themes = ["mystery detective", "mystery detective", "mystery detective", "mystery detective",
              "mecha robot", "mecha robot", "mecha robot", "mecha robot",
              "comedy school", "comedy school", "comedy school", "comedy school"]
    records = []
    for position, title in enumerate(titles):
        words = themes[position]
        records.append(blank(
            f"anime:al:{position}", title, "anime", synopsis=(words + " ") * 4, genres=[words.split()[0]],
            tags={words.split()[0]: 0.9, words.split()[1]: 0.7}, year=2000 + position, rating=8.0,
            popularity=80 + position, format="TV", episodes=12, runtime=24,
            studios=[f"studio {position // 4}"], creators=[f"director:person {position // 4}"],
        ))
    records[4]["kind"] = "tv"
    records[0]["related"] = [records[1]["id"]]
    records[1]["related"] = [records[0]["id"]]
    for position, record in enumerate(records):
        cluster = position // 4
        record["recs"] = {
            records[other]["id"]: 30
            for other in range(cluster * 4, cluster * 4 + 4)
            if other != position and {position, other} != {0, 1}
        }
    records[11]["adult"] = True
    cluster_franchises(records)
    derive_priors(records)
    index = Index(records)
    rng = np.random.default_rng(0)
    centroids = rng.normal(size=(3, 32)).astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    for aspect in ASPECTS:
        matrix = np.zeros((len(records), 32), dtype=np.float32)
        for position in range(len(records)):
            noise = 0.01 if position in {0, 2, 3} else 0.05
            vector = centroids[position // 4] + noise * rng.normal(size=32).astype(np.float32)
            matrix[position] = vector / np.linalg.norm(vector)
        index.aspects[aspect] = matrix.astype(np.float16)
    index.build_sparse(None)
    index.set_graph(index.all_edges())
    return index


class RankingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = toy_index()
        cls.engine = Engine(cls.index)

    def test_lookup_and_franchise_holdout(self):
        hits = self.engine.lookup("Night Inquiry")
        self.assertEqual(hits[0]["id"], "anime:al:0")
        payload = self.engine.recommend({"liked": ["anime:al:0"], "limit": 8, "diversity": 0})
        identifiers = [item["id"] for item in payload["results"]]
        self.assertIn("anime:al:2", identifiers[:3])
        self.assertNotIn("anime:al:1", identifiers)
        self.assertNotIn("anime:al:11", identifiers)
        self.assertTrue(all(item["kind"] != "tv" for item in self.engine.recommend(
            {"liked": ["anime:al:0"], "cross_format": False, "limit": 8})["results"]))
        late = self.engine.recommend({"liked": ["anime:al:0"], "year_min": 2010, "limit": 8})
        self.assertTrue(all(item["year"] >= 2010 for item in late["results"]))
        self.assertNotIn("anime:al:11", [item["id"] for item in late["results"]])
        json.dumps(payload, allow_nan=False)

    def test_dislike_demotes_shared_tone(self):
        plain = self.engine.recommend({"liked": ["anime:al:0"], "limit": 1, "diversity": 0})
        penalized = self.engine.recommend(
            {"liked": ["anime:al:0"], "disliked": ["anime:al:2"], "limit": 3, "diversity": 0})
        self.assertIn(plain["results"][0]["id"], {"anime:al:2", "anime:al:3"})
        self.assertNotIn("anime:al:2", [item["id"] for item in penalized["results"]])
        self.assertNotIn(penalized["results"][0]["id"], {"anime:al:2", "anime:al:3"})
        self.assertTrue(penalized["results"][0]["explanation"])

    def test_query_path_without_downloading_a_model(self):
        vector = np.asarray(self.index.aspects["whole"][2], dtype=np.float32)
        self.engine._encode_query = lambda text: vector
        payload = self.engine.recommend({"query": "mystery detective", "limit": 5, "diversity": 0})
        self.assertTrue(payload["results"])
        json.dumps(payload, allow_nan=False)
        self.engine._encode_query = lambda text: None

    def test_split_hides_seeds(self):
        edges = self.index.all_edges()
        split = make_split(self.index, edges, 0.2, 0.3, seed=3)
        held = set(split.test_seeds.tolist()) | set(split.ranker_seeds.tolist())
        self.assertTrue(split.test_seeds.size)
        for left, right in split.adapter_edges:
            self.assertTrue(left not in held and right not in held)
        for left, right in split.cold_graph:
            self.assertTrue(left not in set(split.test_seeds.tolist()) and right not in set(split.test_seeds.tolist()))

    def test_approx_ndcg_prefers_the_right_order(self):
        import torch
        gains = torch.tensor([3.0, 2.0, 0.0, 0.0])
        good = float(approx_ndcg_loss(torch.tensor([3.0, 2.0, 0.1, 0.0]), gains))
        bad = float(approx_ndcg_loss(torch.tensor([0.0, 0.1, 2.0, 3.0]), gains))
        self.assertLess(good, bad)
        self.assertEqual(RankerNet(22)(torch.zeros(4, 22)).shape, (4,))

    def test_validation_and_bad_host(self):
        with self.assertRaises(ValueError):
            self.engine.recommend({"liked": ["missing"]})
        port = 18765
        box = {}

        def ready(httpd):
            box["server"] = httpd

        thread = threading.Thread(target=serve, args=(self.engine, port, ready), daemon=True)
        thread.start()
        try:
            for _ in range(50):
                if "server" in box:
                    break
                time.sleep(0.05)
            self.assertIn("server", box)
            time.sleep(0.1)
            connection = HTTPConnection("127.0.0.1", port, timeout=2)
            connection.request("GET", "/api/stats", headers={"Host": "evil.example"})
            response = connection.getresponse()
            self.assertEqual(response.status, 403)
            connection.close()
            connection = HTTPConnection("127.0.0.1", port, timeout=2)
            connection.request("GET", "/api/stats", headers={"Host": f"127.0.0.1:{port}"})
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            body = json.loads(response.read().decode())
            self.assertEqual(body["count"], 12)
            connection.close()
        finally:
            if "server" in box:
                box["server"].shutdown()
                thread.join(timeout=2)
                box["server"].server_close()

    def test_catalog_dedup(self):
        anime = [blank("anime:al:9", "Bee", "anime", year=2010, format="TV", genres=["drama"], tags={"mystery": 0.5})]
        other = [blank("tv:tvmaze:9", "Bee", "tv", year=2010, genres=["anime"], tags={"mystery": 0.4}, language="Japanese")]
        kept, mapping = drop_anime_duplicates(anime, other)
        self.assertEqual(mapping["tv:tvmaze:9"], "anime:al:9")
        self.assertEqual(kept, [])

    def test_large_graph_spectral_vectors(self):
        edges = {(row, (row + 1) % 100): 3 for row in range(99)}
        edges[(0, 99)] = 3
        graph = Graph(100, edges, dimensions=8)
        self.assertEqual(graph.embedding.shape, (100, 8))
        self.assertTrue(np.isfinite(graph.embedding).all())
        self.assertTrue(np.linalg.norm(graph.embedding, axis=1).min() > 0.9)


class SyntheticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = synthetic_index()
        cls.engine = Engine(cls.index)
        cls.ids = [record["id"] for record in cls.index.records]

    def clusters(self, payload):
        return [cluster_of(self.index, item["id"]) for item in payload["results"]]

    def test_single_seed_stays_in_its_cluster(self):
        payload = self.engine.recommend({"liked": [self.ids[3]], "limit": 10, "diversity": 0})
        self.assertGreaterEqual(self.clusters(payload).count(0), 8)

    def test_outlying_like_is_not_crowded_out(self):
        liked = [self.ids[0], self.ids[5], self.ids[9], self.ids[25 * 7 + 2]]
        payload = self.engine.recommend({"liked": liked, "limit": 12})
        found = self.clusters(payload)
        self.assertGreaterEqual(found.count(7), 2)
        self.assertGreater(found.count(0), found.count(7))
        self.assertIn(payload["results"][0]["because"]["id"], liked)

    def test_query_is_encoded_once(self):
        calls = []
        vector = self.index.dense32("whole")[40]

        def encode(text):
            calls.append(text)
            return vector

        self.engine._encode_query = encode
        try:
            payload = self.engine.recommend({"query": "word6 word7 word8", "limit": 10})
        finally:
            del self.engine._encode_query
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.clusters(payload)[0], 1)

    def test_query_words_only_credit_real_matches(self):
        payload = self.engine.recommend({"query": "word6", "limit": 12, "explain": True})
        for item in payload["results"]:
            if "query words" in item["inspect"]["found_by"]:
                self.assertIn("word6", self.index.records[self.index.ids[item["id"]]]["synopsis"])

    def test_inspector_accounts_for_the_score(self):
        plain = self.engine.recommend({"liked": [self.ids[30]], "limit": 5})
        self.assertNotIn("inspect", plain["results"][0])
        payload = self.engine.recommend({"liked": [self.ids[30], self.ids[60]], "limit": 5, "explain": True})
        json.dumps(payload, allow_nan=False)
        for item in payload["results"]:
            inspect = item["inspect"]
            self.assertAlmostEqual(sum(inspect["score_parts"].values()), item["score"], delta=0.01)
            self.assertTrue(inspect["found_by"])
            self.assertTrue(inspect["features"])
            self.assertTrue(inspect["liked"])

    def test_excluded_tag_is_removed(self):
        payload = self.engine.recommend({"liked": [self.ids[3]], "limit": 10, "exclude_tags": ["tag0", "no such tag"]})
        self.assertTrue(payload["results"])
        for item in payload["results"]:
            self.assertNotIn("tag0", self.index.records[self.index.ids[item["id"]]]["tags"])

    def test_format_comparison(self):
        report = compare_formats(self.index, samples=40, k=10)
        whole = report["aspects"]["whole"]
        self.assertEqual(whole["float16 (stored)"]["recall"], 1.0)
        self.assertGreaterEqual(whole["int8"]["recall"], whole["binary"]["recall"])
        self.assertLess(whole["binary"]["megabytes"], whole["int8"]["megabytes"])


class TrainingSmoke(unittest.TestCase):
    def test_adapter_and_ranker_smoke(self):
        index = toy_index()
        edges = index.all_edges()
        messages = []
        model = train_adapter(index, edges, epochs=1, output_dim=32, batch_size=4, seed=1,
                              validation_fraction=0, progress=messages.append)
        import torch
        device = next(model.parameters()).device
        self.assertEqual(model(torch_input(index).to(device)).shape[-1], 32)
        ranker = train_ranker(index, np.array([2, 3, 4, 5, 6, 8], dtype=np.int64), edges,
                              epochs=1, seed=1, progress=messages.append)
        self.assertTrue(any("mined negatives" in message for message in messages))
        self.assertTrue(any("epoch 1/1, batch" in message for message in messages))
        self.assertTrue(any("built lists" in message for message in messages))
        engine = Engine(index, ranker)
        payload = engine.recommend({"liked": ["anime:al:0"], "limit": 5, "diversity": 0.2})
        self.assertTrue(payload["results"])
        json.dumps(payload, allow_nan=False)
        self.assertIn("aspects", payload["results"][0])


def torch_input(index):
    import torch
    return torch.from_numpy(index.adapter_input(np.arange(index.size)))


if __name__ == "__main__":
    unittest.main()
