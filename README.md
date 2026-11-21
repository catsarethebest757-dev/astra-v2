# Astra v2

A Python prototype for recommending anime, TV, and movies. It combines plot, theme, and tone embeddings with a taste adapter trained on fan recommendation pairs.

Clone the repository. With Python 3.10 or newer, run:

```sh
python -m pip install -r requirements.txt
python -m astranami self-test
```

On Windows, open `run.bat` to launch the app. The repository includes offline regression tests. The first full run downloads the public catalog and model dependencies separately.

## Using it

```sh
python -m astranami prepare-serve                      # build whatever is missing, then open the local UI
python -m astranami search "cowboy bebop"              # find a title id
python -m astranami similar --id anime:al:1 --explain  # rank similar titles with a score breakdown
python -m astranami similar --id anime:al:1 --id anime:al:5114 --query "slow melancholy mystery"
python -m astranami compare-index                      # what int8 and binary vectors would cost
python -m astranami import --titles titles.jsonl.gz --keys vector_keys.json --vectors vectors.npy --training nami_train
python -m astranami benchmark eval.json                # engine and baselines against your own judged lists
```

Liking several titles gives each one a fair share of the list, so one outlying like is not crowded out by a group of similar likes. Each result names the liked title it matches best.

Tick "Show why each title ranked" in the UI, pass `--explain` on the command line, or send `"explain": true` to `/api/recommend` to see, per result: which retrieval views found it, how the score splits into taste, query, agreement, quality, and dislike parts, its score against each liked title, and its strongest features.

`import` builds the catalog and index from files you already have instead of downloading sources and running an embedding model: title records (JSON lines), a vector per title, and optionally a folder with `titles.jsonl` (franchise families, curated notes) and `pairs.jsonl` (recommendation pairs). The supplied vectors become the "whole" aspect; plot, themes, and tone are computed locally from each title's text and tags. Run `train` afterwards. Data stays in `.aftercredits/`, which git ignores.

`benchmark` scores each single signal, the heuristic scorer, and the trained ranker on a judged file of the form `{"group": [{"seed": id, "relevant": [ids]}], "pools": {"group": [ids]}}`.

`compare-index` reports, for each embedding aspect, how many of the stored float16 vectors' nearest neighbors survive in int8, binary, and binary-with-int8-rescoring formats, next to the size of each.
