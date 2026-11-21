# Astra v2

A Python prototype for recommending anime, TV, and movies. It combines plot, theme, and tone embeddings with a taste adapter trained on fan recommendation pairs.

Clone the repository. With Python 3.10 or newer, run:

```sh
python -m pip install -r requirements.txt
python -m astranami self-test
```

On Windows, open `run.bat` to launch the app. The repository includes offline regression tests. The first full run downloads the public catalog and model dependencies separately.

The next milestone is to inspect why each title was recommended, then compare compressed indexes for size, latency, and ranking changes on older devices.
