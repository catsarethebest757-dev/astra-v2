"""Desktop launcher: build the catalog, train the ranker, and open the local UI."""
from __future__ import annotations

import os
import queue
import sys
import threading
import time
import traceback
import webbrowser
from pathlib import Path
from tkinter import BOTH, DISABLED, END, LEFT, NORMAL, WORD, BooleanVar, StringVar, Tk, ttk
from tkinter.scrolledtext import ScrolledText

ROOT = Path(__file__).resolve().parent
DEFAULT_DIR = ".aftercredits"
DEFAULT_PORT = 8765


class StreamToQueue:
    def __init__(self, original, log_queue):
        self.original = original
        self.log_queue = log_queue

    def write(self, text):
        if self.original:
            try:
                self.original.write(text)
            except Exception:
                pass
        if text:
            self.log_queue.put(text)

    def flush(self):
        if self.original:
            try:
                self.original.flush()
            except Exception:
                pass


class App:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("Aftercredits")
        self.root.geometry("980x760")
        self.root.minsize(760, 560)
        self.engine = None
        self.server = None
        self.busy = False
        self._phase = ""
        self._phase_started = 0.0
        self.liked = []
        self.found = []
        self.log_queue = queue.Queue()
        self.status = StringVar(value="Click Run. The first build downloads data, encodes titles, and trains the ranker.")
        self.kind = StringVar(value="all")
        self.query = StringVar()
        self.title_query = StringVar()
        self.open_browser = BooleanVar(value=True)
        self._build()
        self.root.after(120, self._drain_log)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _build(self):
        pad = {"padx": 12, "pady": 6}
        top = ttk.Frame(self.root, padding=14)
        top.pack(fill="x")
        ttk.Label(top, text="AFTERCREDITS", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(top, text="Crowd-distilled similarity for anime, TV, and movies.").pack(anchor="w")

        actions = ttk.Frame(self.root, padding=(14, 0))
        actions.pack(fill="x")
        self.run_btn = ttk.Button(actions, text="Run", command=self.run)
        self.run_btn.pack(side=LEFT, padx=(0, 8))
        ttk.Button(actions, text="Open browser", command=self._open_browser).pack(side=LEFT, padx=(0, 8))
        ttk.Checkbutton(actions, text="Open browser after Run", variable=self.open_browser).pack(side=LEFT)
        self.progress = ttk.Progressbar(actions, mode="indeterminate", length=160)
        self.progress.pack(side=LEFT, padx=12)

        search = ttk.LabelFrame(self.root, text="Search", padding=12)
        search.pack(fill="x", **pad)
        find_row = ttk.Frame(search)
        find_row.pack(fill="x")
        ttk.Label(find_row, text="Like a title").pack(side=LEFT)
        self.title_entry = ttk.Entry(find_row, textvariable=self.title_query)
        self.title_entry.pack(side=LEFT, fill="x", expand=True, padx=8)
        self.find_btn = ttk.Button(find_row, text="Find", command=self.lookup, state=DISABLED)
        self.find_btn.pack(side=LEFT)
        self.title_entry.bind("<Return>", lambda _event: self.lookup())

        self.matches = ttk.Combobox(search, state="readonly")
        self.matches.pack(fill="x", pady=(8, 0))
        self.matches.bind("<<ComboboxSelected>>", self._like_selected)
        liked_row = ttk.Frame(search)
        liked_row.pack(fill="x", pady=(6, 0))
        self.liked_label = ttk.Label(liked_row, text="Liked: none")
        self.liked_label.pack(side=LEFT)
        ttk.Button(liked_row, text="Like selected", command=self._like_selected).pack(side=LEFT, padx=8)
        ttk.Button(liked_row, text="Clear likes", command=self._clear_likes).pack(side=LEFT)

        mood = ttk.Frame(search)
        mood.pack(fill="x", pady=(8, 0))
        ttk.Label(mood, text="Mood / query").pack(side=LEFT)
        self.query_entry = ttk.Entry(mood, textvariable=self.query)
        self.query_entry.pack(side=LEFT, fill="x", expand=True, padx=8)
        ttk.Label(mood, text="Kind").pack(side=LEFT)
        ttk.Combobox(mood, textvariable=self.kind, values=("all", "anime", "movie", "tv"),
                     width=8, state="readonly").pack(side=LEFT, padx=(4, 8))
        self.discover_btn = ttk.Button(mood, text="Discover", command=self.discover, state=DISABLED)
        self.discover_btn.pack(side=LEFT)
        self.query_entry.bind("<Return>", lambda _event: self.discover())

        ttk.Label(self.root, textvariable=self.status, wraplength=940).pack(anchor="w", padx=14)
        self.results = ScrolledText(self.root, height=16, wrap=WORD, state=DISABLED)
        self.results.pack(fill=BOTH, expand=True, **pad)
        ttk.Label(self.root, text="Log").pack(anchor="w", padx=14)
        self.log = ScrolledText(self.root, height=10, wrap=WORD, state=DISABLED)
        self.log.pack(fill=BOTH, expand=True, **pad)

    def _set_search(self, enabled: bool):
        state = NORMAL if enabled else DISABLED
        self.find_btn.configure(state=state)
        self.discover_btn.configure(state=state)

    def _append_log(self, text: str):
        self.log.configure(state=NORMAL)
        self.log.insert(END, text)
        self.log.see(END)
        self.log.configure(state=DISABLED)

    def _drain_log(self):
        try:
            while True:
                self._append_log(self.log_queue.get_nowait())
        except queue.Empty:
            pass
        self.root.after(120, self._drain_log)

    def _show_results(self, text: str):
        self.results.configure(state=NORMAL)
        self.results.delete("1.0", END)
        self.results.insert(END, text)
        self.results.configure(state=DISABLED)

    def _open_browser(self):
        webbrowser.open(f"http://127.0.0.1:{DEFAULT_PORT}")

    def run(self):
        if self.busy:
            return
        if self.engine and self.server:
            self.status.set("Already running.")
            if self.open_browser.get():
                self._open_browser()
            return
        self.busy = True
        self.run_btn.configure(state=DISABLED)
        self.progress.start(12)
        self._show_phase("Starting the build")
        self.root.after(1000, self._refresh_elapsed)
        threading.Thread(target=self._run_worker, daemon=True).start()

    def _show_phase(self, message: str):
        self._phase = message
        self._phase_started = time.monotonic()
        self.status.set(message)

    def _refresh_elapsed(self):
        if not self.busy:
            return
        elapsed = int(time.monotonic() - self._phase_started)
        self.status.set(f"{self._phase}  ·  working for {elapsed}s")
        self.root.after(1000, self._refresh_elapsed)

    def _run_worker(self):
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout = StreamToQueue(stdout, self.log_queue)
        sys.stderr = StreamToQueue(stderr, self.log_queue)
        try:
            os.chdir(ROOT)
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))
            from astranami.pipeline import prepare
            from astranami.util import relative_path

            def progress(message: str) -> None:
                self.log_queue.put(message + "\n")
                self.root.after(0, lambda text=message: self._show_phase(text))

            self.engine = prepare(relative_path(DEFAULT_DIR), progress=progress)
            self.root.after(0, self._on_ready)
        except Exception as error:
            self.log_queue.put("".join(traceback.format_exception(error)) + "\n")
            self.root.after(0, lambda err=error: self._on_fail(err))
        finally:
            sys.stdout = stdout
            sys.stderr = stderr

    def _on_ready(self):
        count = self.engine.index.size if self.engine else 0
        self.progress.stop()
        self.status.set(f"Ready. {count:,} titles. Search below or open the browser.")
        self._set_search(True)
        self.run_btn.configure(state=NORMAL, text="Running")
        self.busy = False
        if self.server is None:
            threading.Thread(target=self._serve, daemon=True).start()
        if self.open_browser.get():
            self.root.after(400, self._open_browser)

    def _on_fail(self, error):
        self.progress.stop()
        self.status.set(f"Failed: {error}")
        self.busy = False
        self.run_btn.configure(state=NORMAL)

    def _serve(self):
        from astranami.server import serve

        def ready(httpd):
            self.server = httpd

        try:
            serve(self.engine, DEFAULT_PORT, ready=ready)
        except Exception as error:
            self.log_queue.put(f"Server error: {error}\n")
            self.root.after(0, lambda err=error: self.status.set(f"Server failed: {err}"))

    def lookup(self):
        if not self.engine:
            return
        query = self.title_query.get().strip()
        if not query:
            return
        try:
            self.found = self.engine.lookup(query, limit=12)
        except Exception as error:
            self.status.set(str(error))
            return
        if not self.found:
            self.matches.set("")
            self.matches["values"] = ()
            self.status.set("No titles found.")
            return
        labels = [f"{item['title']}  [{item['kind']}]  {item.get('year') or ''}  ::{item['id']}" for item in self.found]
        self.matches["values"] = labels
        self.matches.current(0)
        self.status.set(f"{len(self.found)} title matches.")

    def _like_selected(self, _event=None):
        value = self.matches.get()
        if "::" not in value:
            return
        identifier = value.rsplit("::", 1)[1]
        title = value.split("  [", 1)[0]
        if any(item["id"] == identifier for item in self.liked):
            return
        self.liked.append({"id": identifier, "title": title})
        self._refresh_likes()

    def _clear_likes(self):
        self.liked.clear()
        self._refresh_likes()

    def _refresh_likes(self):
        text = "Liked: " + ", ".join(item["title"] for item in self.liked) if self.liked else "Liked: none"
        self.liked_label.configure(text=text)

    def discover(self):
        if not self.engine:
            return
        request = {
            "query": self.query.get().strip(),
            "liked": [item["id"] for item in self.liked],
            "kind": self.kind.get(),
            "limit": 12,
        }
        if not request["query"] and not request["liked"]:
            self.status.set("Type a mood or like at least one title.")
            return
        try:
            payload = self.engine.recommend(request)
        except Exception as error:
            self.status.set(str(error))
            return
        lines = [payload.get("message", ""), ""]
        for item in payload["results"]:
            year = item.get("year") or "year unknown"
            rating = f"{item['rating']:.1f}" if item.get("rating") else "unrated"
            tags = " / ".join(item.get("tags", [])[:8])
            synopsis = (item.get("synopsis") or "No synopsis in this source.")[:350]
            lines += [
                item["title"],
                f"  {item['kind'].upper()} · {year} · {rating} · score {item.get('score', '')}",
                f"  {tags}" if tags else "",
                f"  {synopsis}",
                f"  {item.get('explanation', '')}",
                "",
            ]
        self._show_results("\n".join(line for line in lines if line is not None))
        self.status.set(f"{len(payload['results'])} discoveries.")

    def _close(self):
        server = self.server
        if server is not None:
            threading.Thread(target=server.shutdown, daemon=True).start()
        self.root.destroy()


def main():
    root = Tk()
    ttk.Style().theme_use("clam")
    App(root)
    root.mainloop()


if __name__ == "__main__":
    raise SystemExit(main())
