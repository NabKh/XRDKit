# Deploying xrdkit as a no-install web app

The goal: give experimentalists a **URL** they open in a browser — no Python,
no `pip`, no environment. The app you run locally with `streamlit run app.py` is
the same one that gets hosted; only the host changes.

The recommended host is **Streamlit Community Cloud** — free, backed directly by
a GitHub repo, and the natural home for a `streamlit` app. Alternatives (Hugging
Face Spaces, Render, a self-hosted VM) are listed at the bottom.

---

## Option A — Streamlit Community Cloud (recommended, free)

**One-time setup (~5 min):**

1. Push this repository to a **public** GitHub repo (e.g. `your-name/xrdkit`).
   Everything needed is already here: `app.py` at the root, `requirements.txt`,
   and `.streamlit/config.toml`.
2. Go to <https://share.streamlit.io> and sign in with your GitHub account.
3. Click **"Create app"** → **"Deploy a public app from GitHub"**.
4. Fill in:
   - **Repository:** `your-name/xrdkit`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **Deploy**. The first build takes a few minutes (it installs
   `requirements.txt`). When it finishes you get a permanent URL like
   `https://xrdkit.streamlit.app` — share that with experimentalists.

That's it. Every `git push` to `main` redeploys automatically.

**Add a one-click badge to the README** (replace the URL with yours):

```markdown
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://xrdkit.streamlit.app)
```

### Resource notes (free tier = ~1 GB RAM)
- `pymatgen` + `mp-api` are heavy. They are only needed for the **CIF/POSCAR
  upload** and **Materials Project search** tabs. The built-in ICDD cards, the
  COD search, the measured-pattern overlay, and the **observed − simulated
  difference** feature all work without them.
- If the free tier struggles with the pymatgen install, either (a) accept the
  longer cold-start, or (b) move `pymatgen`/`mp-api` to optional and lazy-import
  them inside `db.py` so the core app boots light. Ask and this can be wired up.
- The **Materials Project** tab needs each user's own free API key — it is typed
  into the app at runtime, never stored in the repo.

---

## Option B — Hugging Face Spaces (free, good for science audiences)

1. Create a new **Space** at <https://huggingface.co/new-space>, SDK = **Streamlit**.
2. Push this repo's contents into the Space (it reads `app.py` + `requirements.txt`).
3. You get a URL like `https://huggingface.co/spaces/your-name/xrdkit`.

Spaces gives a bit more RAM than Streamlit Cloud's free tier and is familiar to
many materials/ML researchers.

## Option C — Self-hosted (institutional server / VM)

```bash
pip install -r requirements.txt
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Put it behind nginx with TLS if it should be reachable outside the LAN. This is
the route if your data cannot leave institutional infrastructure.

---

## Privacy note to put in front of experimentalists
Uploaded patterns are processed in-memory by the running app instance and are
not persisted by xrdkit itself. On a shared public host (Options A/B) the data
still transits a third-party server, so for unpublished/sensitive measurements
prefer Option C or running locally.
