# Semantic Geometry Explorer

> An interactive 3D microscope for the inside of a language model. Watch meaning
> take physical shape as a sentence flows through a transformer — token by
> token, layer by layer.

**▶ Live demo:** _add your Vercel URL here_ &nbsp;·&nbsp; **Guided tour** built in — no setup required.

![Open Graph preview](frontend/src/app/opengraph-image.tsx)

---

## What it does

A sentence enters a transformer as a handful of points. As it passes through the
layers, each token moves, bends, and clusters — picking up context from its
neighbours. This tool extracts that hidden motion (the *residual stream*) and
renders it as something you can orbit, scrub, and explore in your browser.

Five things to explore:

| Mode | Question it answers |
|------|---------------------|
| **Trajectories** | How does a word's representation evolve through the layers? (+ attention edges) |
| **Anchor view** | What if we describe each word by its *distance to reference concepts* instead of raw coordinates? |
| **Cross-model compare** | Do *different models* build the same internal geometry? (CKA + Procrustes) |
| **Concept probe** | Drop dozens of concepts into the space — do they self-organize by meaning? |
| **Corpus topology** | Does philosophy have a different *topological shape* than poetry or physics? (persistent homology) |
| **Training time-lapse** | Watch concept geometry crystallize across a model's training run (300B tokens) |

The research question behind it: the **Platonic Representation Hypothesis** — the
idea that models trained differently converge on the same underlying geometry of
meaning.

## Experiments

Small, honest studies run with the tool's own backend — full writeups in
[docs/experiments.md](docs/experiments.md):

- **Where does a transformer disambiguate a word?** Pythia-70M splits the two
  senses of *bank* in the middle of the stack; GPT-2 waits until its final layer.
- **Can activation topology detect AI text?** A cautionary tale — a
  perfect-looking 100% separation that dissolves, under layer/metric sweeps and a
  homology-dimension decomposition, into a single era-confounded scalar. A study
  that fails the right way.

## How it's deployed (and why it always works)

A portfolio demo must never break when someone clicks the link. So the site ships
in two halves:

- **A precomputed gallery** (`frontend/public/gallery/`) — curated examples
  captured as static JSON. The deployed site replays them through the same state
  layer the live app uses, so it's **fully interactive with no backend running**:
  orbit the scene, scrub layers, toggle attention. This is the bulletproof core.
- **An optional live ML backend** — for visitors who want to run *their own* text.
  It's a containerized FastAPI service (PyTorch + TransformerLens) on a free-tier
  ML host. If it's asleep, the UI offers a one-click "wake (~30s)" and the gallery
  keeps working in the meantime.

```
Frontend (Next.js + R3F)  ──static──▶  Vercel  (always on, instant)
        │ custom input
        ▼
Backend (FastAPI + PyTorch) ─docker─▶  Hugging Face Spaces / Render (wakes on demand)
```

## Tech stack

**Backend** — FastAPI · PyTorch · TransformerLens · scikit-learn · UMAP · ripser
(persistent homology)
**Frontend** — Next.js 14 · React Three Fiber · drei · zustand · TypeScript
**Models** — Pythia-70M (default), GPT-2, Pythia-160M

## Local development

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
# first run downloads Pythia-70M (~160MB)
```

### Frontend

```bash
cd frontend
npm install
npm run dev    # → http://localhost:3000
```

The frontend reads the backend URL from `NEXT_PUBLIC_API_URL` (defaults to
`http://localhost:8000`).

## Regenerating the gallery

With the backend running, capture fresh showcase examples:

```bash
cd backend && python scripts/build_gallery.py
```

This writes JSON + a manifest into `frontend/public/gallery/`.

## Deploying

- **Frontend → Vercel.** Import the repo, set **Root Directory** to `frontend`,
  and set env vars `NEXT_PUBLIC_API_URL` (your backend URL),
  `NEXT_PUBLIC_SITE_URL` (your Vercel URL), and `NEXT_PUBLIC_GITHUB_URL`.
- **Backend → Hugging Face Spaces** (Docker SDK) or Render. `backend/Dockerfile`
  is CPU-only and pre-caches the model. Set `SGE_ALLOWED_ORIGINS` to your Vercel
  origin if you use a custom domain (any `*.vercel.app` origin is allowed by
  default).

## Project layout

```
backend/
  app/            # FastAPI endpoints + extraction / projection / topology
  scripts/        # build_gallery.py — precompute showcase data
  Dockerfile
frontend/
  src/app/        # Next.js app, layout, OG image
  src/components/ # Viewport (R3F), Controls, Hero, Tour, AboutPanel, ...
  src/lib/        # api client, gallery loader, config
  src/store/      # zustand stores (explorer, corpus, song, ui)
  public/gallery/ # precomputed showcase JSON
```
