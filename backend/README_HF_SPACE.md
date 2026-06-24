# Hugging Face Space config for the backend

When you create the Docker Space, its repo needs a `README.md` at the root whose
YAML front-matter tells HF how to run the container. Copy the block below into
that `README.md` (the Space's, not this repo's).

```markdown
---
title: ConceptShape Backend
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: Semantic Geometry Explorer ML API (FastAPI + TransformerLens)
---

# ConceptShape Backend

FastAPI service that extracts transformer residual-stream geometry for the
[Semantic Geometry Explorer](https://github.com/finnpounds/conceptshape).
Built from the `Dockerfile` in this Space. Endpoints: `/analyze`,
`/anchor-analyze`, `/compare`, `/probe`, `/text-shape`, `/compare-shapes`,
`/health`.
```

## Deploy steps

1. Create a new **Docker** Space at https://huggingface.co/new-space
   (name it `conceptshape-backend`, hardware **CPU basic — free**).
2. Push the *contents of this `backend/` folder* to the Space repo root:

   ```bash
   huggingface-cli login                       # paste a write token
   git clone https://huggingface.co/spaces/<your-hf-username>/conceptshape-backend hf-space
   cp -R backend/Dockerfile backend/app backend/scripts backend/requirements.txt hf-space/
   # create hf-space/README.md with the YAML block above
   cd hf-space && git add -A && git commit -m "Deploy ConceptShape backend" && git push
   ```

3. The Space builds the Docker image and serves at
   `https://<your-hf-username>-conceptshape-backend.hf.space`.
4. In the Space **Settings → Variables**, add
   `SGE_ALLOWED_ORIGINS = https://conceptshape.vercel.app`.
5. Put that Space URL into Vercel as `NEXT_PUBLIC_API_URL` and redeploy.
