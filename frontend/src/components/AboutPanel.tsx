"use client";

import { useUIStore } from "@/store/ui";
import { GITHUB_URL } from "@/lib/config";

/** Modal explaining what the project is, how it works, and the stack. */
export default function AboutPanel() {
  const { showAbout, setShowAbout } = useUIStore();
  if (!showAbout) return null;

  return (
    <div className="about-overlay" onClick={() => setShowAbout(false)}>
      <div className="about-modal" onClick={(e) => e.stopPropagation()}>
        <button className="about-close" onClick={() => setShowAbout(false)}>
          ✕
        </button>

        <h2 className="about-title">Semantic Geometry Explorer</h2>
        <p className="about-lead">
          An interactive 3D microscope for the inside of a language model. It
          extracts the hidden activations a transformer produces as it reads
          text, then projects that high-dimensional motion into space you can
          orbit, scrub, and explore.
        </p>

        <h3 className="about-h3">How it works</h3>
        <ol className="about-list">
          <li>
            <b>Extract.</b> Text runs through a transformer (Pythia-70M, GPT-2…)
            via TransformerLens. We capture the residual stream — the model’s
            internal representation — at every layer.
          </li>
          <li>
            <b>Project.</b> Those 512-dimensional vectors are reduced to 3D with
            PCA or UMAP, so each token becomes a moving point with a trail.
          </li>
          <li>
            <b>Render.</b> React Three Fiber draws the trajectories, attention
            edges, and concept clouds in real time in your browser.
          </li>
        </ol>

        <h3 className="about-h3">What you can explore</h3>
        <ul className="about-list">
          <li>
            <b>Trajectories</b> — watch tokens diverge through the layers, with
            attention edges showing which words inform which.
          </li>
          <li>
            <b>Anchor view</b> — re-describe each word by its distance to
            reference concepts, turning an opaque space into a readable map.
          </li>
          <li>
            <b>Logit lens</b> — decode every layer through the model’s
            unembedding to see what it would predict next; scrub the slider and
            watch the guess resolve from noise into grammar.
          </li>
          <li>
            <b>Cross-model compare</b> — do different models build the same
            geometry? CKA &amp; Procrustes metrics put a number on it.
          </li>
          <li>
            <b>Concept probe</b> — embed dozens of concepts and watch them
            self-organize by meaning.
          </li>
          <li>
            <b>Corpus topology</b> — persistent homology gives each text a
            “shape fingerprint”; compare philosophy vs poetry vs physics.
          </li>
          <li>
            <b>Training time-lapse</b> — the same model re-loaded at nine
            checkpoints across its training run: watch the concept map
            crystallize out of noise over 300 billion tokens.
          </li>
        </ul>

        <h3 className="about-h3">The question behind it</h3>
        <p className="about-p">
          It’s a research toy for the <i>Platonic Representation Hypothesis</i>:
          the idea that different models, trained differently, converge on the
          same underlying geometry of meaning. This tool lets you look.
        </p>

        <h3 className="about-h3">Stack</h3>
        <p className="about-stack">
          FastAPI · PyTorch · TransformerLens · scikit-learn · UMAP · ripser
          (persistent homology) &nbsp;·&nbsp; Next.js · React Three Fiber ·
          zustand
        </p>

        <div className="about-footer">
          <a
            className="about-gh"
            href={GITHUB_URL}
            target="_blank"
            rel="noopener noreferrer"
          >
            View source on GitHub →
          </a>
          <span className="about-note">
            Curated examples load instantly. “Run your own text” uses a live ML
            backend that may take ~30s to wake.
          </span>
        </div>
      </div>
    </div>
  );
}
