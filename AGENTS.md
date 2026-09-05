# Repository instructions

- In notebooks, display plots inline by default. Do not save figures as PNG, PDF, SVG, or any other image/file format unless the user explicitly asks for saved figure files.
- Keep notebook Markdown minimal and paper-clean. A concise title or necessary section heading is fine, but do not add speculative interpretations, narrative explanations, result claims, or instructional prose unless the user explicitly requests that text.
- Keep the plot stsyle coherent, using "Editorial" colours from the common theme of the repo
- When new feature is added to training or inference or what ever, don't automateically opt in to do it as an conditioned on option. Just implement it as is, unless it's already part of something conditional or you think it's really critical, at that point just ask me.
- Don't run the notebooks and code urself unless I ask you to. You can check the integrity and if it compiles, just not the actual
- For work on `07b_mixed_reid_depablo_lj_context_ablation.ipynb`, read and update `notebooks/latent_space/07b_experiment_log.md`. Record completed experiments with their exact recipe and source-wise held-out rollout results; do not rely on pooled metrics alone.
