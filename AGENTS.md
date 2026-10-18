# Working in this repository

- Plan changes as small, testable milestones; keep commits focused.
- Use `uv` for dependencies and commands. Do not add `requirements.txt` as a second source of truth.
- Keep domain logic in `src/smartcampus_ai`; keep `app.py` limited to Streamlit presentation.
- Keep deterministic safety/business rules before retrieval or LLM calls.
- Never treat bundled procedures as real university policy or expose secrets/student data.
- Run `uv run ruff check .` and `uv run pytest` before committing relevant changes.
- Update README or `docs/` whenever setup, architecture, behavior, or commands change.
- Do not commit generated models, evaluation plots, local secrets, caches, or virtual environments.
