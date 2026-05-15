# Documentation site and GitHub Pages

The published guide at **[kaushikdey647.github.io/shunya](https://kaushikdey647.github.io/shunya/)** is built with **[MkDocs](https://www.mkdocs.org/)** and the **[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)** theme. The **left navigation sidebar**, search, and styling come from that build — not from raw Markdown in the repository.

## Build locally

From the repository root (same commands as [Install → Docs site locally](../install.md)):

```bash
uv sync --group dev --group docs
uv run mkdocs serve
```

Open the URL MkDocs prints (often **`http://127.0.0.1:8000`**). To match CI:

```bash
uv run mkdocs build --strict
```

The static output is written to **`site/`** (gitignored). Configuration and nav live in **`mkdocs.yml`**; Markdown sources live under **`docs/`**.

## Publishing on GitHub Pages (keep the sidebar)

The repository ships a workflow **Deploy documentation** (`.github/workflows/docs.yml`) that:

1. Runs **`uv run mkdocs build --strict`**
2. Adds **`site/.nojekyll`** so GitHub Pages does not run **Jekyll** on the uploaded artifact
3. Uploads the **`site/`** folder as the Pages artifact

**Repository → Settings → Pages → Build and deployment → Source** must be **GitHub Actions**, not **Deploy from a branch**.

If Source is set to **Deploy from a branch** and the folder **`/ (root)`** or **`/docs`**, GitHub serves Markdown/HTML through **Jekyll**. That produces a **plain, mostly unstyled page with no Material sidebar** — it is not the MkDocs site, even though the Markdown files look similar.

**Fix:** set Pages **Source** to **GitHub Actions**, push a change that triggers the workflow (or use **Actions → Deploy documentation → Run workflow**), wait for a green run, then hard-refresh the site (cache can hold an old broken layout briefly).

## After editing `docs/` or `mkdocs.yml`

- Keep **`mkdocs.yml` → `nav:`** in sync with new pages you expect in the sidebar; orphan pages are still built but do not appear in the nav unless you add them.
- Run **`uv run mkdocs build --strict`** before pushing; strict mode fails on warnings that would break the published site.

## Related

- [Install](../install.md) — docs groups and local serve
- [HTTP API](../http-api.md) — API outline (separate from this static site)
