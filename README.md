# w3iz3n's Blog

A standalone Hugo blog with Markdown, LaTeX mathematics, article search, RSS, responsive layouts, and light and dark themes.

## Local preview

Use Hugo 0.168.0:

```powershell
hugo server -D
```

On this Windows checkout, the downloaded tool is also available as `.\.tools\hugo.exe server -D`.

## Publish an article

Create a Markdown file in `content/posts/` with front matter like this:

```toml
+++
title = 'Article title'
date = '2026-10-23T12:00:00+08:00'
draft = false
+++
```

Write the article after the closing `+++`. Articles with `draft = true` are excluded from the published site. Future-dated articles become eligible after their date and require another build to publish.

Use `$a_i^2$` or `\(a_i^2\)` for inline equations, and `$$ ... $$` or `\[ ... \]` for display equations. MathJax renders matrices, aligned equations, and common LaTeX commands. Code blocks are excluded from math rendering. Escape a literal dollar sign with `\$`.

Commit and push to `main` to trigger automatic deployment to https://w3iz3n.github.io/ through GitHub Actions:

```powershell
git add content
git commit -m "Update article"
git push
```

Study notes live in `content/docs/`, styles in `static/style.css`, and templates in `layouts/`.

Imported unpublished articles retain their draft status. Set `draft = false` when they are ready to publish. Existing article paths are preserved so that old links continue to work.
