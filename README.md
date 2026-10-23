# w3iz3n 的博客

独立 Hugo 博客，支持 Markdown、LaTeX、文章搜索、RSS、响应式布局与深浅色切换。

## 本地预览

使用 Hugo 0.168.0：

```powershell
hugo server -D
```

Windows 下本次下载的构建工具也可以使用 `.\.tools\hugo.exe server -D`。

## 发布文章

在 `content/posts/` 下创建 Markdown 文件，例如：

```toml
+++
title = '文章标题'
date = '2026-10-23T12:00:00+08:00'
draft = false
+++
```

正文写在第二个 `+++` 之后。`draft = true` 的文章不会公开；未来日期的文章到达日期并重新构建后才发布。

行内公式使用 `$a_i^2$` 或 `\(a_i^2\)`；独立公式使用 `$$ ... $$` 或 `\[ ... \]`。矩阵、多行公式和常用 LaTeX 命令通过 MathJax 渲染。代码块不会作为公式处理。正文里的字面美元符号可写为 `\$`。

更新后提交并推送到 `main`，GitHub Actions 会自动构建并部署到 https://w3iz3n.github.io/ 。

```powershell
git add content
git commit -m "Update article"
git push
```

笔记放在 `content/docs/`，样式在 `static/style.css`，页面模板在 `layouts/`。

旧工作区中未发布的 Conclusion2025 和 Way to GLP 已迁移为草稿，确认后可将 `draft` 改为 `false`。
