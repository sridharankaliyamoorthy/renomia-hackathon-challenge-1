# MrRooT.Ai — Renomia Challenge 1 (documentation site)

This folder is a [Docusaurus](https://docusaurus.io/) site that documents the insurance offer comparison solution in the repository root.

## Commands

```bash
npm install
npm start      # http://localhost:3000
npm run build  # static output in build/
```

## GitHub Pages

1. Set `url` and `baseUrl` in `docusaurus.config.ts` (for a project site, `baseUrl` is usually `/repository-name/`).
2. Configure deployment (e.g. GitHub Action `peaceiris/actions-gh-pages` or `docusaurus deploy`) to publish the `build/` folder.

## Source

Documentation markdown lives in `docs/`. Edit pages on GitHub via the “Edit this page” link when deployed.
