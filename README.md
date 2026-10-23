# Nick Sokolov — AI Research

Static personal research site built with Astro and Markdown. Palette: `#fbf9f5`, `#dcdcdc`, `#b40303`, `#000000`.

## Local preview

Requires Node.js 22.12 or newer.

```sh
npm ci
npm run dev
```

## Build and verify

```sh
npm run build
npm test
npm run preview
```

## Website and source

- Website: [nicks0kolov.github.io](https://nicks0kolov.github.io/)
- Source code: [NickS0kolov/nicks0kolov.github.io](https://github.com/NickS0kolov/nicks0kolov.github.io)
- Deployment status: [GitHub Actions](https://github.com/NickS0kolov/nicks0kolov.github.io/actions)

The workflow in `.github/workflows/deploy.yml` builds, checks, and publishes every push to `main`. GitHub Pages uses **GitHub Actions** as its deployment source.

The canonical address is configured as `https://nicks0kolov.github.io`. It is a user site, so no repository subpath is needed.

Deployment follows the [official Astro GitHub Pages guide](https://docs.astro.build/en/guides/deploy/github/).

To update the site, edit the source and commit to `main`. Keep `node_modules/`, `dist/`, and `.astro/` out of Git; the workflow installs dependencies from the lockfile and generates the site.

## Add an article

Create a Markdown file in `src/pages/projects/`. Use an existing file as the frontmatter template. The entry appears automatically on Home and Projects. Set `layout: ../../layouts/Article.astro`; provide `title`, `subtitle`, `description`, `category`, `kind`, `date`, `image`, `imageAlt`, `source`, and `notebook`.

Store images in `public/images/`; reference them as `/images/filename.png`. Add article text below the frontmatter. Mathematical notation uses `$...$` inline and `$$...$$` for display equations. Fenced code blocks support syntax highlighting and a copy button.

The two entries contain the complete published articles imported from the supplied Medium account export. All 23 article figures are saved in `public/images/image-captioning/` and `public/images/vision-transformer/`. Text, code, figure captions, and tables are preserved. Links between these articles point to the local site; Kaggle implementations remain external. Drafts, comments, and other account data are not included.

Original sources:

- [From Pixels to Words](https://medium.com/@sokolovnick/from-pixels-to-words-how-machines-learned-to-describe-the-world-v-1-88055eef087d)
- [Vision Transformer from scratch](https://medium.com/@sokolovnick/visual-transformer-from-scratch-ee636010c7bb)

The hero uses a custom ImageGen illustration of a robotic arm and a red motion trajectory. It is a conceptual illustration, not a recorded experiment. Article cover images are from the linked Medium publications. Google Fonts is optional: the stylesheet includes system fallbacks.
