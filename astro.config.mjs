import { defineConfig } from 'astro/config';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

export default defineConfig({
  site: 'https://nicks0kolov.github.io',
  trailingSlash: 'always',
  devToolbar: { enabled: false },
  markdown: {
    smartypants: false,
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
    shikiConfig: { theme: 'github-light' },
  },
});
