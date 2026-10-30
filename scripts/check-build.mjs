import assert from 'node:assert/strict';
import { readFile, readdir, stat } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = new URL('../dist/', import.meta.url);
async function walk(dir) {
  return (await Promise.all((await readdir(dir)).map(async name => {
    const path = join(dir, name);
    return (await stat(path)).isDirectory() ? walk(path) : [path];
  }))).flat();
}
const files = await walk(fileURLToPath(root));
const referencedAssets = new Set();
const htmlFiles = files.filter(path => path.endsWith('.html'));
const vit = await readFile(new URL('projects/vision-transformer/index.html', root), 'utf8');
assert.match(vit, /astro-code/,'Code must have syntax highlighting');
const captioning = await readFile(new URL('projects/image-captioning/index.html', root), 'utf8');
const smolvla = await readFile(new URL('projects/smolvla-training-strategies/index.html', root), 'utf8');
assert.equal((smolvla.match(/<figure\b/g) || []).length, 13, 'SmolVLA must include the animated hook and all 12 figures');
assert.match(smolvla, /class="katex/,'SmolVLA equations must be rendered');
assert.match(smolvla, /href="https:\/\/github.com\/NickS0kolov\/lerobot\/tree\/feat\/smolvla-alignment">Code on GitHub<\/a>/);
assert.doesNotMatch(smolvla, /Code on Kaggle|href="undefined"/);
assert.match(vit, /Code on Kaggle/, 'Existing notebook links must remain available');
assert.equal((smolvla.match(/<video\b/g) || []).length, 1, 'Task demos must share a single player');
assert.deepEqual([...smolvla.matchAll(/data-task="(\d)"/g)].map(match => Number(match[1])), Array.from({length: 10}, (_, i) => i), 'Videos must follow LIBERO task order');
assert.equal((vit.match(/<figure>/g)||[]).length, 8, 'All ViT figures must be present');
assert.equal((captioning.match(/<figure>/g)||[]).length, 14, 'Captioning figures must exclude the incorrectly reused training plot');
assert.doesNotMatch(vit + captioning, /Research overview\.|Read the complete article|<img[^>]+src="https?:/);
const articleCount = (await readdir(new URL('../src/pages/projects/', import.meta.url))).filter(file => file.endsWith('.md')).length;
assert.equal(htmlFiles.length, 3 + articleCount, 'Expected all sections, articles, and the 404 page');
for (const file of htmlFiles) {
  const html = await readFile(file, 'utf8');
  assert.match(html, /<html lang="en">/);
  assert.doesNotMatch(html, /href="\/(?:about|publications)\/"|class="(?:lower-grid|notebook-index|article-notice|article-cover)"/);
  for (const [, asset] of html.matchAll(/(?:href|src)="(\/[^"#?]*)"/g)) {
    referencedAssets.add(asset);
    const target = new URL(asset.slice(1), root);
    if (asset.endsWith('/')) target.pathname += 'index.html';
    assert.ok(await stat(target).catch(() => false), `${file}: missing ${asset}`);
  }
}
for (const route of ['about/', 'publications/']) {
  assert.equal(await stat(new URL(route, root)).catch(() => null), null, `Deleted route remains: ${route}`);
}
const publicRoot = new URL('../public/', import.meta.url);
const publicFiles = await walk(fileURLToPath(publicRoot));
for (const file of publicFiles) {
  const asset = '/' + file.slice(fileURLToPath(publicRoot).length).replaceAll('\\', '/');
  assert.ok(referencedAssets.has(asset), `Unused public asset: ${asset}`);
}
console.log(`Verified ${htmlFiles.length} pages, local links, and ${publicFiles.length} used public assets. Deleted pages are absent.`);
