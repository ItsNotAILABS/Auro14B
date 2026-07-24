import { cp, mkdir, rm, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { readdir, readFile, stat } from "node:fs/promises";
import path from "node:path";

const root = new URL("..", import.meta.url).pathname;
const site = path.join(root, "site");
const out = path.join(root, "dist", "site");
await rm(out, { recursive: true, force: true });
await mkdir(out, { recursive: true });
await cp(site, out, { recursive: true });

async function files(dir) {
  const result = [];
  for (const name of await readdir(dir)) {
    const full = path.join(dir, name);
    const info = await stat(full);
    if (info.isDirectory()) result.push(...await files(full));
    else result.push(full);
  }
  return result;
}

const manifest = [];
for (const file of await files(out)) {
  const body = await readFile(file);
  manifest.push({
    path: path.relative(out, file).replaceAll(path.sep, "/"),
    bytes: body.length,
    sha256: createHash("sha256").update(body).digest("hex")
  });
}
await writeFile(path.join(out, "release-manifest.json"), JSON.stringify({ schema: "auro14b.static-release.v1", generated_at: new Date().toISOString(), files: manifest }, null, 2));
console.log(JSON.stringify({ ok: true, output: out, files: manifest.length }));
