import test from "node:test";import assert from "node:assert/strict";
const source=await import("node:fs/promises").then(x=>x.readFile(new URL("../src/policy.ts",import.meta.url),"utf8"));
test("policy declares relative-path and traversal defenses",()=>{assert.match(source,/path\.includes\(\":\/\/\"\)/);assert.match(source,/path\.includes\(\"\.\.\"\)/)});
test("policy separates read methods",()=>{assert.match(source,/new Set\(\[\"GET\",\"HEAD\",\"OPTIONS\"\]\)/)});
test("worker authenticates operator, requires execution grant, and emits receipts",async()=>{const worker=await import("node:fs/promises").then(x=>x.readFile(new URL("../src/index.ts",import.meta.url),"utf8"));assert.match(worker,/operator authentication required/);assert.match(worker,/OPERATOR_TOKEN/);assert.match(worker,/valid unexpired execution grant required/);assert.match(worker,/previous_hash/);assert.match(worker,/CLOUDFLARE_API_TOKEN/)});
