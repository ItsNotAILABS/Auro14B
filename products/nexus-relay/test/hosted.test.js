import test from "node:test";
import assert from "node:assert/strict";
import { bearerToken, sha256 } from "../src/auth.js";
import { normalizeCsv, normalizeSitemap, normalizeText } from "../src/extract.js";
import { NEXUS_RELAY_SKILL } from "../src/skill.js";
import { detectMode } from "../src/index.js";

const headers = new Headers({ "content-type": "text/plain" });

test("bearer authentication parser accepts hosted key header", () => {
  const request = new Request("https://relay.test/v1/read", { headers: { authorization: "Bearer nr_live_secret" } });
  assert.equal(bearerToken(request), "nr_live_secret");
});

test("API keys can be deterministically hashed without storing plaintext", async () => {
  assert.equal(await sha256("nr_live_secret"), await sha256("nr_live_secret"));
  assert.notEqual(await sha256("nr_live_secret"), await sha256("nr_live_other"));
});

test("CSV normalizer produces named records", () => {
  const result = normalizeCsv({ text: 'name,price\n"Desk, Pro",99\nChair,49', url: "https://e/data.csv", fetchedAt: "now", status: 200, headers });
  assert.equal(result.kind, "csv");
  assert.equal(result.content.row_count, 2);
  assert.equal(result.content.rows[0].name, "Desk, Pro");
});

test("sitemap normalizer extracts URLs and timestamps", () => {
  const xml = "<urlset><url><loc>https://e/a</loc><lastmod>2026-07-22</lastmod></url></urlset>";
  const result = normalizeSitemap({ xml, url: "https://e/sitemap.xml", fetchedAt: "now", status: 200, headers });
  assert.equal(result.kind, "sitemap");
  assert.equal(result.content.urls[0].url, "https://e/a");
});

test("markdown and text normalizer preserves source text", () => {
  const result = normalizeText({ text: "# Title\nBody", kind: "markdown", url: "https://e/readme.md", fetchedAt: "now", status: 200, headers });
  assert.equal(result.kind, "markdown");
  assert.equal(result.content.text_length, 12);
});

test("automatic mode detection covers expanded source types", () => {
  assert.equal(detectMode("text/csv", "a,b", "auto", "/data.csv"), "csv");
  assert.equal(detectMode("text/plain", "# Readme", "auto", "/README.md"), "markdown");
  assert.equal(detectMode("application/xml", "<urlset/>", "auto", "/sitemap.xml"), "sitemap");
  assert.equal(detectMode("text/plain", "hello", "auto", "/file.txt"), "text");
});

test("downloadable skill describes hosted key and metered endpoints", () => {
  assert.match(NEXUS_RELAY_SKILL, /NEXUS_RELAY_API_KEY/);
  assert.match(NEXUS_RELAY_SKILL, /\/v1\/usage/);
  assert.match(NEXUS_RELAY_SKILL, /\/mcp/);
});
