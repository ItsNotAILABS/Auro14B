import test from "node:test";
import assert from "node:assert/strict";
import { decodeEntities, extractJsonLd, extractMeta, normalizeFeed, stripHtml } from "../src/extract.js";
import { detectMode, isPrivateHost, validateUrl } from "../src/index.js";
import { handleMcp } from "../src/mcp.js";

test("html extraction strips active content and preserves readable text", () => {
  const html = '<title>Hello</title><script>alert(1)</script><p>A &amp; B</p>';
  assert.equal(stripHtml(html), "Hello A & B");
  assert.equal(extractMeta(html, "https://example.com").title, "Hello");
});

test("json-ld extraction is bounded and tolerant", () => {
  const html = '<script type="application/ld+json">{"@type":"Article","name":"X"}</script>';
  assert.equal(extractJsonLd(html)[0].name, "X");
});

test("feed normalization produces common records", () => {
  const xml = '<rss><channel><title>News</title><item><guid>1</guid><title>A</title><link>https://e/a</link><description>Body</description></item></channel></rss>';
  const result = normalizeFeed({ xml, url: "https://e/feed", fetchedAt: "now", status: 200, headers: new Headers({"content-type":"application/rss+xml"}) });
  assert.equal(result.content.item_count, 1);
  assert.equal(result.content.items[0].url, "https://e/a");
});

test("private and loopback targets are blocked", () => {
  for (const host of ["localhost", "127.0.0.1", "10.2.3.4", "192.168.1.2", "172.16.0.1", "172.31.1.1"]) assert.equal(isPrivateHost(host), true);
  assert.throws(() => validateUrl("http://localhost/private"));
  assert.equal(validateUrl("https://example.com/a#b").href, "https://example.com/a");
});

test("mode detection supports json feeds and html", () => {
  assert.equal(detectMode("application/json", "{}", "auto"), "json");
  assert.equal(detectMode("application/xml", "<rss/>", "auto"), "feed");
  assert.equal(detectMode("text/html", "<html/>", "auto"), "html");
});

test("MCP exposes relay_read and returns structured content", async () => {
  const init = await handleMcp(new Request("https://x/mcp", { method: "POST", body: JSON.stringify({jsonrpc:"2.0",id:1,method:"initialize"}) }), async () => ({}));
  assert.equal(init.result.serverInfo.name, "nexus-relay");
  const listed = await handleMcp(new Request("https://x/mcp", { method: "POST", body: JSON.stringify({jsonrpc:"2.0",id:2,method:"tools/list"}) }), async () => ({}));
  assert.equal(listed.result.tools[0].name, "relay_read");
  const called = await handleMcp(new Request("https://x/mcp", { method: "POST", body: JSON.stringify({jsonrpc:"2.0",id:3,method:"tools/call",params:{name:"relay_read",arguments:{url:"https://example.com"}}}) }), async () => ({ok:true,kind:"document"}));
  assert.equal(called.result.structuredContent.kind, "document");
});

test("entity decoder handles named and numeric entities", () => {
  assert.equal(decodeEntities("A&amp;B &#33; &#x3f;"), "A&B ! ?");
});
