import test from "node:test";
import assert from "node:assert/strict";
import { SignalLensRelayClient } from "../clients/signallens.mjs";

test("SignalLens preserves Relay receipts and never falls back to direct fetch", async () => {
  const calls = [];
  const client = new SignalLensRelayClient({
    baseUrl: "https://relay.test",
    apiKey: "nr_test_key",
    fetchFn: async (url, options) => {
      calls.push({ url, options });
      return new Response(JSON.stringify({
        ok: true,
        source: { url: "https://example.com", status: 200 },
        content: { text: "hello" },
        receipt: { content_sha256: "a".repeat(64), final_url: "https://example.com" },
        metering: { event_id: "evt_1" }
      }), { status: 200, headers: { "content-type": "application/json" } });
    }
  });
  const result = await client.read("https://example.com");
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, "https://relay.test/v1/read");
  assert.equal(result.relay_metering.event_id, "evt_1");
  assert.equal(result.perception_sha256.length, 64);
});

test("SignalLens fails closed when Relay denies retrieval", async () => {
  const client = new SignalLensRelayClient({
    baseUrl: "https://relay.test",
    apiKey: "nr_test_key",
    fetchFn: async () => new Response(JSON.stringify({ ok: false, error: "quota exceeded" }), { status: 429 })
  });
  await assert.rejects(() => client.read("https://example.com"), /quota exceeded/);
});
