import test from "node:test";
import assert from "node:assert/strict";
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { PersistentNonceStore, authenticateRequest } from "../egress/server.mjs";

function signedHeaders(secret, timestamp, nonce, target) {
  return {
    "x-relay-timestamp": String(timestamp),
    "x-relay-nonce": nonce,
    "x-relay-signature": crypto.createHmac("sha256", secret).update(`${timestamp}.${nonce}.${target}`).digest("hex")
  };
}

test("signed egress request can be claimed only once inside freshness window", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "relay-nonce-"));
  const storePath = path.join(root, "nonces.jsonl");
  const store = new PersistentNonceStore(storePath, 60_000);
  const now = 1_800_000_000_000;
  const target = "https://example.com/source";
  const nonce = "nonce_0123456789abcdefghijkl";
  const headers = signedHeaders("secret", now, nonce, target);

  const accepted = authenticateRequest(headers, target, { secret: "secret", nonceStore: store, now });
  assert.equal(accepted.nonce, nonce);
  assert.throws(
    () => authenticateRequest(headers, target, { secret: "secret", nonceStore: store, now: now + 1 }),
    /replayed egress request/
  );
});

test("nonce replay remains rejected after store restart", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "relay-nonce-restart-"));
  const storePath = path.join(root, "nonces.jsonl");
  const now = 1_800_000_000_000;
  const target = "https://example.com/source";
  const nonce = "nonce_abcdefghijklmnopqrstuvwxyz";
  const headers = signedHeaders("secret", now, nonce, target);

  authenticateRequest(headers, target, {
    secret: "secret",
    nonceStore: new PersistentNonceStore(storePath, 60_000),
    now
  });
  assert.throws(
    () => authenticateRequest(headers, target, {
      secret: "secret",
      nonceStore: new PersistentNonceStore(storePath, 60_000),
      now: now + 10
    }),
    /replayed egress request/
  );
});

test("stale requests and signatures that omit the nonce are rejected", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "relay-nonce-stale-"));
  const now = 1_800_000_000_000;
  const target = "https://example.com/source";
  const nonce = "nonce_0123456789abcdefghijkl";
  const stale = now - 60_001;
  assert.throws(
    () => authenticateRequest(signedHeaders("secret", stale, nonce, target), target, {
      secret: "secret",
      nonceStore: new PersistentNonceStore(path.join(root, "stale.jsonl"), 60_000),
      now
    }),
    /stale egress request/
  );

  const badHeaders = {
    "x-relay-timestamp": String(now),
    "x-relay-nonce": nonce,
    "x-relay-signature": crypto.createHmac("sha256", "secret").update(`${now}.${target}`).digest("hex")
  };
  assert.throws(
    () => authenticateRequest(badHeaders, target, {
      secret: "secret",
      nonceStore: new PersistentNonceStore(path.join(root, "bad.jsonl"), 60_000),
      now
    }),
    /invalid egress signature/
  );
});
