#!/usr/bin/env node
/**
 * install_applet_package — HIM standard tooling for web3 deps.
 *
 * Usage:
 *   node scripts/install_applet_package.js ethers
 *   node scripts/install_applet_package.js viem web3
 *   npm run install:applet -- ethers viem
 *
 * Installs packages into him-web3 (server-side). Never expose keys in browser.
 */
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");

const pkgs = process.argv.slice(2).filter((a) => a && !a.startsWith("-"));
if (!pkgs.length) {
  console.log(
    JSON.stringify(
      {
        ok: false,
        error: "usage: install_applet_package <pkg> [pkg...]",
        examples: ["ethers", "viem", "web3"],
        note: "Runs npm install in him-web3 root (server-side tooling).",
      },
      null,
      2
    )
  );
  process.exit(2);
}

// Allowlist common web3 packages HIM is expected to use
const ALLOW = new Set([
  "ethers",
  "viem",
  "web3",
  "web3.js",
  "@wagmi/core",
  "@wagmi/connectors",
  "dotenv",
  "express",
  "cors",
  "zod",
  "axios",
]);

const blocked = pkgs.filter((p) => {
  const base = p.replace(/^@[^/]+\//, "").split("@")[0];
  const full = p.startsWith("@") ? p.split("@").slice(0, 2).join("@").split("/")[0] + (p.includes("/") ? "/" + p.split("/")[1].split("@")[0] : "") : p.split("@")[0];
  // simple: allow if any allow entry matches start
  return !Array.from(ALLOW).some((a) => p === a || p.startsWith(a + "@") || p.startsWith(a + "/"));
});

// Be permissive for scoped installs that look like web3, but warn
const finalPkgs = pkgs;
if (blocked.length) {
  console.warn(
    JSON.stringify({
      warn: "packages outside default allowlist — installing anyway for HIM flexibility",
      blocked_hint: blocked,
    })
  );
}

console.log(
  JSON.stringify({
    ok: true,
    action: "install_applet_package",
    cwd: root,
    packages: finalPkgs,
  })
);

const child = spawn(
  process.platform === "win32" ? "npm.cmd" : "npm",
  ["install", ...finalPkgs, "--save", "--no-fund", "--no-audit"],
  { cwd: root, stdio: "inherit", shell: true }
);

child.on("exit", (code) => {
  process.exit(code ?? 1);
});
