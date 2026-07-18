/**
 * Secure RPC helpers — keys only from process.env (server).
 * Prefer ethers + viem dual-path for HIM agentic tooling.
 */
import { createPublicClient, http, formatEther } from "viem";
import { mainnet, sepolia } from "viem/chains";
import { ethers } from "ethers";

function chainFromId(id) {
  const n = Number(id || process.env.CHAIN_ID || 1);
  if (n === 11155111) return sepolia;
  return mainnet;
}

export function getRpcUrl() {
  const url =
    process.env.ETH_RPC_URL ||
    process.env.ALCHEMY_URL ||
    process.env.INFURA_URL ||
    "";
  return url.trim();
}

export function hasRpc() {
  return Boolean(getRpcUrl());
}

export function getViemClient() {
  const url = getRpcUrl();
  if (!url) {
    throw new Error(
      "ETH_RPC_URL not configured. Set him-web3/.env from .env.example (Alchemy/Infura)."
    );
  }
  return createPublicClient({
    chain: chainFromId(process.env.CHAIN_ID),
    transport: http(url),
  });
}

export function getEthersProvider() {
  const url = getRpcUrl();
  if (!url) {
    throw new Error("ETH_RPC_URL not configured on server");
  }
  return new ethers.JsonRpcProvider(url);
}

export async function getBlockNumber() {
  const client = getViemClient();
  const n = await client.getBlockNumber();
  return n.toString();
}

export async function getBalance(address) {
  const client = getViemClient();
  const bal = await client.getBalance({ address });
  return {
    address,
    wei: bal.toString(),
    eth: formatEther(bal),
  };
}

export async function getBlock(blockTag = "latest") {
  const provider = getEthersProvider();
  const block = await provider.getBlock(blockTag);
  if (!block) return null;
  return {
    number: block.number,
    hash: block.hash,
    timestamp: block.timestamp,
    transactions: block.transactions?.length ?? 0,
  };
}

export async function callContract({ address, abi, functionName, args = [] }) {
  const provider = getEthersProvider();
  const contract = new ethers.Contract(address, abi, provider);
  if (typeof contract[functionName] !== "function") {
    throw new Error(`function ${functionName} not on ABI`);
  }
  const result = await contract[functionName](...args);
  // normalize BigInt
  return JSON.parse(
    JSON.stringify(result, (_, v) => (typeof v === "bigint" ? v.toString() : v))
  );
}

export function healthPayload() {
  return {
    ok: true,
    service: "him-web3",
    name: "HIM",
    rpc_configured: hasRpc(),
    chain_id: Number(process.env.CHAIN_ID || 1),
    secrets: "server-only (.env) — never sent to React",
    libraries: ["ethers", "viem", "express"],
  };
}
