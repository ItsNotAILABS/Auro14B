import http from 'node:http';
import { randomUUID } from 'node:crypto';
import { createPlatformClient } from './platform-client.mjs';
import { createReceiptChain } from './receipt-chain.mjs';
import { createScheduler } from './scheduler.mjs';
import { describeModelFamily } from './model-family.mjs';

const port = Number.parseInt(process.env.PORT ?? '8787', 10);
const host = process.env.HOST ?? '0.0.0.0';
const serviceName = process.env.SERVICE_NAME ?? 'medina-node-fabric';
const startedAt = new Date().toISOString();

const receipts = createReceiptChain();
const platform = createPlatformClient({