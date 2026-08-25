const TILE = 8;
const $ = (selector) => document.querySelector(selector);
const query = new URLSearchParams(location.search);
const fragment = new URLSearchParams(location.hash.replace(/^#/, ""));

const bootstrap = {
  coordinator: fragment.get("coordinator") || query.get("coordinator") || sessionStorage.getItem("auro.webgpu.coordinator") || "http://127.0.0.1:8765",
  token: fragment.get("token") || sessionStorage.getItem("auro.webgpu.token") || "",
  workerId: fragment.get("worker_id") || sessionStorage.getItem("auro.webgpu.worker_id") || `browser-${crypto.randomUUID()}`,
  autostart: fragment.get("autostart") === "1" || query.get("autostart") === "1",
};
if (location.hash) history.replaceState(null, "", location.pathname + location.search);

let stopped = true;
let running = false;
let device = null;
let adapter = null;
let adapterInfo = {};
let completedJobs = 0;
let shaderSha256 = "";

$("#coordinator").value = bootstrap.coordinator.replace(/\/$/, "");
$("#token").value = bootstrap.token;
$("#workerName").value = bootstrap.workerId;
$("#worker").textContent = bootstrap.workerId;

function config() {
  return {
    coordinator: $("#coordinator").value.trim().replace(/\/$/, ""),
    token: $("#token").value,
    workerId: $("#workerName").value.trim(),
  };
}

function saveConfig() {
  const value = config();
  if (!/^https?:\/\//.test(value.coordinator)) throw new Error("Coordinator must be an absolute HTTP(S) URL");
  if (!value.workerId || value.workerId.length > 160) throw new Error("Worker name must contain 1..160 characters");
  sessionStorage.setItem("auro.webgpu.coordinator", value.coordinator);
  sessionStorage.setItem("auro.webgpu.token", value.token);
  sessionStorage.setItem("auro.webgpu.worker_id", value.workerId);
  $("#worker").textContent = value.workerId;
  return value;
}

function decode64(value) {
  const raw = atob(value);
  const bytes = new Uint8Array(raw.length);
  for (let index = 0; index < raw.length; index += 1) bytes[index] = raw.charCodeAt(index);
  return new Float32Array(bytes.buffer);
}

function encode64(values) {
  const bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
  let output = "";
  const chunk = 0x8000;
  for (let index = 0; index < bytes.length; index += chunk) {
    output += String.fromCharCode(...bytes.subarray(index, Math.min(bytes.length, index + chunk)));
  }
  return btoa(output);
}

async function sha256Text(value) {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
  return [...new Uint8Array(digest)].map((item) => item.toString(16).padStart(2, "0")).join("");
}

function log(value) {
  const line = `${new Date().toISOString()} ${value}`;
  $("#log").textContent = `${line}\n${$("#log").textContent}`.slice(0, 24000);
}

function setState(value, style = "") {
  const element = $("#state");
  element.textContent = value;
  element.className = style ? `state-${style}` : "";
}

function headers() {
  const value = config();
  const result = { "content-type": "application/json" };
  if (value.token) result["x-auro-cluster-token"] = value.token;
  return result;
}

async function request(path, options = {}) {
  const value = config();
  const response = await fetch(`${value.coordinator}${path}`, {
    ...options,
    headers: { ...headers(), ...(options.headers || {}) },
  });
  const text = await response.text();
  let body = {};
  try { body = text ? JSON.parse(text) : {}; } catch { body = { error: text }; }
  if (!response.ok) throw new Error(`${path} ${response.status}: ${body.error || text}`);
  return body;
}

const shader = `
struct Dims { m:u32, k:u32, n:u32, pad:u32 }
@group(0) @binding(0) var<storage,read> A:array<f32>;
@group(0) @binding(1) var<storage,read> B:array<f32>;
@group(0) @binding(2) var<storage,read_write> C:array<f32>;
@group(0) @binding(3) var<uniform> dims:Dims;
var<workgroup> tileA:array<f32,64>;
var<workgroup> tileB:array<f32,64>;
@compute @workgroup_size(8,8,1)
fn main(@builtin(local_invocation_id) lid:vec3<u32>, @builtin(workgroup_id) wid:vec3<u32>) {
  let row=wid.y*8u+lid.y;
  let col=wid.x*8u+lid.x;
  var sum=0.0;
  let tiles=(dims.k+7u)/8u;
  for(var t=0u; t<tiles; t=t+1u) {
    let aCol=t*8u+lid.x;
    let bRow=t*8u+lid.y;
    let index=lid.y*8u+lid.x;
    var av=0.0;
    var bv=0.0;
    if(row<dims.m && aCol<dims.k) { av=A[row*dims.k+aCol]; }
    if(bRow<dims.k && col<dims.n) { bv=B[bRow*dims.n+col]; }
    tileA[index]=av;
    tileB[index]=bv;
    workgroupBarrier();
    for(var p=0u; p<8u; p=p+1u) {
      sum=sum+tileA[lid.y*8u+p]*tileB[p*8u+lid.x];
    }
    workgroupBarrier();
  }
  if(row<dims.m && col<dims.n) { C[row*dims.n+col]=sum; }
}`;

async function describeAdapter(value) {
  try {
    if (value.info) return { ...value.info };
    if (typeof value.requestAdapterInfo === "function") return { ...(await value.requestAdapterInfo()) };
  } catch (error) {
    log(`adapter info unavailable: ${error.message}`);
  }
  return {};
}

async function ensureDevice() {
  if (device) return device;
  if (!window.isSecureContext) throw new Error("WebGPU requires HTTPS or a loopback secure context");
  if (!navigator.gpu) throw new Error("WebGPU is unavailable in this browser");
  adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("No WebGPU adapter was returned");
  adapterInfo = await describeAdapter(adapter);
  device = await adapter.requestDevice();
  device.lost.then((info) => {
    log(`device lost: ${info.message || info.reason}`);
    device = null;
    setState("device lost", "blocked");
  });
  shaderSha256 = await sha256Text(shader);
  const description = adapterInfo.description || adapterInfo.device || adapterInfo.vendor || "WebGPU adapter ready";
  $("#device").textContent = description;
  return device;
}

function storageBuffer(gpu, data, usage = GPUBufferUsage.STORAGE) {
  const buffer = gpu.createBuffer({
    size: Math.max(4, (data.byteLength + 3) & ~3),
    usage: usage | GPUBufferUsage.COPY_DST,
  });
  gpu.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

function assertJobFitsDevice(job, gpu) {
  const bytes = [job.a.base64.length * 0.75, job.b.base64.length * 0.75, job.output_shape[0] * job.output_shape[1] * 4];
  const limit = Number(gpu.limits.maxStorageBufferBindingSize || Number.MAX_SAFE_INTEGER);
  if (bytes.some((size) => size > limit)) throw new Error(`job exceeds maxStorageBufferBindingSize=${limit}`);
}

async function matmul(a, b, m, k, n, job = null) {
  const gpu = await ensureDevice();
  if (job) assertJobFitsDevice(job, gpu);
  const started = performance.now();
  const module = gpu.createShaderModule({ code: shader });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter((message) => message.type === "error");
  if (errors.length) throw new Error(`WGSL compilation failed: ${errors.map((item) => item.message).join("; ")}`);
  const pipeline = gpu.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
  const aBuffer = storageBuffer(gpu, a);
  const bBuffer = storageBuffer(gpu, b);
  const outputBytes = m * n * 4;
  const cBuffer = gpu.createBuffer({ size: Math.max(4, outputBytes), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const dimsBuffer = gpu.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  gpu.queue.writeBuffer(dimsBuffer, 0, new Uint32Array([m, k, n, 0]));
  const group = gpu.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: aBuffer } },
      { binding: 1, resource: { buffer: bBuffer } },
      { binding: 2, resource: { buffer: cBuffer } },
      { binding: 3, resource: { buffer: dimsBuffer } },
    ],
  });
  const read = gpu.createBuffer({ size: Math.max(4, outputBytes), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = gpu.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(Math.ceil(n / TILE), Math.ceil(m / TILE));
  pass.end();
  encoder.copyBufferToBuffer(cBuffer, 0, read, 0, outputBytes);
  gpu.queue.submit([encoder.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const output = new Float32Array(read.getMappedRange().slice(0));
  read.unmap();
  for (const buffer of [aBuffer, bBuffer, cBuffer, dimsBuffer, read]) buffer.destroy();
  return { output, elapsedMs: performance.now() - started };
}

function capabilities() {
  return {
    webgpu: true,
    wasm: typeof WebAssembly !== "undefined",
    hardwareConcurrency: navigator.hardwareConcurrency || 1,
    adapter: adapterInfo,
    limits: device ? {
      maxStorageBufferBindingSize: Number(device.limits.maxStorageBufferBindingSize || 0),
      maxComputeWorkgroupsPerDimension: Number(device.limits.maxComputeWorkgroupsPerDimension || 0),
    } : {},
    shaderSha256,
    protocol: "auro.webgpu.matmul-job.v2",
  };
}

async function coordinatorStatus() {
  const status = await request("/status");
  $("#coordinatorState").textContent = `${status.ready_workers || 0} ready / ${status.queue_depth || 0} queued`;
  return status;
}

async function pull() {
  const value = saveConfig();
  const caps = encodeURIComponent(JSON.stringify(capabilities()));
  const body = await request(`/job?worker_id=${encodeURIComponent(value.workerId)}&wait=15&capabilities=${caps}`);
  return body.job || null;
}

async function renew(job) {
  return request("/lease/renew", {
    method: "POST",
    body: JSON.stringify({
      job_id: job.job_id,
      worker_id: config().workerId,
      lease_token: job.lease_token,
    }),
  });
}

async function submit(job, result, error = null) {
  const workerEvidence = {
    secureContext: window.isSecureContext,
    userAgent: navigator.userAgent,
    adapter: adapterInfo,
    shaderSha256,
    protocol: "auro.webgpu.matmul-job.v2",
  };
  const payload = {
    job_id: job.job_id,
    worker_id: config().workerId,
    lease_token: job.lease_token,
    backend: "browser-webgpu",
    elapsed_ms: result?.elapsedMs || 0,
    error,
    result: result ? { shape: job.output_shape, base64: encode64(result.output) } : undefined,
    worker_evidence: workerEvidence,
  };
  return request("/result", { method: "POST", body: JSON.stringify(payload) });
}

async function executeJob(job) {
  const a = decode64(job.a.base64);
  const b = decode64(job.b.base64);
  const [m, k] = job.a.shape;
  const [bk, n] = job.b.shape;
  if (k !== bk) throw new Error("job matrix shapes do not align");
  let renewalFailure = null;
  const intervalMs = Math.max(2500, Math.min(15000, ((job.lease_expires_at * 1000) - Date.now()) / 2));
  const timer = setInterval(() => {
    renew(job).catch((error) => {
      renewalFailure = error;
      log(`lease renewal failed: ${error.message}`);
    });
  }, intervalMs);
  try {
    const result = await matmul(a, b, m, k, n, job);
    if (renewalFailure) throw renewalFailure;
    return result;
  } finally {
    clearInterval(timer);
  }
}

async function loop() {
  if (running) return;
  saveConfig();
  await ensureDevice();
  running = true;
  stopped = false;
  $("#start").disabled = true;
  $("#stop").disabled = false;
  setState("working", "working");
  while (!stopped) {
    let job = null;
    try {
      await coordinatorStatus();
      job = await pull();
      if (!job) continue;
      log(`leased ${job.job_id} attempt ${job.attempt}/${job.max_attempts}`);
      const result = await executeJob(job);
      const response = await submit(job, result);
      completedJobs += 1;
      $("#jobs").textContent = String(completedJobs);
      $("#receipt").textContent = response.receipt?.receipt_sha256 || "missing";
      log(`${job.job_id} ${job.a.shape.join("x")} @ ${job.b.shape.join("x")} in ${result.elapsedMs.toFixed(2)} ms`);
    } catch (error) {
      log(error.message);
      if (job) {
        try { await submit(job, null, error.message); }
        catch (reportError) { log(`failure receipt error: ${reportError.message}`); }
      }
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
  }
  running = false;
  $("#start").disabled = false;
  $("#stop").disabled = true;
  setState("stopped");
}

async function benchmark() {
  const size = Math.max(16, Math.min(2048, Number($("#size").value) || 256));
  const a = new Float32Array(size * size);
  const b = new Float32Array(size * size);
  for (let index = 0; index < a.length; index += 1) {
    a[index] = Math.sin(index * 0.01);
    b[index] = Math.cos(index * 0.013);
  }
  const result = await matmul(a, b, size, size, size);
  const gflops = (2 * size * size * size) / (result.elapsedMs * 1e6);
  $("#benchmark").textContent = `${size} x ${size}: ${result.elapsedMs.toFixed(2)} ms / ${gflops.toFixed(2)} GFLOP/s`;
  log(`local benchmark: ${$("#benchmark").textContent}`);
}

$("#save").onclick = () => {
  try { saveConfig(); log("tab configuration saved"); }
  catch (error) { log(error.message); }
};
$("#start").onclick = () => loop().catch((error) => { setState("blocked", "blocked"); log(error.message); });
$("#stop").onclick = () => { stopped = true; };
$("#bench").onclick = () => benchmark().catch((error) => log(error.message));

ensureDevice()
  .then(async () => {
    setState("ready", "ready");
    try { await coordinatorStatus(); } catch (error) { log(`coordinator: ${error.message}`); }
    if (bootstrap.autostart) await loop();
  })
  .catch((error) => {
    setState("blocked", "blocked");
    log(error.message);
  });
