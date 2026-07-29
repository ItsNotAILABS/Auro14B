const TILE=8;
const $=selector=>document.querySelector(selector);
const params=new URLSearchParams(location.search);
const coordinator=(params.get('coordinator')||'http://127.0.0.1:8765').replace(/\/$/,'');
const token=params.get('token')||'';
const workerId=params.get('worker_id')||`browser-${crypto.randomUUID()}`;
let stopped=false,device=null,adapter=null,jobs=0;

function decode64(value){const raw=atob(value),bytes=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);return new Float32Array(bytes.buffer)}
function encode64(values){const bytes=new Uint8Array(values.buffer,values.byteOffset,values.byteLength);let out='';const chunk=0x8000;for(let i=0;i<bytes.length;i+=chunk)out+=String.fromCharCode(...bytes.subarray(i,Math.min(bytes.length,i+chunk)));return btoa(out)}
function log(value){$('#log').textContent=`${new Date().toISOString()} ${value}\n`+$('#log').textContent.slice(0,12000)}
function headers(){const h={'content-type':'application/json'};if(token)h['x-auro-cluster-token']=token;return h}

const shader=`
struct Dims { m:u32, k:u32, n:u32, pad:u32 }
@group(0) @binding(0) var<storage,read> A:array<f32>;
@group(0) @binding(1) var<storage,read> B:array<f32>;
@group(0) @binding(2) var<storage,read_write> C:array<f32>;
@group(0) @binding(3) var<uniform> dims:Dims;
var<workgroup> tileA:array<f32,64>;
var<workgroup> tileB:array<f32,64>;
@compute @workgroup_size(8,8,1)
fn main(@builtin(local_invocation_id) lid:vec3<u32>,@builtin(workgroup_id) wid:vec3<u32>){
 let row=wid.y*8u+lid.y;
 let col=wid.x*8u+lid.x;
 var sum=0.0;
 let tiles=(dims.k+7u)/8u;
 for(var t=0u;t<tiles;t=t+1u){
  let aCol=t*8u+lid.x;
  let bRow=t*8u+lid.y;
  let idx=lid.y*8u+lid.x;
  tileA[idx]=select(0.0,A[row*dims.k+aCol],row<dims.m&&aCol<dims.k);
  tileB[idx]=select(0.0,B[bRow*dims.n+col],bRow<dims.k&&col<dims.n);
  workgroupBarrier();
  for(var p=0u;p<8u;p=p+1u){sum=sum+tileA[lid.y*8u+p]*tileB[p*8u+lid.x];}
  workgroupBarrier();
 }
 if(row<dims.m&&col<dims.n){C[row*dims.n+col]=sum;}
}`;

async function ensureDevice(){
 if(device)return device;
 if(!navigator.gpu)throw new Error('WebGPU is unavailable in this browser');
 adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});
 if(!adapter)throw new Error('No WebGPU adapter');
 device=await adapter.requestDevice();
 device.lost.then(info=>{log(`device lost: ${info.message}`);device=null});
 $('#device').textContent=adapter.info?JSON.stringify(adapter.info):'WebGPU adapter ready';
 return device;
}

function storageBuffer(dev,data,usage=GPUBufferUsage.STORAGE){const b=dev.createBuffer({size:Math.max(4,(data.byteLength+3)&~3),usage:usage|GPUBufferUsage.COPY_DST});dev.queue.writeBuffer(b,0,data);return b}
async function matmul(a,b,m,k,n){
 const dev=await ensureDevice(),start=performance.now();
 const module=dev.createShaderModule({code:shader});
 const pipeline=dev.createComputePipeline({layout:'auto',compute:{module,entryPoint:'main'}});
 const aBuffer=storageBuffer(dev,a),bBuffer=storageBuffer(dev,b);
 const cBytes=m*n*4,cBuffer=dev.createBuffer({size:Math.max(4,cBytes),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
 const dimsBuffer=dev.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
 dev.queue.writeBuffer(dimsBuffer,0,new Uint32Array([m,k,n,0]));
 const bind=pipeline.getBindGroupLayout(0),group=dev.createBindGroup({layout:bind,entries:[
  {binding:0,resource:{buffer:aBuffer}},{binding:1,resource:{buffer:bBuffer}},{binding:2,resource:{buffer:cBuffer}},{binding:3,resource:{buffer:dimsBuffer}}
 ]});
 const read=dev.createBuffer({size:Math.max(4,cBytes),usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 const encoder=dev.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(n/TILE),Math.ceil(m/TILE));pass.end();encoder.copyBufferToBuffer(cBuffer,0,read,0,cBytes);dev.queue.submit([encoder.finish()]);
 await read.mapAsync(GPUMapMode.READ);const out=new Float32Array(read.getMappedRange().slice(0));read.unmap();
 for(const buffer of [aBuffer,bBuffer,cBuffer,dimsBuffer,read])buffer.destroy();
 return {out,elapsedMs:performance.now()-start};
}

async function pull(){
 const capabilities=encodeURIComponent(JSON.stringify({webgpu:true,wasm:typeof WebAssembly!=='undefined',hardwareConcurrency:navigator.hardwareConcurrency||1}));
 const response=await fetch(`${coordinator}/job?worker_id=${encodeURIComponent(workerId)}&wait=15&capabilities=${capabilities}`,{headers:headers()});
 if(!response.ok)throw new Error(`job poll ${response.status}`);return (await response.json()).job;
}
async function submit(job,result,error=null){
 const body={job_id:job.job_id,worker_id:workerId,backend:'browser-webgpu',elapsed_ms:result?.elapsedMs||0,error,result:result?{shape:job.output_shape,base64:encode64(result.out)}:undefined};
 const response=await fetch(`${coordinator}/result`,{method:'POST',headers:headers(),body:JSON.stringify(body)});if(!response.ok)throw new Error(`result submit ${response.status}: ${await response.text()}`);return response.json();
}
async function loop(){
 stopped=false;$('#state').textContent='working';
 while(!stopped){
  try{const job=await pull();if(!job)continue;const a=decode64(job.a.base64),b=decode64(job.b.base64);const [m,k]=job.a.shape,[bk,n]=job.b.shape;if(k!==bk)throw new Error('shape mismatch');const result=await matmul(a,b,m,k,n);await submit(job,result);jobs++;$('#jobs').textContent=String(jobs);log(`${job.job_id} ${m}x${k} @ ${k}x${n} in ${result.elapsedMs.toFixed(2)} ms`)}
  catch(error){log(error.message);await new Promise(resolve=>setTimeout(resolve,1000))}
 }
 $('#state').textContent='stopped';
}
async function benchmark(){const size=Number($('#size').value)||256,a=new Float32Array(size*size),b=new Float32Array(size*size);for(let i=0;i<a.length;i++){a[i]=Math.sin(i*.01);b[i]=Math.cos(i*.013)}const result=await matmul(a,b,size,size,size),gflops=(2*size*size*size)/(result.elapsedMs*1e6);$('#benchmark').textContent=`${size}²: ${result.elapsedMs.toFixed(2)} ms · ${gflops.toFixed(2)} GFLOP/s`;log($('#benchmark').textContent)}
$('#start').onclick=()=>loop();$('#stop').onclick=()=>{stopped=true};$('#bench').onclick=()=>benchmark().catch(e=>log(e.message));$('#worker').textContent=workerId;ensureDevice().then(()=>{$('#state').textContent='ready'}).catch(e=>{$('#state').textContent='blocked';log(e.message)});
