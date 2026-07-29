const PYTHON_SOURCE=String.raw`
import hashlib,json,re

def _parts(value):
    if value is None:return []
    if isinstance(value,str):return [" ".join(x.split()) for x in re.split(r"(?<=[.!?])\\s+|\\n+|\\s*[•*-]\\s+",value) if x.strip()]
    if isinstance(value,dict):
        out=[]
        for key,item in value.items():
            if key not in {"raw","receipt","metadata","actions"}:out.extend(_parts(item))
        return out
    if isinstance(value,list):
        out=[]
        for item in value:out.extend(_parts(item))
        return out
    return [str(value)]

def fluidize(report):
    order=("answer","consensus","summary","key_points","recommendations","reasoning_summary","caveats","limitations","next_steps")
    raw=[]
    for key in order:raw.extend(_parts(report.get(key)))
    if not raw:raw=_parts(report)
    seen=set();out=[]
    for sentence in raw:
        key=re.sub(r"[^a-z0-9]+"," ",sentence.lower()).strip()
        if not key or key in seen:continue
        seen.add(key);text=sentence.strip();text=(text[:1].upper()+text[1:]) if text else text
        if text and text[-1] not in ".!?;:)]":text+="."
        out.append(text)
        if len(out)>=18:break
    transitions=("","More specifically, ","At the same time, ","Operationally, ","The key boundary is that ")
    rendered=[]
    for index,sentence in enumerate(out):
        prefix=transitions[min(index,len(transitions)-1)]
        rendered.append(prefix+sentence[:1].lower()+sentence[1:] if prefix else sentence)
    text=" ".join(rendered) or "No supported response content was produced."
    citations=report.get("citations") or []
    if citations:text+=" Sources: "+"; ".join(dict.fromkeys(map(str,citations)))+"."
    return {"schema":"auro.python_wasm.fluid_text.v1","text":text,"source_sha256":hashlib.sha256(json.dumps(report,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest(),"output_sha256":hashlib.sha256(text.encode()).hexdigest(),"sentences_out":len(out)}

result_json=json.dumps(fluidize(json.loads(report_json)),ensure_ascii=False)
`;

export class PythonWasmFluidizer{
  constructor({indexURL='/vendor/pyodide/',moduleURL='/vendor/pyodide/pyodide.mjs'}={}){this.indexURL=indexURL;this.moduleURL=moduleURL;this.pyodide=null;this.loading=null}
  async load(){
    if(this.pyodide)return this;
    if(!this.loading)this.loading=(async()=>{const mod=await import(this.moduleURL);const load=mod.loadPyodide||globalThis.loadPyodide;if(!load)throw new Error('local Pyodide loader is unavailable');this.pyodide=await load({indexURL:this.indexURL});return this})();
    return this.loading;
  }
  async fluidize(report){await this.load();this.pyodide.globals.set('report_json',JSON.stringify(report));const raw=await this.pyodide.runPythonAsync(PYTHON_SOURCE);return JSON.parse(raw)}
  status(){return {schema:'auro.python_wasm.fluidizer.status.v1',loaded:Boolean(this.pyodide),remotePackagesAllowed:false,moduleURL:this.moduleURL}}
}
