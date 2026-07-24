"""Create content-addressed Manifest V3 browser-extension packages."""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def package_extension(out_dir, name="Auro Sovereign Workspace", api_base="http://127.0.0.1:8090"):
    root=Path(out_dir); root.mkdir(parents=True,exist_ok=True)
    safe=re.sub(r"[^A-Za-z0-9._-]+","-",name).strip("-").lower() or "auro-extension"
    manifest={"manifest_version":3,"name":name,"version":"0.1.0","description":"Local Auro/MESIE workspace connector.","action":{"default_popup":"popup.html"},"host_permissions":[api_base.rstrip("/")+"/*"],"permissions":["storage"]}
    files={
      "manifest.json":json.dumps(manifest,indent=2),
      "popup.html":"<!doctype html><meta charset='utf-8'><title>Auro</title><main><h1>Auro</h1><p id='status'>Checking local runtime...</p><script src='popup.js'></script></main>",
      "popup.js":f"fetch('{api_base.rstrip('/')}/health').then(r=>r.json()).then(x=>document.querySelector('#status').textContent=x.ok?'Runtime online':'Runtime unavailable').catch(()=>document.querySelector('#status').textContent='Runtime unavailable');",
    }
    archive=root/(safe+".zip")
    with ZipFile(archive,"w",ZIP_DEFLATED) as package:
        for path,content in files.items(): package.writestr(path,content)
    data=archive.read_bytes(); digest=hashlib.sha256(data).hexdigest()
    return {"schema":"auro.extension.package.v1","name":name,"path":str(archive),"bytes":len(data),"sha256":digest,"manifest_version":3,"unpacked_files":sorted(files)}
