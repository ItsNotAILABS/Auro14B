"""Convert HIM-native-v0 NPZ into browser-readable float32 tensor blobs."""
from __future__ import annotations
import base64, io, json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
source=ROOT/"checkpoints/open/HIM-native-v0"
target=Path(__file__).resolve().parents[1]/"models/HIM-native-v0"
target.mkdir(parents=True,exist_ok=True)
raw=base64.b64decode((source/"weights.npz.b64").read_text(encoding="ascii"))
with np.load(io.BytesIO(raw)) as values:
 data={"schema":"him.browser.weights.v1","config":json.loads((source/"config.json").read_text()),"tensors":{}}
 for name in values.files:
  value=values[name].astype("<f4",copy=False)
  data["tensors"][name]={"shape":list(value.shape),"dtype":"float32-le","base64":base64.b64encode(value.tobytes()).decode("ascii")}
(target/"weights.json").write_text(json.dumps(data,separators=(",",":")),encoding="utf-8")
(target/"tokenizer.json").write_text((source/"tokenizer.json").read_text(),encoding="utf-8")
print(f"HIM_BROWSER_WEIGHTS_EXPORTED bytes={(target/'weights.json').stat().st_size}")
