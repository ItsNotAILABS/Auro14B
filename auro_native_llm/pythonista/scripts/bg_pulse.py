"""Background pulse script — logs MESIE heartbeat style ticks into DB."""

import db
from datetime import datetime

db.create_table(
    "pulses",
    {
        "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "ts": "TEXT",
        "msg": "TEXT",
    },
)
db.insert(
    "pulses",
    {
        "ts": datetime.utcnow().isoformat() + "Z",
        "msg": "background pulse under JS service (Pythonista pattern)",
    },
)
print("pulse written", datetime.utcnow().isoformat())
