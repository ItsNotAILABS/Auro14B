"""Launch the native Auro/HIM runtime and Cloudflare commercial UI together.

This command never uses Workers AI or an external model fallback. It starts the
repository-native production API, waits for readiness, starts the Cloudflare
Worker locally, verifies the gateway, opens the browser, and shuts both
processes down together.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import