#!/usr/bin/env python3
"""Build and audit an Auro-4B structured-prewired native checkpoint.

This creates two identical smoke/dev geometries: the repository baseline and the
structured-prewired candidate. It records deterministic checksums and initial
activation statistics. Full training remains a separate governed operation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
