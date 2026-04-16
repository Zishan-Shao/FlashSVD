#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.decode.bench_flashsvd_vs_svd_decode import main as _bench_main


def main() -> int:
    # Keep the old wrapper's conservative default unless explicitly overridden.
    if "--mlp_cuda_graph_scope" not in sys.argv:
        sys.argv.extend(["--mlp_cuda_graph_scope", "mlp"])
    return int(_bench_main())


if __name__ == "__main__":
    raise SystemExit(main())
