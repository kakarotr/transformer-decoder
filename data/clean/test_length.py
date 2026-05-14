from pathlib import Path

import numpy as np
import pandas as pd

lengths = []
path = Path("F:/transformer-decoder/pretraining/1.12B/clean/finewiki")
files = path.glob("*.parquet")
for file in files:
    df = pd.read_parquet(file)
    for text in df["text"]:
        lengths.append(len(text))


lengths = np.array(lengths)
for pct in [50, 75, 90, 95, 99]:
    print(f"P{pct}: {np.percentile(lengths, pct):.0f} chars")
print(f"超过 8000 chars 的比例: {(lengths > 8000).mean() * 100:.1f}%")
