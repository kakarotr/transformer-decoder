from pathlib import Path

import pandas as pd
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from data.clean.base_cleaner import BaseCleaner

input_path = Path("F:/transformer-decoder/pretraining-data/raw/Ultra-FineWeb")
output_path = Path("F:/transformer-decoder/pretraining-data/clean/Ultra-FineWeb")

cleaner = BaseCleaner()

parquet_files = sorted(input_path.glob("*.parquet"))
pending = [f for f in parquet_files if not (output_path / f.name).exists()]

with Progress(
    SpinnerColumn(),
    TextColumn("[bold cyan]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TaskProgressColumn(),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
) as progress:
    file_task = progress.add_task("处理文件", total=len(pending))

    for parquet_file in pending:
        progress.update(file_task, description=f"[bold cyan]{parquet_file.name}")

        df = pd.read_parquet(parquet_file)
        total_rows = len(df)

        row_task = progress.add_task("  ↳ 清洗中", total=total_rows)

        records = []
        for text, score in zip(df["content"], df["score"]):
            if float(score) >= 0.85:
                result = cleaner.clean(text)
                if result.passed:
                    records.append({"text": result.text, "score": score})
            progress.advance(row_task)

        pd.DataFrame(records).to_parquet(output_path / parquet_file.name, index=False)

        passed = len(records)
        filtered = total_rows - passed

        progress.update(row_task, description=f"  ↳ 完成 [green]{passed}[/green] 保留 / [red]{filtered}[/red] 过滤")
        progress.advance(file_task)
