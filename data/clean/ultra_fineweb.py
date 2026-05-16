from concurrent.futures import ProcessPoolExecutor, as_completed
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

input_path = Path("F:/transformer-decoder/pretraining/raw/Ultra-FineWeb")
output_path = Path("F:/transformer-decoder/pretraining/clean/Ultra-FineWeb")

NUM_WORKERS = 8


def process_file(parquet_file: Path) -> tuple[str, int, int]:
    """
    单个文件的清洗逻辑，在子进程中执行。
    返回 (文件名, 保留数量, 过滤数量)
    """
    cleaner = BaseCleaner()
    df = pd.read_parquet(parquet_file)

    records = []
    for text, score in zip(df["content"], df["score"]):
        if float(score) >= 0.7:
            result = cleaner.clean(text)
            if result.passed:
                records.append({"text": result.text, "score": score})

    out_file = output_path / parquet_file.name
    pd.DataFrame(records).to_parquet(out_file, index=False)

    passed = len(records)
    filtered = len(df) - passed
    return parquet_file.name, passed, filtered


if __name__ == "__main__":
    output_path.mkdir(parents=True, exist_ok=True)

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

        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {}
            file_tasks = {}
            for f in pending:
                future = executor.submit(process_file, f)
                tid = progress.add_task(f"  ↳ {f.name} 处理中...", total=None)
                file_tasks[future] = tid
                futures[future] = f

            for future in as_completed(futures):
                filename, passed, filtered = future.result()
                tid = file_tasks[future]
                progress.remove_task(tid)
                progress.update(
                    file_task,
                    advance=1,
                    description=f"[bold cyan]最近完成: {filename} [green]{passed}[/green] 保留 / [red]{filtered}[/red] 过滤",
                )
