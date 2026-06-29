import argparse
import json
from pathlib import Path

from openai import OpenAI
from pydantic import ValidationError
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from data.continued.paths import JSON_DIR, JSON_JP_DIR
from data.continued.book.prompts import translate_prompt
from data.continued.book.structure import BookPage, TextOutput
from data.continued.utils import get_model

base_url, api_key, model = get_model(provider="DOUBAO")
client = OpenAI(base_url=base_url, api_key=api_key)

console = Console()


def translate(json_path: str, output_path: str, start: int = 0):
    image_dir = Path(json_path)
    output_dir = Path(output_path) / image_dir.name

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    jsons = sorted(image_dir.glob("*.json"), key=lambda x: int(x.stem))
    jsons_to_process = [item for item in jsons if int(item.stem) >= start]

    console.print(f"[bold cyan]📂 来源:[/] {image_dir}")
    console.print(f"[bold cyan]💾 输出:[/] {output_dir}")
    console.print(f"[bold cyan]🖼  共计:[/] {len(jsons_to_process)} 个文件\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]翻译中"),
        TextColumn("[yellow]{task.fields[filename]}[/]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("", total=len(jsons_to_process), filename="")

        for item in jsons_to_process:
            name = item.stem
            progress.update(task, filename=f"{name}.png")

            result = invoke(item)

            with open(f"{output_dir}/{name}.json", mode="w", encoding="utf-8") as f:
                f.write(result.model_dump_json(indent=2))

            progress.advance(task)


def invoke(json_path: Path):
    with open(json_path, mode="r", encoding="utf-8") as f:
        content = BookPage.model_validate_json(f.read())

    for item in content.paragraphs:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": translate_prompt.format(
                        json_schema=json.dumps(TextOutput.model_json_schema(), ensure_ascii=False)
                    ),
                },
                {"role": "user", "content": item.content},
            ],
            temperature=0.2,
            top_p=0.9,
            extra_body={"thinking": {"type": "disabled"}},
            max_tokens=16 * 1024,
        )

        result = response.choices[0].message.content
        if result is None:
            raise ValueError("提取结果为空")
        try:
            text = TextOutput.model_validate_json(result).text
        except ValidationError:
            text = result
            console.print(f"{json_path.stem} {result[:30]}")

        item.content = text

    return content


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--start", type=int, required=False, default=0)

    args = parser.parse_args()

    translate(
        json_path=str(JSON_JP_DIR / args.name),
        output_path=str(JSON_DIR),
        start=args.start,
    )
