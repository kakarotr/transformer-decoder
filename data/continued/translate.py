import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
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

from data.continued.prompts import json_schema_section, translate_prompt

load_dotenv()

base_url = os.environ["BASE_URL"]
api_key = os.environ["API_KEY"]
model = os.environ["MODEL"]
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
            if item.stem == "18":
                name = item.stem
                progress.update(task, filename=f"{name}.png")

                invoke(item)
                break


def invoke(json: Path):
    prompt = "\n\n".join([translate_prompt, json_schema_section])
    with open(json, mode="r", encoding="utf-8") as f:
        content = f.read()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": content},
        ],
        temperature=0.2,
        top_p=0.9,
        extra_body={"thinking": {"type": "disabled"}},
    )

    result = response.choices[0].message.content
    if result is None:
        raise ValueError("提取结果为空")

    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--start", type=int, required=False, default=0)

    args = parser.parse_args()

    translate(
        json_path=f"/Users/linyongjin/Sengoku/Json/JP/{args.name}",
        output_path="/Users/linyongjin/Sengoku/Json",
        start=args.start,
    )
