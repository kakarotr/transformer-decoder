import argparse
import json
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

from data.continued.prompts import epub_section, json_schema_section
from data.continued.structure import BookPage

load_dotenv()
base_url = os.environ["DEEPSEEK_BASE_URL"]
api_key = os.environ["DEEPSEEK_API_KEY"]
model = os.environ["DEEPSEEK_MODEL"]
client = OpenAI(base_url=base_url, api_key=api_key)

console = Console()


def extract(html_path: str, output_path: str, start: int = 0):
    html_dir = Path(html_path)
    output_dir = Path(output_path) / html_dir.name
    css_dir = html_dir / "css"

    css_contents: list[str] = []
    css_content: str | None = None
    if css_dir.exists():
        for css in css_dir.glob("*.css"):
            with open(css, mode="r", encoding="utf-8") as f:
                css_contents.append(f.read())
    if css_contents:
        css_content = "\n".join(css_contents)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    htmls = sorted(html_dir.glob("*.html"), key=lambda x: int(x.stem))
    htmls_to_process = [html for html in htmls if int(html.stem) >= start]

    prompt = "\n\n".join(
        [
            epub_section,
            json_schema_section.format(json_schema=json.dumps(BookPage.model_json_schema(), ensure_ascii=False)),
        ]
    )

    console.print(f"[bold cyan]📂 来源:[/] {html_dir}")
    console.print(f"[bold cyan]💾 输出:[/] {output_dir}")
    console.print(f"[bold cyan]🖼  共计:[/] {len(htmls_to_process)} 个\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]提取中"),
        TextColumn("[yellow]{task.fields[filename]}[/]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("", total=len(htmls_to_process), filename="")

        for html in htmls_to_process:
            name = html.stem
            progress.update(task, filename=f"{name}.html")

            with open(html, mode="r", encoding="utf-8") as f:
                html_content = f.read()

            result = invoke(prompt, html_content, css_content)

            if result.paragraphs:
                with open(f"{output_dir}/{name}.json", mode="w", encoding="utf-8") as f:
                    f.write(result.model_dump_json(indent=2))

            progress.advance(task)

    console.print(f"\n[bold green]✅ 完成！[/] 已提取 {len(htmls_to_process)} 个，结果保存至 {output_dir}")


def invoke(prompt: str, html: str, css: str | None):
    user_input = f"<css>\n{css}\n</css>\n\n<html>\n{html}\n</html>" if css else f"<html>\n{html}\n</html>"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input},
        ],
        extra_body={"thinking": {"type": "disabled"}},
        max_tokens=64 * 1024,
        response_format={"type": "json_object"},
    )

    result = response.choices[0].message.content
    if result is None:
        raise ValueError("提取结果为空")

    try:
        result = BookPage.model_validate_json(result)
    except Exception as e:
        print(result)
        raise e

    result.first_paragraph_has_indent = True
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--start", type=int, required=False, default=0)
    parser.add_argument("--is_jp", type=bool, required=False, default=False)

    args = parser.parse_args()

    extract(
        html_path=f"/Users/kakarot/Data/CPT/Sengoku/EPUB/{args.name}",
        output_path="/Users/kakarot/Data/CPT/Sengoku/Json"
        if not args.is_jp
        else "/Users/kakarot/Data/CPT/Sengoku/Json/JP",
        start=args.start,
    )
