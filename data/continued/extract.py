import base64
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

from data.continued.prompts import base_section, json_schema_section
from data.continued.structure import BookPage

load_dotenv()

base_url = os.environ["BASE_URL"]
api_key = os.environ["API_KEY"]
model = os.environ["MODEL"]
client = OpenAI(base_url=base_url, api_key=api_key)

console = Console()


def extract(image_path: str, output_path: str, start: int = 0):
    image_dir = Path(image_path)
    output_dir = Path(output_path) / image_dir.name

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    images = sorted(image_dir.glob("*.png"), key=lambda x: int(x.stem))
    prompt = "\n\n".join(
        [
            base_section,
            json_schema_section.format(json_schema=json.dumps(BookPage.model_json_schema(), ensure_ascii=False)),
        ]
    )

    console.print(f"[bold cyan]📂 来源:[/] {image_dir}")
    console.print(f"[bold cyan]💾 输出:[/] {output_dir}")
    console.print(f"[bold cyan]🖼  共计:[/] {len(images)} 张\n")

    last_paragraph = ""

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
        task = progress.add_task("", total=len(images), filename="")

        for image in images:
            name = image.stem
            if int(name) < start:
                continue
            progress.update(task, filename=f"{name}.png")

            result = invoke(image, prompt, last_paragraph)
            last_paragraph = result.paragraphs[-1].content

            if result.paragraphs and result.paragraphs[0].type != "title":
                result.first_paragraph_has_indent = is_continuation(last_paragraph, result.first_paragraph_has_indent)

            with open(f"{output_dir}/{name}.json", mode="w", encoding="utf-8") as f:
                f.write(result.model_dump_json(indent=2))

            progress.advance(task)

    console.print(f"\n[bold green]✅ 完成！[/] 已提取 {len(images)} 张，结果保存至 {output_dir}")


def invoke(image: Path, prompt: str, last_paragraph: str):
    user_input = []
    if last_paragraph:
        user_input.append({"type": "text", "text": f"【上一页末尾段落】：「{last_paragraph}」\n\n请提取以下页面内容："})
    user_input.append(
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{to_base64(image)}"},
        }
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input},
        ],
        temperature=0.1,
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

    return result


def to_base64(image: Path):
    with open(image, mode="rb") as file:
        base64_bytes = base64.b64encode(file.read())
        return base64_bytes.decode("utf-8")


def is_continuation(last_paragraph: str | None, first_paragraph_has_indent: bool):
    if not last_paragraph:
        return first_paragraph_has_indent

    text = last_paragraph.rstrip()
    last_char = text[-1]
    if last_char == "\u201d":
        text = text[:-1].rstrip()
        last_char = text[-1] if text else ""

    if last_char not in {"。", "？", "！", "…"}:
        return False

    return first_paragraph_has_indent


if __name__ == "__main__":
    extract(
        image_path="/Users/linyongjin/Sengoku/Image/战国日本2：败者的美学",
        output_path="/Users/linyongjin/Sengoku/Json",
    )
