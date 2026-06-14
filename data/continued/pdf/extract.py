import argparse
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

from data.continued.prompts import (
    horizontal_section,
    json_schema_section,
    vertical_section,
)
from data.continued.structure import BookPage

load_dotenv()

base_url = os.environ["BASE_URL"]
api_key = os.environ["API_KEY"]
model = os.environ["MODEL"]
client = OpenAI(base_url=base_url, api_key=api_key)

console = Console()


def extract(image_path: str, output_path: str, start: int = 0, base_sections: list[str] = [horizontal_section]):
    image_dir = Path(image_path)
    output_dir = Path(output_path) / image_dir.name

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    images = sorted(image_dir.glob("*.png"), key=lambda x: int(x.stem))
    images_to_process = [img for img in images if int(img.stem) >= start]

    prompt = "\n\n".join(
        [
            *base_sections,
            json_schema_section.format(json_schema=json.dumps(BookPage.model_json_schema(), ensure_ascii=False)),
        ]
    )

    console.print(f"[bold cyan]📂 来源:[/] {image_dir}")
    console.print(f"[bold cyan]💾 输出:[/] {output_dir}")
    console.print(f"[bold cyan]🖼  共计:[/] {len(images_to_process)} 张\n")

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
        task = progress.add_task("", total=len(images_to_process), filename="")

        for image in images_to_process:
            name = image.stem
            progress.update(task, filename=f"{name}.png")

            result = invoke(image, prompt, last_paragraph)

            if result.paragraphs and result.paragraphs[0].type != "title":
                result.first_paragraph_has_indent = is_continuation(last_paragraph, result.first_paragraph_has_indent)

            paragraph_blocks = [p for p in result.paragraphs if p.type == "paragraph"]
            if paragraph_blocks:
                last_paragraph = paragraph_blocks[-1].content
            elif any(p.type == "title" for p in result.paragraphs):
                last_paragraph = "。"

            with open(f"{output_dir}/{name}.json", mode="w", encoding="utf-8") as f:
                f.write(result.model_dump_json(indent=2))

            progress.advance(task)

    console.print(f"\n[bold green]✅ 完成！[/] 已提取 {len(images_to_process)} 张，结果保存至 {output_dir}")


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
        extra_body={"thinking": {"type": "disabled"}},
        # response_format={
        # "type": "json_object"
        # "type": "json_schema",
        # "json_schema": {
        #     "name": "BookPage",
        #     "description": "书页的结构",
        #     "schema": BookPage.model_json_schema(),
        #     "strict": True,
        # },
        # },
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

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--start", type=int, required=False, default=0)
    parser.add_argument("--is_jp", type=bool, required=False, default=False)

    args = parser.parse_args()

    extract(
        image_path=f"/Users/linyongjin/Sengoku/Image/{args.name}",
        output_path="/Users/linyongjin/Sengoku/Json" if not args.is_jp else "/Users/linyongjin/Sengoku/Json/JP",
        start=args.start,
        base_sections=[vertical_section],
    )
