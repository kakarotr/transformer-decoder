from pathlib import Path

from pdf2image import convert_from_path
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

console = Console()


def pdf_to_images(pdf_path: str, output_path: str, dpi: int = 200) -> list[str]:
    output_dir = Path(output_path) / Path(pdf_path).stem
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        Panel(
            f"[bold cyan]PDF → Image[/bold cyan]\n"
            f"[dim]来源:[/dim] {pdf_path}\n"
            f"[dim]输出:[/dim] {output_path}\n"
            f"[dim]DPI :[/dim] {dpi}"
        )
    )

    with console.status("[bold yellow]正在解析 PDF...[/bold yellow]", spinner="dots"):
        pages = convert_from_path(pdf_path, dpi=dpi)

    console.print(f"[green]✓[/green] 共 [bold]{len(pages)}[/bold] 页，开始转换\n")

    saved_paths = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[bold cyan]{task.completed}/{task.total}[/bold cyan]"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[yellow]转换中...", total=len(pages))

        for i, page in enumerate(pages, start=1):
            image_path = output_dir / f"{i}.png"
            page.save(image_path, "PNG", quality=100)
            saved_paths.append(str(image_path))
            progress.update(task, advance=1, description=f"[yellow]正在保存 [cyan]{i}.png[/cyan]")

    console.print(f"\n[bold green]✓ 转换完成！[/bold green] {len(pages)} 张图片已保存至 [cyan]{output_path}[/cyan]")
    return saved_paths


pdf_path = "/Users/linyongjin/Sengoku/PDF"
OUTPUT_PATH = "/Users/linyongjin/Sengoku/Image"


if __name__ == "__main__":
    pdf_to_images(pdf_path=f"{pdf_path}/战国日本2：败者的美学.pdf", output_path=OUTPUT_PATH, dpi=300)
