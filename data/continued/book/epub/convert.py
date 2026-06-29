import argparse
import zipfile
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree as ET

from bs4 import BeautifulSoup

from data.continued.paths import EPUB_DIR

NAMESPACES = {
    "container": "urn:oasis:names:tc:opendocument:xmlns:container",
    "opf": "http://www.idpf.org/2007/opf",
}


def get_opf_path(zf: zipfile.ZipFile) -> str:
    raw = zf.read("META-INF/container.xml")
    root = ET.fromstring(raw)
    return root.find(".//container:rootfile", NAMESPACES).get("full-path")  # type: ignore


def parse_manifest(zf: zipfile.ZipFile, opf_path: str) -> tuple[list[dict], list[dict]]:
    """返回 (html_items, css_items)，html_items 按 spine 顺序排列。"""
    raw = zf.read(opf_path)
    opf = ET.fromstring(raw)
    opf_base = str(PurePosixPath(opf_path).parent)

    def to_zip_path(href: str) -> str:
        base = f"{opf_base}/{href}" if opf_base != "." else href
        return base.split("#")[0]

    manifest = {}
    for item in opf.findall(".//opf:manifest/opf:item", NAMESPACES):
        manifest[item.get("id")] = {
            "href": item.get("href"),
            "media_type": item.get("media-type", ""),
        }

    html_items = []
    for itemref in opf.findall(".//opf:spine/opf:itemref", NAMESPACES):
        idref = itemref.get("idref")
        item = manifest.get(idref)
        if item and "html" in item["media_type"]:
            html_items.append({"id": idref, "zip_path": to_zip_path(item["href"])})

    css_items = []
    for item in manifest.values():
        if item["media_type"] == "text/css":
            css_items.append(
                {
                    "filename": PurePosixPath(item["href"]).name,
                    "zip_path": to_zip_path(item["href"]),
                }
            )

    return html_items, css_items


def clean_html(html_bytes: bytes) -> bytes:
    soup = BeautifulSoup(html_bytes, "lxml-xml")

    for tag in soup.find_all("rt"):
        tag.decompose()  # 假名注音
    for tag in soup.find_all("rp"):
        tag.decompose()  # 注音括号
    for tag in soup.find_all(["rb", "ruby"]):
        tag.unwrap()  # 保留汉字，去掉容器

    for p in soup.find_all("p"):
        if not p.get_text(strip=True):
            p.decompose()  # 空段落

    return str(soup).encode("utf-8")


def extract_epub(epub_path: str, output_dir: str) -> dict[str, list[Path]]:
    """
    输出结构：
      output_dir/
        1.html, 2.html, ...   ← 清洗后的正文，按阅读顺序
        css/
          style.css, ...      ← 原始 CSS，供 LLM prompt 使用
    """
    out = Path(output_dir)
    css_dir = out / "css"
    out.mkdir(parents=True, exist_ok=True)
    css_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(epub_path, "r") as zf:
        opf_path = get_opf_path(zf)
        html_items, css_items = parse_manifest(zf, opf_path)

        html_files = []
        for i, item in enumerate(html_items):
            dest = out / f"{i + 1}.html"
            try:
                dest.write_bytes(clean_html(zf.read(item["zip_path"])).replace(b"\n", b"").replace(b"\r", b""))
                print(f"[html] {dest.name}")
                html_files.append(dest)
            except KeyError:
                print(f"[warn] 找不到: {item['zip_path']}，跳过")

        css_files = []
        for item in css_items:
            dest = css_dir / item["filename"]
            try:
                dest.write_bytes(zf.read(item["zip_path"]).replace(b"\n", b"").replace(b"\r", b""))
                print(f"[css]  css/{dest.name}")
                css_files.append(dest)
            except KeyError:
                print(f"[warn] 找不到: {item['zip_path']}，跳过")

    return {"html": html_files, "css": css_files}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--need_convert_sp", default=False, type=bool, required=False)

    args = parser.parse_args()

    extract_epub(
        epub_path=str(EPUB_DIR / f"{args.name}.epub"),
        output_dir=str(EPUB_DIR / args.name),
    )
