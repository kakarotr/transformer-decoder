import requests
from bs4 import BeautifulSoup

from data.continued.wiki.article.cleaner import WikiCleaner


def fetch():
    response = requests.get(
        "https://ja.wikipedia.org/api/rest_v1/page/html/%E8%B1%8A%E8%87%A3%E7%A7%80%E5%90%89",
        headers={
            "User-Agent": "WarringStatesBot/1.0 (https://github.com/kakarot/transformer-decoder; kakarotter7@gmail@gmail.com) python-requests/2.32.0"
        },
    )
    soup = BeautifulSoup(response.text, "html.parser")
    with open("/Users/kakarot/Documents/豊臣秀吉.html", mode="w", encoding="utf-8") as f:
        f.write(soup.prettify())


if __name__ == "__main__":
    with open("/Users/kakarot/Documents/豊臣秀吉_1.html", mode="r", encoding="utf-8") as f:
        content = f.read()

    cleaner = WikiCleaner(content)
    content = cleaner.clean()

    with open("/Users/kakarot/Documents/豊臣秀吉.html", mode="w", encoding="utf-8") as f:
        f.write(content)
