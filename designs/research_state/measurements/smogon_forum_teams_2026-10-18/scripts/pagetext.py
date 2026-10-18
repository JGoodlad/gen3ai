"""Pull the paste text + metadata out of a cached pastebin.com page or pokepast.es /json body."""
import html
import json
import re


def pastebin(page: str) -> dict[str, str]:
    i = page.find('class="source')
    j = page.find("</ol>", i)
    lines = re.findall(r'<div class="de\d">(.*?)</div>', page[i:j], re.DOTALL)
    text = "\n".join(html.unescape(re.sub(r"<[^>]+>", "", ln)).replace("\xa0", " ").rstrip()
                     for ln in lines)
    title = re.findall(r"<h1>(.*?)</h1>", page)
    user = re.findall(r'class="username".*?>(.*?)<', page, re.DOTALL)
    date = re.findall(r'class="date".*?<span title="(.*?)"', page, re.DOTALL)
    return {"text": text, "paste_title": html.unescape(title[0]).strip() if title else "",
            "paste_user": user[0].strip() if user else "", "paste_date": date[0] if date else ""}


def pokepaste(body: str) -> dict[str, str]:
    d = json.loads(body)
    return {"text": d["paste"].replace("\r\n", "\n"), "paste_title": d.get("title", ""),
            "paste_user": d.get("author", ""), "paste_notes": d.get("notes", "")}
