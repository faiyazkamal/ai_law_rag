import re
from pathlib import Path


def normalize(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Join broken lines (simple heuristic)
    lines = text.split("\n")
    out = []
    buf = ""

    for line in lines:
        s = line.strip()

        if not s:
            if buf:
                out.append(buf.strip())
                buf = ""
            out.append("")
            continue

        if buf and not re.search(r"[.!?:;]$", buf):
            buf += " " + s
        else:
            if buf:
                out.append(buf.strip())
            buf = s

    if buf:
        out.append(buf.strip())

    cleaned = "\n".join(out)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def main():
    root = Path(__file__).resolve().parents[1]
    data = root / "data"
    raw_path = data / "law_raw.txt"
    out_path = data / "law_clean.md"

    raw = raw_path.read_text(encoding="utf-8", errors="ignore")
    out_path.write_text(normalize(raw), encoding="utf-8")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
