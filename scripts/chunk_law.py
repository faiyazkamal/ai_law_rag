import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Detect chapter headers like:
# "CHAPTER I BANGLADESH MILITARY LAW – ITS ORIGIN AND EXTENT"
# "Chapter I 1 Military Law – Its origin and extent Paras 1 – 3"
CHAPTER_RE = re.compile(
    r"^(?:CHAPTER|Chapter)\s+([IVXLC0-9]+)\b(.*)$"
)

# Detect paragraph numbers like:
# "1. Disciplinary Code ..." or "4. Persons permanently ..."
PARA_RE = re.compile(r"^(\d{1,4})\.\s+(.*)$")

def clean_weird_chars(s: str) -> str:
    # Your text has artifacts like ï€­, â€“ etc.
    # Keep it simple: remove obvious junk sequences and normalize spaces.
    s = s.replace("ï€­", " ")
    s = s.replace("â€“", "-")
    s = s.replace("â€”", "-")
    s = s.replace("â€", '"').replace("â€œ", '"')
    s = s.replace("â€˜", "'").replace("â€™", "'")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

def split_into_chapters(lines: List[str]) -> List[Tuple[str, str, List[str]]]:
    """
    Returns list of (chapter_id, chapter_title, chapter_lines)
    If no chapter found, everything becomes CHAPTER-0.
    """
    chapters = []
    current_id = "0"
    current_title = "Preamble/Unlabeled"
    buf = []

    def flush():
        nonlocal buf, current_id, current_title
        if buf:
            chapters.append((current_id, current_title, buf))
        buf = []

    for ln in lines:
        s = clean_weird_chars(ln)
        if not s:
            buf.append("")
            continue

        m = CHAPTER_RE.match(s)
        # avoid false positives: must contain "Chapter" at start
        if m:
            # start new chapter
            flush()
            roman = m.group(1).strip()
            rest = m.group(2).strip()
            current_id = roman
            current_title = rest if rest else f"Chapter {roman}"
            buf.append(s)
        else:
            buf.append(s)

    flush()
    return chapters

def split_chapter_into_paras(chapter_id: str, chapter_title: str, chapter_lines: List[str]) -> List[Dict]:
    chunks = []
    current_para_no: Optional[str] = None
    current_para_title: str = ""
    buf: List[str] = []

    def flush():
        nonlocal buf, current_para_no, current_para_title
        if not buf:
            return

        text = "\n".join(buf).strip()
        if not text:
            buf = []
            return

        if current_para_no is None:
            # chapter intro / header material before first numbered para
            chunk_id = f"CHAPTER-{chapter_id}-INTRO-0"
            chunks.append({
                "section_kind": "ChapterIntro",
                "section_id": f"CHAPTER-{chapter_id}",
                "section_title": chapter_title,
                "chunk_id": chunk_id,
                "text": text
            })
        else:
            chunk_id = f"CHAPTER-{chapter_id}-PARA-{current_para_no}"
            chunks.append({
                "section_kind": "Paragraph",
                "section_id": f"{chapter_id}.{current_para_no}",
                "section_title": current_para_title if current_para_title else f"Para {current_para_no}",
                "chunk_id": chunk_id,
                "text": text
            })

        buf = []

    for ln in chapter_lines:
        s = ln.strip()

        m = PARA_RE.match(s)
        if m:
            # New para begins
            flush()
            current_para_no = m.group(1)
            current_para_title = clean_weird_chars(m.group(2))
            buf.append(s)
        else:
            buf.append(s)

    flush()
    return chunks

def split_big_chunks(chunks: List[Dict], max_chars: int = 2500) -> List[Dict]:
    out = []
    for c in chunks:
        t = c["text"]
        if len(t) <= max_chars:
            out.append(c)
            continue

        # Split by blank lines
        parts = re.split(r"\n\s*\n", t)
        acc = []
        part_id = 0

        def emit():
            nonlocal part_id, acc
            if not acc:
                return
            out.append({
                **{k: c[k] for k in ["section_kind", "section_id", "section_title"]},
                "chunk_id": f"{c['chunk_id']}-PART-{part_id}",
                "text": "\n\n".join(acc).strip()
            })
            part_id += 1
            acc = []

        for p in parts:
            p = p.strip()
            if not p:
                continue
            if sum(len(x) for x in acc) + len(p) + 2 > max_chars and acc:
                emit()
                acc.append(p)
            else:
                acc.append(p)

        emit()

    return out

def main():
    root = Path(__file__).resolve().parents[1]
    data = root / "data"
    clean_path = data / "law_clean.md"
    out_path = data / "chunks.jsonl"

    text = clean_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.split("\n")

    chapters = split_into_chapters(lines)

    all_chunks: List[Dict] = []
    for chap_id, chap_title, chap_lines in chapters:
        all_chunks.extend(split_chapter_into_paras(chap_id, chap_title, chap_lines))

    all_chunks = split_big_chunks(all_chunks, max_chars=2500)

    with out_path.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_chunks)} chunks -> {out_path}")

if __name__ == "__main__":
    main()
