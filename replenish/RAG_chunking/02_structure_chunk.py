import re
from typing import List, Dict

heading_pat = re.compile(r'^(#{1,6})\s+(.*)$')  # 标题
fence_pat = re.compile(r'^```')  # fenced code fence


def split_markdown_structure(text, chunk_size=900, min_chunk=250, overlap_ratio=0.1) -> List[Dict]:
    lines = text.splitlines()
    sections = []
    in_code = False
    current = {"level": 0, "title": "", "content": [], "path": []}

    path_stack = []  # [(level, title)]

    for ln in lines:
        if fence_pat.match(ln):
            in_code = not in_code
        m = heading_pat.match(ln) if not in_code else None
        if m:
            if current["content"]:
                sections.append(current)
            level = len(m.group(1))
            title = m.group(2).strip()

            while path_stack and path_stack[-1][0] >= level:
                path_stack.pop()
            path_stack.append((level, title))
            breadcrumbs = [t for _, t in path_stack]
            current = {"level": level, "title": title, "content": [], "path": breadcrumbs}
        else:
            current["content"].append(ln)

    if current["content"]:
        sections.append(current)

    # 通过二次拆分/合并将部分平铺成块
    chunks = []

    def emit_chunk(text_block: str, path: List[str], level: int):
        chunks.append({
            "text": text_block.strip(),
            "meta": {
                "section_title": path[-1] if path else "",
                "breadcrumbs": path,
                "section_level": level,
            }
        })

    for sec in sections:
        raw = "\n".join(sec["content"]).strip()
        if not raw:
            continue
        if len(raw) <= chunk_size:
            emit_chunk(raw, sec["path"], sec["level"])
        else:
            paras = [p.strip() for p in  raw.split("\n\n") if p.strip()]
            buf = ""
            for p in paras:
                if len(buf) + len(p) + 2 <= chunk_size:
                    buf += (("\n\n" + p) if buf else p)
                else:
                    if buf:
                        emit_chunk(buf, sec["path"], sec["level"])
                    buf = p
            if buf:
                emit_chunk(buf, sec["path"], sec["level"])

    merged = []
    for ch in chunks:
        if not merged:
            merged.append(ch)
            continue
        if len(ch["text"]) < min_chunk and merged[-1]["meta"]["breadcrumbs"] == ch["meta"]["breadcrumbs"]:
            merged[-1]["text"] += "\n\n" + ch["text"]
        else:
            merged.append(ch)

    overlap = int(chunk_size * overlap_ratio)
    for ch in merged:
        bc = " > ".join(ch["meta"]["breadcrumbs"][-5:])
        prefix = f"[{bc}]\n" if bc else ""
        if prefix and not ch["text"].startswith(prefix):
            ch["text"] = prefix + ch["text"]
        # optional character overlap can在检索阶段用邻接聚合替代，这里略

    return merged




if __name__ == '__main__':

    text = open('../datas/visual_encoder.md', 'r', encoding='utf-8').read()

    chunks = split_markdown_structure(text, chunk_size=90, min_chunk=20)
    # print(chunks)
    for chunk in chunks:
        print(chunk['text'])
        print('=' * 100)

