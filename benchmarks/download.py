from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import pickle
import random
import re
import sys
import urllib.request
import zlib
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
_UA = {"User-Agent": "EnTrance-benchmark-downloader/1.0"}

DS_SERVER = "https://datasets-server.huggingface.co"

# ---------------------------------------------------------------------------
# datasets-server plumbing
# ---------------------------------------------------------------------------

def _fetch_json(url: str, headers: dict | None = None) -> dict:
    request = urllib.request.Request(url, headers={**_UA, **(headers or {})})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _rows(dataset: str, config: str, split: str, offset: int, length: int) -> list[dict]:
    url = (
        f"{DS_SERVER}/rows?dataset={dataset}&config={config}"
        f"&split={split}&offset={offset}&length={length}"
    )
    return _fetch_json(url)["rows"]

def _fetch_text(url: str, headers: dict | None = None) -> str:
    request = urllib.request.Request(url, headers={**_UA, **(headers or {})})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")

def _fetch_all(
    dataset: str,
    config: str,
    split: str,
    limit: int | None = None,
    page_size: int = 100,
) -> list[dict]:
    # Fetch rows, paginating by ``page_size`` (the API caps at 100).
    collected: list[dict] = []
    offset = 0
    while True:
        if limit is not None:
            remaining = limit - len(collected)
            if remaining <= 0:
                break
            page_size = min(page_size, remaining)
        batch = _rows(dataset, config, split, offset, page_size)
        if not batch:
            break
        collected.extend(batch)
        offset += len(batch)
        if len(batch) < page_size:
            break
        if limit is not None and len(collected) >= limit:
            break
    return collected

def download_aime2025(limit: int | None = None) -> list[dict]:
    problems: list[dict] = []
    # The opencompass dataset is JSONL-backed, so the datasets-server ``rows``
    # endpoint is not available for it; fetch the raw data files instead.
    for label, filename in (("I", "aime2025-I.jsonl"), ("II", "aime2025-II.jsonl")):
        url = (
            "https://huggingface.co/datasets/opencompass/AIME2025"
            f"/resolve/main/{filename}"
        )
        for index, line in enumerate(_fetch_text(url).splitlines()):
            if not line.strip():
                continue
            row = json.loads(line)
            # The opencompass dataset renders a few answers with LaTeX units
            # (e.g. "336^\\circ"); AIME answers are always plain integers.
            answer = re.sub(r"\^\s*\\circ", "", str(row["answer"])).strip()
            problems.append(
                {
                    "id": f"aime2025_{label}_{index + 1}",
                    "question": row["question"],
                    "answer": answer,
                }
            )
    if limit is not None:
        problems = problems[:limit]
    return problems

def download_gpqa_official(token: str, limit: int | None = None) -> list[dict]:
    url = "https://huggingface.co/datasets/Idavidrein/gpqa/resolve/main/gpqa_diamond.csv"
    request = urllib.request.Request(
        url, headers={**_UA, "Authorization": f"Bearer {token}"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        text = response.read().decode("utf-8")

    labels = ("A", "B", "C", "D")
    problems: list[dict] = []
    for index, row in enumerate(csv.DictReader(io.StringIO(text))):
        options = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        # Deterministic shuffle so the generated JSON is stable, with the
        # correct answer placed at a random but reproducible position.
        seed = row.get("Record ID") or str(index)
        order = list(range(4))
        random.Random(seed).shuffle(order)
        correct_index = order.index(0)
        rendered = "\n".join(
            f"{labels[position]}. {options[original]}"
            for position, original in enumerate(order)
        )
        problems.append(
            {
                "id": f"gpqa_diamond_{index + 1}",
                "question": f"{row['Question']}\n\n{rendered}",
                "answer": labels[correct_index],
            }
        )
        if limit is not None and len(problems) >= limit:
            break
    return problems

_LOWER_OPT = re.compile(r"^([a-d])\)\s*(.+)$")
_MAPPING = re.compile(r"^([A-D])\.\s*([a-d])$")

def _clean_mirror_question(question: str, answer: str) -> tuple[str, str]:
    nonempty = [line.strip() for line in question.splitlines() if line.strip()]
    if len(nonempty) < 8:
        return question, answer

    tail = nonempty[-4:]
    if not all(_MAPPING.match(line) for line in tail):
        return question, answer

    mapping = {m.group(1): m.group(2) for line in tail for m in [_MAPPING.match(line)]}

    # Collect the four lowercase option lines immediately above the mapping.
    option_lines: list[tuple[str, str]] = []
    for line in reversed(nonempty[:-4]):
        match = _LOWER_OPT.match(line)
        if match:
            option_lines.insert(0, (match.group(1), match.group(2)))
            if len(option_lines) == 4:
                break
        elif option_lines:
            break
    if len(option_lines) != 4:
        return question, answer

    header_count = len(nonempty) - 4 - 4
    header = "\n".join(nonempty[:header_count]).rstrip()
    labels = "ABCD"
    rendered = "\n".join(
        f"{labels[k]}. {option_lines[k][1]}" for k in range(4)
    )
    correct_lower = mapping.get(answer.upper(), "")
    try:
        new_answer = labels[ord(correct_lower) - ord("a")]
    except (IndexError, TypeError):
        return question, answer
    return f"{header}\n\n{rendered}", new_answer

def download_gpqa_mirror(limit: int | None = None) -> list[dict]:
    rows = _fetch_all("fingertap/GPQA-Diamond", "default", "test", limit=limit)
    problems: list[dict] = []
    for entry in rows:
        question, answer = _clean_mirror_question(
            entry["row"]["question"], str(entry["row"]["answer"]).strip().upper()
        )
        problems.append(
            {
                "id": f"gpqa_diamond_{entry['row_idx'] + 1}",
                "question": question,
                "answer": answer,
            }
        )
    return problems

def download_gpqa(limit: int | None = None) -> list[dict]:
    token = os.environ.get("HF_TOKEN")
    if token:
        try:
            return download_gpqa_official(token, limit)
        except Exception as exc:  # noqa: BLE001
            print(
                f"Warning: official GPQA download failed ({exc}); "
                "falling back to the ungated fingertap mirror.",
                file=sys.stderr,
            )
    return download_gpqa_mirror(limit)

def _decode_tests(raw: str | None) -> list[dict]:
    if not raw:
        return []

    # 1) Plain JSON list.
    try:
        data = json.loads(raw.strip())
        if isinstance(data, list):
            return data
    except Exception:  # noqa: BLE001 - fall through to the encoded form
        pass

    # 2) base64 -> zlib -> pickle -> JSON string.
    try:
        blob = base64.b64decode(raw.strip())
        try:
            blob = zlib.decompress(blob)
        except Exception:  # noqa: BLE001 - maybe not compressed
            pass
        try:
            blob = pickle.loads(blob)
        except Exception:  # noqa: BLE001 - maybe not pickled
            if isinstance(blob, bytes):
                blob = blob.decode("utf-8")
        if isinstance(blob, bytes):
            blob = blob.decode("utf-8")
        if isinstance(blob, str):
            blob = json.loads(blob)
        if isinstance(blob, list):
            return blob
    except Exception:  # noqa: BLE001
        return []
    return []

def download_livecodebench(limit: int | None = None) -> list[dict]:
    rows = _fetch_all(
        "lighteval/code_generation_lite",
        "release_v1",
        "test",
        limit=limit,
    )
    problems: list[dict] = []
    for entry in rows:
        row = entry["row"]
        problems.append(
            {
                "id": row["question_id"],
                "title": row["question_title"],
                "question_content": row["question_content"],
                "starter_code": row.get("starter_code") or "",
                "difficulty": row.get("difficulty") or "",
                "public_tests": _decode_tests(row.get("public_test_cases")),
                "private_tests": _decode_tests(row.get("private_test_cases")),
            }
        )
    return problems

def _write(filename: str, problems: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / filename
    path.write_text(
        json.dumps(problems, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(problems)} problems -> {path}")

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aime-limit", type=int, default=30, help="AIME problems (30 = full)")
    parser.add_argument("--gpqa-limit", type=int, default=198, help="GPQA Diamond problems (198 = full)")
    parser.add_argument("--lcb-limit", type=int, default=12, help="LiveCodeBench problems")
    args = parser.parse_args(argv)

    _write("aime2025.json", download_aime2025(args.aime_limit))
    _write("gpqa_diamond.json", download_gpqa(args.gpqa_limit))
    _write("livecodebench.json", download_livecodebench(args.lcb_limit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
