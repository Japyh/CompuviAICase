"""
Convert provided `dataset.jsonl` to `email_data.jsonl` expected by
`email_compliance_solution.py`.

This script maps the dataset fields to the minimal schema the main
script expects: {"text": ..., "label": ...} and writes a JSONL file
at the repository root named `email_data.jsonl`.
"""
import json
from pathlib import Path


def main():
    src = Path("dataset.jsonl")
    dst = Path("email_data.jsonl")

    if not src.exists():
        print(f"Source file not found: {src.resolve()}")
        return 1

    count = 0
    with src.open('r', encoding='utf-8') as fr, dst.open('w', encoding='utf-8') as fw:
        for line in fr:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                # skip malformed lines
                continue

            text = obj.get('text') or obj.get('body') or obj.get('message') or ''
            # prefer the explicit label keys used in the dataset
            label = obj.get('sub_tag') or obj.get('label') or obj.get('main_tag') or 'clean'

            out = {"text": text, "label": label}
            fw.write(json.dumps(out, ensure_ascii=False) + '\n')
            count += 1

    print(f"Wrote {count} records to {dst.resolve()}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
