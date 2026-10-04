# ruff: noqa: E501
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sqlite3
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from crawler.config import apply_run_name, load_config
from crawler.utils.files import replace_with_retry

DEFAULT_SEED = 20261004
REVIEW_FIELDS = (
    "review_title_correct",
    "review_body_correct",
    "review_structure_ok",
    "review_noise_ok",
    "review_medical_values_ok",
)
BOILERPLATE_MARKERS = (
    "tin liên quan",
    "cùng chuyên mục",
    "bài viết liên quan",
    "chia sẻ bài viết",
    "xem thêm",
    "mọi quyền được bảo lưu",
    "related articles",
    "recommended articles",
    "share this article",
    "all rights reserved",
    "相关阅读",
    "相关文章",
    "推荐阅读",
    "版权所有",
    "联系我们",
)


def _stable_key(doc_id: int, seed: int, group: str) -> str:
    return hashlib.sha256(f"{seed}:{group}:{doc_id}".encode()).hexdigest()


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"\w+", value, flags=re.UNICODE))


def score_document(row: dict[str, Any], duplicate_count: int) -> tuple[int, list[dict[str, Any]], dict[str, int]]:
    text = row.get("text") or ""
    title = row.get("title") or ""
    text_chars = int(row.get("text_chars") or len(text))
    downloaded = max(1, int(row.get("bytes_downloaded") or 0))
    reasons: list[dict[str, Any]] = []
    flags = {
        "short_text": 0,
        "low_text_ratio": 0,
        "duplicate_content": 0,
        "boilerplate": 0,
        "structure_risk": 0,
        "title_mismatch": 0,
    }

    def add(code: str, points: int, detail: str) -> None:
        reasons.append({"code": code, "points": points, "detail": detail})

    if text_chars < 300:
        add("very_short_text", 30, f"only {text_chars} characters")
        flags["short_text"] = 1
    elif text_chars < 600:
        add("short_text", 20, f"only {text_chars} characters")
        flags["short_text"] = 1
    elif text_chars < 1000:
        add("brief_text", 10, f"only {text_chars} characters")

    ratio = text_chars / downloaded
    if ratio < 0.003:
        add("very_low_text_ratio", 20, f"text/download ratio {ratio:.4f}")
        flags["low_text_ratio"] = 1
    elif ratio < 0.01:
        add("low_text_ratio", 10, f"text/download ratio {ratio:.4f}")
        flags["low_text_ratio"] = 1
    elif ratio < 0.02:
        add("moderate_text_ratio", 5, f"text/download ratio {ratio:.4f}")

    if duplicate_count > 1:
        add("duplicate_content", 20, f"same content hash appears {duplicate_count} times")
        flags["duplicate_content"] = 1

    lower = text.casefold()
    marker_hits = [marker for marker in BOILERPLATE_MARKERS if marker in lower]
    if marker_hits:
        points = min(20, 5 * len(marker_hits))
        add("boilerplate_markers", points, ", ".join(marker_hits[:4]))
        flags["boilerplate"] = 1

    normalized_title = _normalize(title)
    normalized_lead = _normalize(text[:4000])
    if not normalized_title:
        add("missing_title", 15, "title is empty")
        flags["title_mismatch"] = 1
    elif normalized_title not in normalized_lead:
        title_tokens = set(normalized_title.split())
        lead_tokens = set(normalized_lead.split())
        overlap = len(title_tokens & lead_tokens) / max(1, len(title_tokens))
        if overlap < 0.35:
            add("title_body_mismatch", 10, f"title token overlap {overlap:.0%}")
            flags["title_mismatch"] = 1

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    structured = sum(line.startswith(("#", "-", "*", "|")) for line in lines)
    if len(lines) >= 6 and structured / len(lines) > 0.60:
        add("list_heading_dominant", 10, f"{structured}/{len(lines)} structured lines")
        flags["structure_risk"] = 1
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if paragraphs and max(map(len, paragraphs)) < 120:
        add("no_substantial_paragraph", 15, "no paragraph reaches 120 characters")
        flags["structure_risk"] = 1
    normalized_blocks = [_normalize(part) for part in paragraphs if len(part) >= 30]
    if normalized_blocks:
        repeated = len(normalized_blocks) - len(set(normalized_blocks))
        if repeated / len(normalized_blocks) >= 0.20:
            add("repeated_blocks", 10, f"{repeated} repeated blocks")
            flags["structure_risk"] = 1

    if not row.get("language"):
        add("missing_language", 5, "language is empty")
    if text_chars > 50_000:
        add("extreme_length", 5, f"{text_chars} characters")

    return min(100, sum(item["points"] for item in reasons)), reasons, flags


def allocate_sqrt_quotas(
    domain_sizes: dict[str, int], capacities: dict[str, int], total: int
) -> dict[str, int]:
    quotas = {domain: 0 for domain in domain_sizes if capacities.get(domain, 0) > 0}
    remaining = total
    while remaining:
        eligible = [d for d in quotas if quotas[d] < capacities[d]]
        if not eligible:
            break
        weight_sum = sum(math.sqrt(domain_sizes[d]) for d in eligible)
        ideals = {d: remaining * math.sqrt(domain_sizes[d]) / weight_sum for d in eligible}
        added = 0
        for domain in eligible:
            amount = min(capacities[domain] - quotas[domain], int(math.floor(ideals[domain])))
            quotas[domain] += amount
            added += amount
        remaining -= added
        if not remaining:
            break
        ranked = sorted(
            eligible,
            key=lambda d: (-(ideals[d] - math.floor(ideals[d])), -domain_sizes[d], d),
        )
        for domain in ranked:
            if remaining == 0:
                break
            if quotas[domain] < capacities[domain]:
                quotas[domain] += 1
                remaining -= 1
        if added == 0 and not ranked:
            break
    if sum(quotas.values()) != total:
        raise ValueError(f"cannot allocate {total} stratified samples")
    return quotas


def _batches(corpus_dir: Path, columns: list[str]) -> Iterable[dict[str, Any]]:
    shards = sorted(corpus_dir.glob("part-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no Parquet shards found in {corpus_dir}")
    for shard in shards:
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(columns=columns, batch_size=16_384):
            yield from batch.to_pylist()


def _atomic_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(text, encoding=encoding, newline="\n")
    replace_with_retry(temporary, path)


def _percentile(connection: sqlite3.Connection, domain: str, count: int, fraction: float) -> int:
    offset = min(count - 1, int((count - 1) * fraction))
    return int(
        connection.execute(
            "SELECT risk FROM records WHERE domain=? ORDER BY risk LIMIT 1 OFFSET ?",
            (domain, offset),
        ).fetchone()[0]
    )


def _write_html(path: Path, samples: list[dict[str, Any]]) -> None:
    payload = json.dumps(samples, ensure_ascii=False).replace("</", "<\\/")
    document = f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>ViBioMIR crawl QA</title><style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#0b1020;color:#e8edf7}}
header{{position:sticky;top:0;background:#111936;padding:16px;z-index:2;border-bottom:1px solid #334}}
.controls{{display:flex;gap:10px;flex-wrap:wrap}}select,input,button,textarea{{padding:7px;background:#17213f;color:#fff;border:1px solid #465477;border-radius:6px}}
main{{max-width:1200px;margin:auto;padding:18px}}article{{background:#121a31;margin:12px 0;padding:16px;border:1px solid #293657;border-radius:10px}}
.badges{{display:flex;gap:7px;flex-wrap:wrap}}.badge{{padding:3px 8px;border-radius:12px;background:#26365e}}.risk{{background:#7b2638}}
a{{color:#7fc7ff}}pre{{white-space:pre-wrap;background:#0a1020;padding:12px;border-radius:8px;max-height:520px;overflow:auto}}
.review{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px;margin-top:12px}}textarea{{width:100%;box-sizing:border-box}}
</style></head><body><header><h1>ViBioMIR crawl QA — {len(samples)} samples</h1>
<div class="controls"><select id="group"><option value="">All groups</option><option>high_risk</option><option>random</option><option>stratified</option></select>
<select id="domain"><option value="">All domains</option></select><input id="minRisk" type="number" min="0" max="100" value="0" placeholder="Min risk">
<button id="exportReview">Export reviewed JSONL</button></div></header><main id="items"></main>
<script>const DATA={payload}; const KEY='vibiomir-qa-review-v1'; let reviews=JSON.parse(localStorage.getItem(KEY)||'{{}}');
const groupSelect=document.getElementById('group'); const domainSelect=document.getElementById('domain');
const minRiskInput=document.getElementById('minRisk'); const itemsContainer=document.getElementById('items');
const exportReviewButton=document.getElementById('exportReview');
const esc=s=>String(s??'').replace(/[&<>\"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}}[c]));
const domains=[...new Set(DATA.map(x=>x.domain))].sort(); domainSelect.innerHTML+=""+domains.map(x=>`<option>${{esc(x)}}</option>`).join('');
function render(){{itemsContainer.innerHTML=''; const g=groupSelect.value,d=domainSelect.value,m=+minRiskInput.value; DATA.filter(x=>(!g||x.sample_group===g)&&(!d||x.domain===d)&&x.risk_score>=m).forEach(x=>{{
const r=reviews[x.doc_id]||{{}}; const el=document.createElement('article'); el.innerHTML=`<div class="badges"><span class="badge">${{esc(x.sample_group)}}</span><span class="badge risk">risk ${{x.risk_score}}</span><span class="badge">${{esc(x.domain)}}</span><span class="badge">${{x.text_chars}} chars</span></div><h2>${{esc(x.title||'(no title)')}}</h2><a target="_blank" href="${{esc(x.url)}}">${{esc(x.url)}}</a><p>${{x.risk_reasons.map(y=>esc(y.code)+' +'+y.points).join(' · ')}}</p><details><summary>Full text</summary><pre>${{esc(x.text)}}</pre></details><div class="review">${{['review_title_correct','review_body_correct','review_structure_ok','review_noise_ok','review_medical_values_ok'].map(k=>`<label>${{k}} <select data-k="${{k}}"><option value=""></option><option value="true">true</option><option value="false">false</option></select></label>`).join('')}}</div><textarea placeholder="Review notes">${{esc(r.review_notes||'')}}</textarea>`;
el.querySelectorAll('select[data-k]').forEach(s=>{{s.value=r[s.dataset.k]===undefined?'':String(r[s.dataset.k]);s.onchange=save}});el.querySelector('textarea').oninput=save;function save(){{const v={{review_notes:el.querySelector('textarea').value}};el.querySelectorAll('select[data-k]').forEach(s=>v[s.dataset.k]=s.value===''?null:s.value==='true');reviews[x.doc_id]=v;localStorage.setItem(KEY,JSON.stringify(reviews));}}itemsContainer.appendChild(el);}})}}
[groupSelect,domainSelect,minRiskInput].forEach(x=>x.oninput=render);exportReviewButton.onclick=()=>{{const lines=DATA.map(x=>JSON.stringify({{...x,...(reviews[x.doc_id]||{{}})}})).join('\\n')+'\\n';const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([lines],{{type:'application/jsonl'}}));a.download='qa_reviewed.jsonl';a.click();}};render();</script></body></html>"""
    _atomic_text(path, document)


def audit_corpus(
    corpus_dir: Path,
    output_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    sample_size: int = 100,
    high_risk_domain_cap: int = 10,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    domain_sizes: Counter[str] = Counter()
    hash_counts: Counter[str] = Counter()
    for row in _batches(corpus_dir, ["crawl_status", "domain", "content_hash"]):
        if row["crawl_status"] != "SUCCESS":
            continue
        domain_sizes[row["domain"]] += 1
        if row.get("content_hash"):
            hash_counts[row["content_hash"]] += 1
    total_success = sum(domain_sizes.values())
    required = sample_size * 3
    if total_success < required:
        raise ValueError(f"need at least {required} SUCCESS rows, found {total_success}")
    duplicates = {value: count for value, count in hash_counts.items() if count > 1}

    handle = tempfile.NamedTemporaryFile(
        prefix=".audit-crawl-",
        suffix=".sqlite",
        dir=output_dir,
        delete=False,
    )
    handle.close()
    database = Path(handle.name)
    connection = sqlite3.connect(database)
    connection.execute(
        """CREATE TABLE records(
        doc_id INTEGER PRIMARY KEY, domain TEXT, risk INTEGER, reasons TEXT,
        random_key TEXT, stratified_key TEXT, short_text INTEGER,
        low_text_ratio INTEGER, duplicate_content INTEGER, boilerplate INTEGER,
        structure_risk INTEGER, title_mismatch INTEGER)"""
    )
    try:
        columns = [
            "doc_id", "domain", "title", "text", "language", "content_hash",
            "text_chars", "bytes_downloaded", "crawl_status",
        ]
        pending = []
        for row in _batches(corpus_dir, columns):
            if row["crawl_status"] != "SUCCESS":
                continue
            score, reasons, flags = score_document(
                row, duplicates.get(row.get("content_hash"), 1)
            )
            doc_id = int(row["doc_id"])
            pending.append(
                (
                    doc_id, row["domain"], score, json.dumps(reasons, ensure_ascii=False),
                    _stable_key(doc_id, seed, "random"),
                    _stable_key(doc_id, seed, "stratified"),
                    flags["short_text"], flags["low_text_ratio"],
                    flags["duplicate_content"], flags["boilerplate"],
                    flags["structure_risk"], flags["title_mismatch"],
                )
            )
            if len(pending) >= 10_000:
                connection.executemany("INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", pending)
                connection.commit()
                pending.clear()
        if pending:
            connection.executemany("INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", pending)
            connection.commit()
        connection.execute("CREATE INDEX idx_records_domain_risk ON records(domain,risk)")
        connection.execute("CREATE TABLE selected(doc_id INTEGER PRIMARY KEY, sample_group TEXT, sample_order INTEGER)")

        high_rows = connection.execute(
            """WITH ranked AS (
            SELECT doc_id,domain,risk,random_key,
                   ROW_NUMBER() OVER(PARTITION BY domain ORDER BY risk DESC,random_key,doc_id) rn
            FROM records)
            SELECT doc_id FROM ranked WHERE rn<=? ORDER BY risk DESC,random_key,doc_id LIMIT ?""",
            (high_risk_domain_cap, sample_size),
        ).fetchall()
        if len(high_rows) != sample_size:
            raise ValueError("not enough rows for capped high-risk sample")
        selected_rows = [(int(row[0]), "high_risk", i) for i, row in enumerate(high_rows)]
        connection.executemany("INSERT INTO selected VALUES (?,?,?)", selected_rows)

        random_rows = connection.execute(
            """SELECT doc_id FROM records WHERE doc_id NOT IN (SELECT doc_id FROM selected)
            ORDER BY random_key,doc_id LIMIT ?""",
            (sample_size,),
        ).fetchall()
        start = len(selected_rows)
        random_selected = [(int(row[0]), "random", start + i) for i, row in enumerate(random_rows)]
        connection.executemany("INSERT INTO selected VALUES (?,?,?)", random_selected)

        capacities = {
            row[0]: int(row[1])
            for row in connection.execute(
                """SELECT domain,count(*) FROM records
                WHERE doc_id NOT IN (SELECT doc_id FROM selected) GROUP BY domain"""
            )
        }
        quotas = allocate_sqrt_quotas(dict(domain_sizes), capacities, sample_size)
        stratified_ids: list[int] = []
        for domain in sorted(quotas):
            if quotas[domain] == 0:
                continue
            rows = connection.execute(
                """SELECT doc_id FROM records WHERE domain=?
                AND doc_id NOT IN (SELECT doc_id FROM selected)
                ORDER BY stratified_key,doc_id LIMIT ?""",
                (domain, quotas[domain]),
            ).fetchall()
            stratified_ids.extend(int(row[0]) for row in rows)
        if len(stratified_ids) != sample_size:
            raise RuntimeError("stratified selection did not produce requested size")
        start += len(random_selected)
        connection.executemany(
            "INSERT INTO selected VALUES (?,?,?)",
            [(doc_id, "stratified", start + i) for i, doc_id in enumerate(stratified_ids)],
        )
        connection.commit()

        selected_meta = {
            int(row[0]): {
                "sample_group": row[1],
                "sample_order": int(row[2]),
                "risk_score": int(row[3]),
                "risk_reasons": json.loads(row[4]),
            }
            for row in connection.execute(
                """SELECT s.doc_id,s.sample_group,s.sample_order,r.risk,r.reasons
                FROM selected s JOIN records r USING(doc_id)"""
            )
        }
        found: dict[int, dict[str, Any]] = {}
        for row in _batches(corpus_dir, list(pq.ParquetFile(next(iter(sorted(corpus_dir.glob('part-*.parquet'))))).schema_arrow.names)):
            doc_id = int(row["doc_id"])
            if doc_id not in selected_meta:
                continue
            item = dict(row)
            item.update(selected_meta[doc_id])
            item["domain_success_count"] = domain_sizes[item["domain"]]
            for field in REVIEW_FIELDS:
                item[field] = None
            item["review_notes"] = ""
            found[doc_id] = item
        samples = sorted(found.values(), key=lambda item: item["sample_order"])
        if len(samples) != required:
            raise RuntimeError(f"expected {required} selected rows, found {len(samples)}")

        jsonl = "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in samples)
        _atomic_text(output_dir / "qa_samples.jsonl", jsonl)
        _write_html(output_dir / "qa_review.html", samples)

        sample_counts = Counter((row["domain"], row["sample_group"]) for row in samples)
        csv_path = output_dir / "domain_summary.csv"
        temporary_csv = csv_path.with_suffix(csv_path.suffix + ".part")
        with temporary_csv.open("w", encoding="utf-8-sig", newline="") as handle_csv:
            fieldnames = [
                "domain", "success_count", "share_pct", "risk_mean", "risk_p50",
                "risk_p90", "risk_max", "high_risk_count", "short_text_count",
                "low_text_ratio_count", "duplicate_content_count", "boilerplate_count",
                "structure_risk_count", "title_mismatch_count", "sample_high_risk",
                "sample_random", "sample_stratified",
            ]
            writer = csv.DictWriter(handle_csv, fieldnames=fieldnames)
            writer.writeheader()
            grouped = connection.execute(
                """SELECT domain,count(*),avg(risk),max(risk),sum(risk>=40),sum(short_text),
                sum(low_text_ratio),sum(duplicate_content),sum(boilerplate),
                sum(structure_risk),sum(title_mismatch) FROM records GROUP BY domain
                ORDER BY count(*) DESC,domain"""
            ).fetchall()
            for row in grouped:
                domain, count = row[0], int(row[1])
                writer.writerow(
                    {
                        "domain": domain, "success_count": count,
                        "share_pct": f"{100 * count / total_success:.4f}",
                        "risk_mean": f"{float(row[2]):.2f}",
                        "risk_p50": _percentile(connection, domain, count, 0.50),
                        "risk_p90": _percentile(connection, domain, count, 0.90),
                        "risk_max": int(row[3]), "high_risk_count": int(row[4]),
                        "short_text_count": int(row[5]), "low_text_ratio_count": int(row[6]),
                        "duplicate_content_count": int(row[7]), "boilerplate_count": int(row[8]),
                        "structure_risk_count": int(row[9]), "title_mismatch_count": int(row[10]),
                        "sample_high_risk": sample_counts[(domain, "high_risk")],
                        "sample_random": sample_counts[(domain, "random")],
                        "sample_stratified": sample_counts[(domain, "stratified")],
                    }
                )
        replace_with_retry(temporary_csv, csv_path)
        return {
            "success_rows": total_success,
            "domains": len(domain_sizes),
            "samples": len(samples),
            "output_dir": str(output_dir),
        }
    finally:
        connection.close()
        database.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only QA audit for crawled SUCCESS rows")
    parser.add_argument("--config", default="config/crawler.yaml")
    parser.add_argument("--run-name")
    parser.add_argument("--corpus-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--high-risk-domain-cap", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = apply_run_name(load_config(args.config), args.run_name)
    corpus_dir = Path(args.corpus_dir).resolve() if args.corpus_dir else config.paths.corpus_dir
    output_dir = Path(args.output_dir).resolve() if args.output_dir else config.paths.logs_dir
    result = audit_corpus(
        corpus_dir,
        output_dir,
        seed=args.seed,
        sample_size=args.sample_size,
        high_risk_domain_cap=args.high_risk_domain_cap,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
