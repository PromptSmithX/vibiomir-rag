#!/usr/bin/env python3
"""
scripts/fetch_pubmed.py
High-speed, robust fetcher for PubMed articles via Europe PMC REST API.
Solves the NCBI Proof-of-Work / JS challenge block by directly retrieving
structured metadata (Title, Abstract, Journal, Year) via Europe PMC.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fetch_pubmed")

EUROPE_PMC_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/searchPOST"
SHARD_SIZE = 10_000
BATCH_SIZE = 100
CONCURRENCY = 6

CORPUS_SCHEMA = pa.schema([
    ("doc_id", pa.int64()),
    ("url", pa.string()),
    ("final_url", pa.string()),
    ("domain", pa.string()),
    ("content_type", pa.string()),
    ("title", pa.string()),
    ("text", pa.string()),
    ("language", pa.string()),
    ("content_hash", pa.string()),
    ("http_status", pa.int32()),
    ("crawl_status", pa.string()),
    ("text_chars", pa.int32()),
    ("bytes_downloaded", pa.int64()),
])


def extract_pmid(url: str) -> str:
    cleaned = url.rstrip("/")
    return cleaned.split("/")[-1]


def build_doc_record(doc_id: int, url: str, item: dict[str, Any] | None) -> dict[str, Any]:
    if item is not None:
        title = item.get("title") or ""
        abstract = item.get("abstractText") or ""
        text = abstract if abstract else title
        crawl_status = "SUCCESS" if text else "EMPTY_CONTENT"
        http_status = 200
    else:
        title = ""
        text = ""
        crawl_status = "NOT_FOUND"
        http_status = 404

    text_bytes = text.encode("utf-8")
    content_hash = hashlib.sha256(text_bytes).hexdigest()

    return {
        "doc_id": doc_id,
        "url": url,
        "final_url": url,
        "domain": "pubmed.ncbi.nlm.nih.gov",
        "content_type": "application/json",
        "title": title,
        "text": text,
        "language": "en",
        "content_hash": content_hash,
        "http_status": http_status,
        "crawl_status": crawl_status,
        "text_chars": len(text),
        "bytes_downloaded": len(text_bytes),
    }


def write_shard(records: list[dict[str, Any]], shard_index: int, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_path = output_dir / f"part-{shard_index:06d}.parquet"

    table = pa.Table.from_pylist(records, schema=CORPUS_SCHEMA)
    pq.write_table(table, shard_path, compression="zstd", row_group_size=2048)
    logger.info("Saved shard %s (%d records)", shard_path.name, len(records))
    return shard_path


async def fetch_batch(
    session: aiohttp.ClientSession,
    batch: list[tuple[int, str, str]],
    semaphore: asyncio.Semaphore,
    max_retries: int = 4,
) -> dict[str, dict[str, Any]]:
    """Fetch metadata for a batch of (doc_id, url, pmid) from Europe PMC."""
    pmids = [pmid for _, _, pmid in batch]
    query = " OR ".join(f"EXT_ID:{p}" for p in pmids)
    data = {
        "query": query,
        "resultType": "core",
        "pageSize": str(len(pmids)),
        "format": "json",
    }
    headers = {
        "User-Agent": "ViBioMIR-BiomedicalCrawler/2.0 (research; pair-prog)",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    async with semaphore:
        for attempt in range(max_retries):
            try:
                async with session.post(
                    EUROPE_PMC_URL,
                    data=data,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=25),
                ) as resp:
                    if resp.status == 200:
                        raw = await resp.json()
                        result_list = raw.get("resultList", {}).get("result", [])
                        return {item.get("pmid"): item for item in result_list if item.get("pmid")}
                    elif resp.status == 429:
                        await asyncio.sleep(2.0 * (attempt + 1))
                    else:
                        await asyncio.sleep(1.0 * (attempt + 1))
            except Exception as exc:
                if attempt == max_retries - 1:
                    logger.warning("Batch fetch failed after %d retries: %s", max_retries, exc)
                await asyncio.sleep(1.0 * (attempt + 1))

    return {}


async def run_fetcher(
    partition_files: list[Path],
    output_dir: Path,
    limit: int | None = None,
    concurrency: int = CONCURRENCY,
    batch_size: int = BATCH_SIZE,
):
    # 1. Read input partitions
    targets: list[tuple[int, str, str]] = []
    seen_doc_ids: set[int] = set()

    for p in partition_files:
        if not p.exists():
            logger.error("Partition file not found: %s", p)
            continue
        logger.info("Reading %s...", p)
        tbl = pq.read_table(p, columns=["doc_id", "url"])
        for doc_id, url in zip(tbl["doc_id"].to_pylist(), tbl["url"].to_pylist()):
            if doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                targets.append((doc_id, url, extract_pmid(url)))

    if limit:
        targets = targets[:limit]

    logger.info("Total unique PubMed targets to fetch: %d", len(targets))

    # 2. Check existing output shards to resume if applicable
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_shards = sorted(output_dir.glob("part-*.parquet"))
    already_fetched_ids: set[int] = set()
    next_shard_idx = len(existing_shards)

    if existing_shards:
        logger.info("Found %d existing shards in %s. Checking completed doc_ids...", len(existing_shards), output_dir)
        for shard in existing_shards:
            try:
                t = pq.read_table(shard, columns=["doc_id"])
                already_fetched_ids.update(t["doc_id"].to_pylist())
            except Exception as e:
                logger.warning("Could not read %s: %e", shard, e)
        logger.info("Resuming: %d doc_ids already present. Skipping them.", len(already_fetched_ids))
        targets = [t for t in targets if t[0] not in already_fetched_ids]
        logger.info("Remaining targets: %d", len(targets))

    if not targets:
        logger.info("All targets already fetched! Exiting.")
        return

    # 3. Batch targets
    batches = [targets[i : i + batch_size] for i in range(0, len(targets), batch_size)]
    logger.info("Split into %d batches of size %d (concurrency=%d)", len(batches), batch_size, concurrency)

    semaphore = asyncio.Semaphore(concurrency)
    conn = aiohttp.TCPConnector(limit=concurrency + 2, ttl_dns_cache=300)
    buffer: list[dict[str, Any]] = []
    total_processed = 0
    total_success = 0
    total_targets = len(targets)
    start_time = time.monotonic()

    async with aiohttp.ClientSession(connector=conn) as session:
        # Process in chunks of concurrent batches
        chunk_step = concurrency * 2
        for chunk_idx in range(0, len(batches), chunk_step):
            active_chunk = batches[chunk_idx : chunk_idx + chunk_step]
            tasks = [fetch_batch(session, b, semaphore) for b in active_chunk]
            results = await asyncio.gather(*tasks)

            for b, res_map in zip(active_chunk, results):
                for doc_id, url, pmid in b:
                    item = res_map.get(pmid)
                    rec = build_doc_record(doc_id, url, item)
                    buffer.append(rec)
                    total_processed += 1
                    if rec["crawl_status"] == "SUCCESS":
                        total_success += 1

                # Write shard when buffer is full
                while len(buffer) >= SHARD_SIZE:
                    shard_records = buffer[:SHARD_SIZE]
                    buffer = buffer[SHARD_SIZE:]
                    write_shard(shard_records, next_shard_idx, output_dir)
                    next_shard_idx += 1

            elapsed = time.monotonic() - start_time
            rate = total_processed / elapsed if elapsed > 0 else 0
            pct = (total_processed / total_targets) * 100
            eta_s = (total_targets - total_processed) / rate if rate > 0 else 0
            logger.info(
                "Progress: %d/%d (%.1f%%) | Success: %d | Rate: %.1f docs/s | ETA: %.1fm",
                total_processed,
                total_targets,
                pct,
                total_success,
                rate,
                eta_s / 60,
            )

        # Flush remaining buffer
        if buffer:
            write_shard(buffer, next_shard_idx, output_dir)
            next_shard_idx += 1

    total_elapsed = time.monotonic() - start_time
    logger.info(
        "FETCH COMPLETED! Total: %d, Success: %d, Time: %.1fs (%.1f docs/s)",
        total_processed,
        total_success,
        total_elapsed,
        total_processed / total_elapsed if total_elapsed > 0 else 0,
    )


def main():
    parser = argparse.ArgumentParser(description="Fetch PubMed abstracts via Europe PMC")
    parser.add_argument(
        "--partitions",
        nargs="+",
        default=["partitions_v3/pubmed_part_1.parquet", "partitions_v3/pubmed_part_2.parquet"],
        help="Input partition parquet files",
    )
    parser.add_argument(
        "--output-dir",
        default="data/crawl/runs/run-pubmed/corpus",
        help="Target corpus directory for parquet shards",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents for testing")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY, help="Number of concurrent requests")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="PMIDs per request")

    args = parser.parse_args()
    partition_paths = [Path(p) for p in args.partitions]
    out_dir = Path(args.output_dir)

    asyncio.run(
        run_fetcher(
            partition_files=partition_paths,
            output_dir=out_dir,
            limit=args.limit,
            concurrency=args.concurrency,
            batch_size=args.batch_size,
        )
    )


if __name__ == "__main__":
    main()
