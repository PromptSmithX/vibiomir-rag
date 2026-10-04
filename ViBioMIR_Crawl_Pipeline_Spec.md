# ViBioMIR — Crawl Pipeline Specification

## 1. Mục tiêu

Tài liệu này chốt thiết kế cho **giai đoạn crawl dữ liệu** của cuộc thi ViBioMIR.

Input chính:

- `links_corpus.parquet`
- Mỗi row gồm:
  - `id`: document ID do Ban Tổ chức cấp
  - `url`: URL nguồn

Output của giai đoạn crawl:

- Corpus sạch, có cấu trúc, có thể dùng lại cho:
  - BM25
  - dense embedding
  - chunking
  - reranking
- Giữ nguyên mapping:
  - `doc_id gốc -> URL -> nội dung đã crawl`

> `doc_id` gốc từ `links_corpus.parquet` là trường quan trọng nhất và phải được giữ xuyên suốt toàn pipeline.

Giai đoạn này **chưa chunk toàn bộ corpus** và **chưa embedding**.

---

## 2. Nguyên tắc thiết kế

Crawler cần đáp ứng:

1. **Không mất `doc_id` gốc**
2. **Có thể resume** nếu chương trình crash hoặc máy restart
3. **Không lưu hàng triệu file `.txt` / `.md` riêng lẻ**
4. **Không giữ raw HTML lâu dài**
5. Crawl bất đồng bộ nhưng phải **giới hạn theo domain**
6. Tách riêng network fetching, content extraction, cleaning và persistence
7. Có thể mở rộng domain-specific extractor về sau
8. Giữ cấu trúc nội dung đủ tốt để chunking sau này
9. Không dùng LLM trong bước crawl
10. Không chạy browser/headless Chrome hàng loạt

---

# 3. Schema dữ liệu đã chốt

## 3.1. Clean corpus schema

Mỗi document là **1 row trong Parquet**.

```text
doc_id            int64
url               string
final_url         string | null
domain            string
content_type      string | null

title             string | null
text              string | null

language          string | null
content_hash      string | null

http_status       int32 | null
crawl_status      string

text_chars        int32
bytes_downloaded  int64
```

Có thể bổ sung sau nếu cần:

```text
description       string | null
canonical_url     string | null
published_date    string | null
language_score    float32 | null
extractor         string | null
```

### Ý nghĩa

- `doc_id`: ID gốc từ Ban Tổ chức; tuyệt đối không tự thay bằng ID khác.
- `url`: URL gốc trong dataset.
- `final_url`: URL sau redirect.
- `domain`: domain parse từ URL.
- `content_type`: ví dụ `text/html`, `application/pdf`.
- `title`: tiêu đề bài viết.
- `text`: clean article body; giữ structure kiểu Markdown-like.
- `language`: `vi`, `en`, `zh`, `unknown`; detect sau khi đã extract text.
- `content_hash`: hash của clean text, dùng để nhận biết duplicate.
- `http_status`: HTTP response code cuối.
- `crawl_status`: trạng thái crawl.
- `text_chars`: số ký tự clean text.
- `bytes_downloaded`: dung lượng response đã tải.

---

# 4. Format của `text`

Không lưu mỗi document thành file `.md`.

Thay vào đó field `text` trong Parquet giữ cấu trúc gần Markdown:

```text
# Treatment of Hypertension

## Diagnosis

First paragraph...

Second paragraph...

## Treatment

- Drug A
- Drug B

Recommended dose: 5 mg/day.
```

Mục tiêu:

- giữ title
- giữ heading
- giữ paragraph boundary
- giữ bullet/list
- giữ table ở dạng text dễ đọc
- không flatten toàn bộ article thành một dòng

Không được clean mất các thông tin y khoa như:

```text
0.5 mg
mg/kg
1:1000
SpO2
Na+
95%
IV
IM
ACE inhibitor
```

---

# 5. Storage layout

```text
data/
├── source/
│   └── links_corpus.parquet
│
├── crawl/
│   ├── corpus/
│   │   ├── part-000000.parquet
│   │   ├── part-000001.parquet
│   │   ├── part-000002.parquet
│   │   └── ...
│   │
│   └── crawl_manifest.sqlite
│
└── logs/
    ├── crawler.log
    └── metrics/
```

Không dùng:

```text
documents/
├── 1.txt
├── 2.txt
├── 3.txt
├── ...
```

vì vài triệu file nhỏ sẽ gây filesystem overhead, copy/backup chậm, directory traversal chậm và khó batch processing.

---

# 6. Parquet strategy

Mỗi shard:

```text
10,000–20,000 documents
```

Khởi đầu nên dùng:

```text
compression = zstd
```

Tên file:

```text
part-000000.parquet
part-000001.parquet
...
```

Không tạo một file Parquet duy nhất hàng chục GB.

---

# 7. Raw HTML strategy

Default:

```text
HTTP response
    ↓
extract
    ↓
clean
    ↓
write Parquet
    ↓
discard raw HTML
```

Không giữ raw HTML sau khi extract thành công.

Nếu cần debug extractor, chỉ giữ sample nhỏ hoặc bật debug mode; không giữ raw HTML toàn corpus.

---

# 8. Tech stack

## Network

```text
aiohttp
```

Dùng cho async HTTP requests, connection pooling, keep-alive, timeout và streaming response.

## URL parsing

```text
urllib.parse
tldextract
```

## Fast HTML parsing

```text
selectolax
```

## Generic article extraction fallback

```text
trafilatura
```

## PDF extraction

```text
PyMuPDF
```

Package Python:

```text
pymupdf
```

Import:

```python
import fitz
```

## Parquet

```text
pyarrow
```

## Manifest

```text
SQLite
```

## Fast serialization phụ trợ

```text
orjson
```

## Language detection

Không dùng LLM. Có thể dùng một trong:

```text
fastText lid.176
lingua-language-detector
pycld3
```

Language detection nên chạy thành batch sau khi extract, không nhất thiết nằm trong hot path của network crawler.

---

# 9. Kiến trúc crawler

```text
links_corpus.parquet
        │
        ▼
  URL / Task Scheduler
        │
        ▼
 Async HTTP Fetchers
        │
        ▼
 Response Router
    ┌───────┴────────┐
    ▼                ▼
   HTML              PDF
    │                │
    ▼                ▼
 HTML Extractor   PDF Extractor
    │                │
    └───────┬────────┘
            ▼
          Cleaner
            │
            ▼
      Quality Validation
            │
            ▼
       Parquet Writer
            │
            ▼
      Manifest Update
```

Các stage nên tách module để có thể thay đổi độc lập.

---

# 10. Project structure đề xuất

```text
crawler/
├── main.py
├── config.py
├── scheduler.py
├── fetcher.py
│
├── extractors/
│   ├── __init__.py
│   ├── html_generic.py
│   ├── pdf.py
│   └── domains/
│       ├── __init__.py
│       └── ...
│
├── cleaner.py
├── language.py
├── writer.py
├── manifest.py
├── metrics.py
└── utils/
    ├── urls.py
    ├── hashing.py
    └── logging.py
```

---

# 11. Crawl manifest

SQLite là source of truth cho trạng thái crawl.

```sql
CREATE TABLE crawl_tasks (
    doc_id INTEGER PRIMARY KEY,
    url TEXT NOT NULL,
    domain TEXT,

    status TEXT NOT NULL DEFAULT 'PENDING',
    attempts INTEGER NOT NULL DEFAULT 0,

    http_status INTEGER,
    final_url TEXT,
    content_type TEXT,

    bytes_downloaded INTEGER,
    text_chars INTEGER,

    output_shard TEXT,
    error TEXT,

    updated_at TEXT
);
```

Các trạng thái:

```text
PENDING
FETCHING
SUCCESS
FAILED
TOO_LARGE
UNSUPPORTED
NEEDS_JS
NEEDS_OCR
```

Nếu chương trình crash:

```sql
SELECT *
FROM crawl_tasks
WHERE status IN ('PENDING', 'FAILED');
```

Không crawl lại document đã `SUCCESS`.

Khi startup, task bị kẹt ở `FETCHING` từ phiên chạy cũ phải được reset về `PENDING`.

---

# 12. Crawl strategy

## 12.1. Không chạy full corpus ngay

Triển khai theo 4 phase.

### Phase A — URL analysis

Chưa request URL.

Thống kê:

```text
total URLs
unique URLs
unique domains
URLs/domain
HTTP vs HTTPS
PDF-like URL count
query-param distribution
exact duplicate URL
```

Đặc biệt cần top domains theo số lượng URL.

### Phase B — pilot 10k–20k

Lấy sample có tính đại diện:

- ưu tiên top domains
- có cả HTML/PDF
- có VI/EN/ZH nếu nhận biết được
- không chỉ random hoàn toàn

Mục tiêu:

- đo throughput
- đo success rate
- đo clean-text size
- inspect chất lượng extraction
- tìm domain cần custom extractor

Không scale tiếp trước khi inspect output.

### Phase C — pilot 100k

Sau khi 10k–20k ổn, crawl `100,000` URL.

Thu metric:

```text
SUCCESS %
404 %
403 %
429 %
timeout %
PDF %
NEEDS_JS %
avg bytes downloaded
avg clean text chars
documents/sec
disk usage
```

Từ đó extrapolate storage/time cho full corpus.

### Phase D — bulk crawl

Chỉ mở full crawl sau khi pilot ổn.

Crawler phải có thể chạy nhiều ngày mà không cần babysit.

---

# 13. Concurrency strategy

Không dùng một global `Semaphore(1000)`.

Khởi đầu:

```text
global concurrency = 64
per-domain concurrency = 2–4
```

Có thể tăng global concurrency sau khi benchmark:

```text
64
→ 96
→ 128
```

Nhưng giữ per-domain thấp.

Lý do:

- tránh 429
- tránh 403/block
- giảm retry
- giảm socket exhaustion
- thân thiện với nguồn dữ liệu
- throughput thực tế ổn định hơn

---

# 14. Domain-aware scheduler

Không crawl hết một domain rồi mới sang domain khác.

Scheduler nên xen kẽ nhiều domain để:

- nhiều domain chạy song song
- reuse connection
- không spam một host
- host chậm không block toàn crawler

Có thể maintain:

```python
domain_queues: dict[str, asyncio.Queue]
domain_semaphores: dict[str, asyncio.Semaphore]
```

---

# 15. HTTP session

Dùng một `aiohttp.ClientSession` dùng chung.

Không tạo session mới cho từng request.

Cần:

- keep-alive
- connection pool
- DNS cache
- sane User-Agent

Crawler phải tuân thủ robots.txt/điều khoản truy cập của nguồn và không bypass CAPTCHA hoặc cơ chế hạn chế truy cập.

---

# 16. Timeout

Baseline:

```text
connect timeout = 7 sec
total timeout   = 30 sec
```

Có thể config.

Không để một page treo vài phút.

---

# 17. Retry policy

Retry tối đa:

```text
2–3 attempts
```

Retry:

```text
timeout
connection reset
429
500
502
503
504
```

Không retry thông thường:

```text
400
401
404
410
```

`403`: default không retry liên tục; ghi lại và xử lý theo domain sau.

Nếu response có `Retry-After` thì tôn trọng header này.

---

# 18. Response size limit

Baseline gợi ý:

```text
HTML max = 5 MB
PDF max  = 25 MB
```

Nếu vượt:

```text
crawl_status = TOO_LARGE
```

Ưu tiên streaming, không load response cực lớn vào RAM.

---

# 19. Content-Type router

Không dựa hoàn toàn vào file extension.

Dùng `HTTP Content-Type`:

```text
text/html
→ HTML extractor

application/pdf
→ PDF extractor

image/*
video/*
application/zip
...
→ UNSUPPORTED
```

Nếu server khai báo sai Content-Type, có thể sniff vài bytes đầu.

---

# 20. HTML extraction strategy

Ưu tiên theo thứ tự:

```text
1. domain-specific extractor (nếu có)
2. fast generic extraction bằng selectolax
3. trafilatura fallback
```

Không gọi extractor nặng nhất cho mọi page.

Nếu một domain có hàng trăm nghìn URL và structure ổn định, viết adapter riêng để lấy title/body trực tiếp.

---

# 21. HTML cleaning

Loại:

```text
script
style
nav
footer
cookie banners
ads
related-content noise
login UI
social-share UI
```

Giữ:

```text
title
headings
paragraphs
lists
tables nếu có nội dung
medical units
numbers
special symbols
```

Không lowercase toàn bộ text ở bước crawl.

Không stemming.

Không stopword removal.

---

# 22. PDF strategy

```text
PDF bytes
    ↓
PyMuPDF
    ↓
extract page text
    ↓
clean
```

Nếu PDF có page nhưng text gần như rỗng:

```text
crawl_status = NEEDS_OCR
```

Không OCR trong bulk crawler.

OCR chỉ làm sau cho subset quan trọng nếu cần.

---

# 23. JavaScript-heavy pages

Không dùng Playwright/Chrome mặc định.

Nếu HTTP response không extract được article:

```text
crawl_status = NEEDS_JS
```

Sau này chỉ xử lý bằng browser nếu document nằm trong candidate set quan trọng.

---

# 24. Deduplication

## Exact URL duplicate

Có thể detect trước crawl nhưng không được làm mất `doc_id`.

Ví dụ:

```text
doc_id 100 -> URL X
doc_id 200 -> URL X
```

Có thể fetch một lần nhưng phải giữ mapping cho cả hai ID.

## Content duplicate

Sau clean:

```python
content_hash = sha256(normalized_clean_text)
```

Có thể tận dụng để tránh embedding duplicate sau này, nhưng không được merge mất ID gốc.

---

# 25. URL normalization

Ở crawler v1 ưu tiên an toàn.

Có thể:

```text
remove #fragment
normalize hostname case
```

Có thể cân nhắc remove tracking params:

```text
utm_source
utm_medium
utm_campaign
fbclid
gclid
```

Nhưng **không xóa toàn bộ query string**.

Các param như:

```text
?id=123
?article=456
?page=789
```

có thể là identifier thật của article.

Nếu chưa chắc, giữ nguyên URL.

---

# 26. Language detection

Không dùng LLM.

Sau khi đã có `title + text`, chạy detector nhẹ.

Input ví dụ:

```python
sample = title + "\n" + text[:3000]
```

Output:

```text
vi
en
zh
unknown
```

Nếu confidence thấp thì để `unknown`.

Language detection không phải yếu tố chặn crawler nên có thể chạy batch sau.

---

# 27. Logging và metrics

Crawler phải log định kỳ.

Ví dụ mỗi 10–30 giây:

```text
processed
success
failed
pending
req/sec
status code counts
bytes downloaded
clean corpus size
top slow domains
top error domains
```

---

# 28. Backpressure

Không để downloader tạo hàng trăm nghìn response chờ extractor.

Giữa các stage nên có bounded queue:

```python
fetch_queue = asyncio.Queue(maxsize=...)
extract_queue = asyncio.Queue(maxsize=...)
write_queue = asyncio.Queue(maxsize=...)
```

Nếu writer chậm thì extractor và fetcher tự chậm lại, tránh RAM tăng không kiểm soát.

---

# 29. Writer behavior

Writer gom batch khoảng:

```text
10k–20k rows
```

rồi flush thành Parquet.

Sau khi shard ghi thành công mới update manifest `SUCCESS`.

Nên ghi atomic:

```text
part-000123.parquet.tmp
```

sau đó rename:

```text
part-000123.parquet
```

để tránh shard half-written.

---

# 30. Quality validation trước khi ghi

HTTP 200 không đồng nghĩa extraction tốt.

Check tối thiểu:

```text
title exists OR enough body text
text_chars >= threshold
```

Ví dụ:

```text
text_chars < 100
```

thì có thể đánh `EMPTY_CONTENT` hoặc `NEEDS_JS` tùy trường hợp.

Không reject quá mạnh ở giai đoạn đầu.

---

# 31. Priority crawl

Có thể thêm `priority` trong manifest:

```text
0 = high
1 = normal
2 = low
```

Vì test query đã biết trước, có thể dùng URL slug/domain/title-like URL signal để tạo candidate sơ bộ và crawl chúng trước.

Nhưng priority crawl là optimization. Crawler core phải hoạt động độc lập với logic retrieval.

---

# 32. Không làm ở giai đoạn crawl

Không:

```text
chunk toàn bộ corpus
embedding
FAISS
reranking
LLM query expansion
OCR toàn bộ PDF
Playwright toàn corpus
lowercase/stemming mạnh
BM25 indexing
```

Crawl stage chỉ có nhiệm vụ:

```text
URL
↓
reliable clean document
↓
persistent corpus
```

---

# 33. Default config v1

```yaml
crawler:
  global_concurrency: 64
  per_domain_concurrency: 3

  connect_timeout_seconds: 7
  total_timeout_seconds: 30

  max_attempts: 3

  max_html_bytes: 5242880       # 5 MB
  max_pdf_bytes: 26214400       # 25 MB

  parquet_rows_per_shard: 10000
  parquet_compression: zstd

  min_text_chars: 100

  queues:
    fetch: 256
    extract: 128
    write: 20000
```

Các giá trị này chỉ là baseline để benchmark, không hard-code sâu trong code.

---

# 34. Acceptance criteria cho crawler v1

## Reliability

- restart được
- resume được
- không mất mapping `doc_id`
- không duplicate task vô ý
- không corrupt shard nếu crash

## Extraction

Random inspect ít nhất 200 document:

- title đúng
- body là article thật
- ít navigation/footer noise
- giữ headings/paragraphs/lists
- không mất số/đơn vị y khoa

## Performance

Pilot 100k phải ghi lại:

```text
documents/sec
MB/sec
success rate
disk/document
CPU usage
RAM peak
```

## Storage

Từ 100k docs phải extrapolate được full corpus có fit budget khoảng 110 GB hay không.

---

# 35. Thứ tự triển khai cho coding agent

## Step 1

Viết:

```text
analyze_urls.py
```

Output:

```text
total URLs
unique URLs
top domains
PDF-like URLs
duplicate URLs
query-param stats
```

## Step 2

Tạo SQLite manifest từ `links_corpus.parquet`.

## Step 3

Implement async fetcher:

```text
aiohttp
timeouts
retry
per-domain concurrency
connection pooling
```

## Step 4

Implement HTML extractor:

```text
selectolax
+
trafilatura fallback
```

## Step 5

Implement PDF extractor bằng PyMuPDF.

## Step 6

Implement cleaner + schema validation.

## Step 7

Implement Parquet shard writer.

## Step 8

Implement resume/recovery.

## Step 9

Run:

```text
10k–20k pilot
```

Inspect thủ công output.

## Step 10

Fix extraction rồi chạy:

```text
100k pilot
```

Measure performance/storage.

## Step 11

Chỉ sau đó mới chạy full bulk crawl.

---

# 36. Tóm tắt quyết định đã khóa

## Schema

```text
doc_id
url
final_url
domain
content_type
title
text
language
content_hash
http_status
crawl_status
text_chars
bytes_downloaded
```

## Storage

```text
Parquet + ZSTD
```

Không dùng hàng triệu `.txt` / `.md`.

## State

```text
SQLite manifest
```

## HTML

```text
selectolax
→ domain adapter nếu có
→ trafilatura fallback
```

## PDF

```text
PyMuPDF
```

## HTTP

```text
aiohttp
```

## Initial concurrency

```text
global = 64
per-domain = 3
```

## Raw HTML

```text
không lưu mặc định
```

## Chunking

```text
chưa làm ở crawl stage
```

## Embedding

```text
chưa làm ở crawl stage
```

## LLM

```text
không dùng trong crawler
```

## Rollout

```text
URL analysis
→ 10k–20k pilot
→ inspect
→ 100k pilot
→ benchmark
→ full crawl
```

---

# 37. Nguyên tắc cuối

Giai đoạn crawl phải tạo ra một **clean document store trung lập với model**.

Sau này có thể đổi:

```text
BM25
Qwen3-Embedding-0.6B
Qwen3-Embedding-4B
BGE-M3
reranker khác
chunk size khác
```

mà **không phải crawl lại dữ liệu**.

Nếu phải thay embedding model mà vẫn dùng lại corpus được, thiết kế crawl đã đúng.
