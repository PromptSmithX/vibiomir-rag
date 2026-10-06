# Chạy ViBioMIR với 1 worker Windows và 5 worker Kaggle CPU

Guide này dùng một bản code crawler cho cả sáu worker. Mỗi worker có domain, SQLite manifest và corpus riêng. Các bước dưới đây chạy trong PowerShell tại root của repo, trừ khi có ghi khác.

## 1. Chuẩn bị sáu partition

Nguồn là `data/source/links_corpus.parquet` với cột `id` và `url`. Script giữ nguyên ID dưới tên `doc_id`, lập profile theo hostname, rồi greedy balance theo `estimated_work`. Domain affinity được giữ mặc định; chỉ domain gây bottleneck rõ ràng mới được split có kiểm soát.

```powershell
uv run python build_domain_profile.py sample `
  --input data/source/links_corpus.parquet `
  --output data/crawl/pilots/domain-profile-calibration-v1.jsonl `
  --per-domain 100 `
  --priority-per-domain 500

uv run python -m crawler --run-name domain-profile-calibration-v4 manifest init `
  --source-path data/source/links_corpus.parquet `
  --selection data/crawl/pilots/domain-profile-calibration-v1.jsonl

uv run python -m crawler --run-name domain-profile-calibration-v4 crawl run `
  --capture-fetch-timings `
  --progress off

uv run python build_domain_profile.py build `
  --input data/source/links_corpus.parquet `
  --status-manifest data/crawl/runs/pilot-10000-v1/crawl_manifest.sqlite `
  --timing-manifest data/crawl/runs/domain-profile-calibration-v4/crawl_manifest.sqlite `
  --timing-log logs/runs/domain-profile-calibration-v4/fetch_timings.jsonl `
  --output-dir data/crawl/runs/pilot-10000-v1

uv run python prepare_crawl_partitions.py `
  --input data/source/links_corpus.parquet `
  --domain-profile data/crawl/runs/pilot-10000-v1/domain_profile.csv `
  --output-dir partitions_v2
```

Calibration sample lấy 100 URL cho mỗi domain và 500 URL cho năm domain trọng điểm. Timing đo thời gian robots/network/read thực tế và loại thời gian đứng chờ semaphore hoặc crawl-delay queue. Lệnh `build` chỉ chạy khi calibration manifest không còn `PENDING`, `FETCHING` hoặc `RETRY_WAIT`; nếu crawl bị ngắt, chạy lại đúng lệnh `crawl run` để resume. `domain_profile.csv` dùng status rate và timing thực, đặt chi phí `ROBOTS_DENIED` ở mức rất thấp, rồi partition v2 mới greedy balance theo `estimated_work`. Domain chỉ được split khi work lớn hơn mức lý tưởng và mô phỏng giảm makespan ít nhất 10%; domain split có tổng concurrency tối đa 2 trên hai worker.

Sau khi script báo thành công, kiểm tra `partitions_v2/partition_summary.csv` và `balance_metadata.json`. Script đọc lại cả sáu output để kiểm tra tổng số dòng, `doc_id` duy nhất, domain affinity, split threshold và tổng concurrency của domain split. Nếu thư mục output đã tồn tại, script sẽ dừng để không ghi đè bản partition đang dùng.

Các file được tạo:

```text
partitions_v2/
  worker_0_local.parquet
  worker_1_kaggle.parquet
  worker_2_kaggle.parquet
  worker_3_kaggle.parquet
  worker_4_kaggle.parquet
  worker_5_kaggle.parquet
  domain_assignment.parquet
  partition_summary.csv
  balance_metadata.json
```

## 2. Đóng gói code crawler

```powershell
uv run python build_kaggle_bundle.py
```

Lệnh tạo `dist/crawler_bundle.zip`. Zip chứa package `crawler`, `config/crawler.yaml`, `pyproject.toml`, `uv.lock` và `requirements-kaggle.txt` đã khóa phiên bản. Nó không chứa source Parquet, manifest, corpus, log hoặc output cũ.

## 3. Tạo hai Kaggle Dataset private

Đăng nhập Kaggle bằng account sẽ chạy cả năm notebooks.

### Dataset `vibiomir-crawl-input`

1. Vào **Datasets → New Dataset**.
2. Upload sáu file sau từ `partitions_v2/`: `worker_1_kaggle.parquet` đến `worker_5_kaggle.parquet`, cùng `domain_assignment.parquet`. Có thể thêm `partition_summary.csv` và `balance_metadata.json` để xem tải của từng worker.
3. Đặt title/slug thành `vibiomir-crawl-input`, chọn **Private**, rồi tạo Dataset.
4. Xác nhận các file xuất hiện trong cây Input. Kaggle có thể mount tại `/kaggle/input/vibiomir-crawl-input/` hoặc `/kaggle/input/datasets/<owner>/vibiomir-crawl-input/`; notebook tự tìm theo Dataset slug. Không cần upload `worker_0_local.parquet` hoặc toàn bộ source Parquet.

### Dataset `vibiomir-crawler-code`

1. Tạo một Dataset mới, upload duy nhất `dist/crawler_bundle.zip`.
2. Đặt title/slug thành `vibiomir-crawler-code`, chọn **Private**.
3. Kaggle có thể giữ `crawler_bundle.zip` hoặc tự giải nén thành `crawler/`, `config/` và các file dependency. Notebook hỗ trợ cả hai dạng và copy code vào `/kaggle/working/app` trước khi chạy.

Kaggle cho phép tạo private Dataset; `Save & Run All` tạo một version chạy từ đầu và lưu output của notebook. Xem [Kaggle Datasets](https://www.kaggle.com/docs/datasets) và [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks).

## 4. Tạo và chạy năm notebooks

1. Import `kaggle_crawl_worker.ipynb` lên Kaggle và tạo năm bản sao private từ cùng template.
2. Trong **Settings**, chọn **CPU** (không bật GPU/TPU) và bật **Internet** để cài dependency từ PyPI và crawl URL.
3. Trong **Input → Add Input**, attach hai private Dataset trên cho từng notebook.
4. Để chạy production, đặt `MAX_DOCS = None`. Chỉ thay dòng `WORKER_ID = 1` ở cell đầu: lần lượt đặt `1`, `2`, `3`, `4`, `5` cho năm notebooks. `INPUT_FILE` và output path tự đổi theo ID; các cell khác dùng nguyên template.
5. Chọn **Save Version → Save & Run All** cho từng notebook. Có thể khởi chạy song song cả năm nếu account còn đủ năm CPU batch slots; đóng các interactive sessions không dùng đến.

Notebook giải nén code vào `/kaggle/working/app`, cài dependencies rồi chạy `crawler.kaggle_runner`. Nó đọc duy nhất partition của `WORKER_ID`, kiểm tra mọi domain với `domain_assignment.parquet`, tôn trọng robots.txt và chỉ ghi output trong `/kaggle/working/crawl_worker_<WORKER_ID>/`:

```text
crawl_worker_<WORKER_ID>/
  manifest.sqlite
  corpus/
    part-000000.parquet
    ...
  checkpoint.json
  summary.json
  crawler.log
  metrics/
    ...
```

`GLOBAL_CONCURRENCY = 32` và `PER_DOMAIN_CONCURRENCY = 2` nằm cùng cell cấu hình nếu cần giảm tải. Mặc định notebook dừng claim URL sau khoảng 10,5 giờ kể từ cell đầu; các request và hàng đợi đang chạy được xử lý hết, shard cuối được ghi và commit trước khi notebook thoát. Kaggle hiện công bố giới hạn 12 giờ cho CPU notebook và khoảng 20 GB `/kaggle/working` được lưu cùng version; hãy theo dõi kích thước output ở `summary.json` nếu worker cần nhiều lần resume. [Nguồn: Kaggle Notebooks](https://www.kaggle.com/docs/notebooks).

### Chạy thử 1.000 URL trước

Chỉ chạy notebook worker 1 với `WORKER_ID = 1` và `MAX_DOCS = 1000`. Khi đủ giới hạn, runner ngừng claim URL mới, drain request đang chạy, commit shard cuối, ghi checkpoint và kết thúc với `exit_reason: "target_limit"`. Tải hoặc attach toàn bộ output của version thử nghiệm làm previous output, đổi `MAX_DOCS = None`, rồi chạy lại đúng notebook worker 1 để tiếp tục phần còn lại. Các URL terminal trong 1.000 URL đầu không bị fetch lại.

Sau khi worker 1 được xác nhận ổn định, chạy workers 2–5 với `MAX_DOCS = None`. Nếu dùng các notebook đã tạo trước đó, cập nhật input `vibiomir-crawler-code` sang version Dataset chứa bundle mới nhất.

Một signal dừng bình thường dùng cơ chế graceful shutdown. Nếu nền tảng chấm dứt process đột ngột, lần resume sẽ dùng cơ chế recovery hiện có để hoàn tất shard đã rename hoặc reset task chưa commit. Đây là lý do phải giữ cả manifest lẫn corpus của version trước.

## 5. Lấy output và resume

Trong trang version của từng notebook, mở tab **Output** và chọn **Download All**. Giải nén file tải về trên máy local. Giữ nguyên toàn bộ thư mục `crawl_worker_<WORKER_ID>`; đừng chỉ tải `manifest.sqlite` hoặc `checkpoint.json` vì những dòng `SUCCESS` trong manifest trỏ tới các Parquet shard cũ.

Để chạy tiếp trên Kaggle:

1. Mở notebook của đúng worker và vào **Input → Add Input → Notebook Output Files**.
2. Chọn output từ version trước của chính worker đó. Chỉ attach **một** previous output cho worker hiện tại; vẫn giữ hai Dataset `vibiomir-crawl-input` và `vibiomir-crawler-code`.
3. Giữ nguyên `WORKER_ID`; chọn **Save Version → Save & Run All** lần nữa.

Runner tìm `checkpoint.json` của worker trong `/kaggle/input`, kiểm tra checksum partition và copy toàn bộ previous output sang `/kaggle/working`. Nó không sửa Kaggle input. Trước khi crawl, runner phục hồi shard đang staging, reset các task `FETCHING` bị bỏ dở thành `PENDING`, xác minh corpus cũ rồi tiếp tục. `SUCCESS` đã commit không được fetch lại. Nếu nhiều previous outputs cùng worker được attach hoặc checksum khác, notebook dừng với lỗi rõ ràng. [Kaggle hỗ trợ dùng output của notebook khác làm Input](https://www.kaggle.com/docs/notebooks).

`checkpoint.json` ghi sau mỗi 5.000 document đã commit và khi kết thúc. `summary.json` có `exit_reason`: `completed`, `target_limit`, `runtime_limit`, `stopped` hoặc `failed`. Khi `completed` và không còn `PENDING`/`FETCHING`/`RETRY_WAIT`, worker đã xong. `target_limit` là kết thúc sạch của lượt smoke test và có thể resume.

## 6. Chạy worker 0 trên Windows

Chạy trên máy local, không có giới hạn 10,5 giờ. `--run-name` tạo manifest/corpus riêng. Partition đã được validate nên có thể bỏ bước phân tích nguồn chung:

```powershell
uv run python -m crawler --run-name worker-0-local manifest init `
  --source-path partitions_v2/worker_0_local.parquet `
  --skip-source-analysis

uv run python -m crawler --run-name worker-0-local crawl run `
  --worker-id 0 `
  --domain-assignment partitions_v2/domain_assignment.parquet `
  --defer-domain ask.39.net
```

`ask.39.net` hiện chuyển hướng crawler sang trang ghép hình xác minh bot, nên lệnh trên giữ các URL của domain này ở trạng thái `PENDING` và tiếp tục crawl các domain khác. Khi có phương án truy cập hợp lệ, chạy lại cùng manifest nhưng bỏ `--defer-domain ask.39.net`.

Nếu bị ngắt, chạy lại đúng lệnh `crawl run` kèm `--worker-id 0`, `--domain-assignment` và `--defer-domain ask.39.net` để resume và giữ concurrency override của domain split. Kiểm tra trạng thái và corpus:

```powershell
uv run python -m crawler --run-name worker-0-local crawl status
uv run python -m crawler --run-name worker-0-local corpus verify
```

Output local nằm trong `data/crawl/runs/worker-0-local/`, với `crawl_manifest.sqlite` và `corpus/`. Không dùng lại manifest cũ của run khác.

## 7. Merge sáu worker về local

Giải nén Output tải từ Kaggle vào năm thư mục local riêng. Trong lệnh dưới, thay các đường dẫn `kaggle_outputs/worker_N/crawl_worker_N` bằng thư mục thực tế chứa trực tiếp `manifest.sqlite` và `corpus/`:

```powershell
uv run python merge_crawl_workers.py `
  --input "0=data/crawl/runs/worker-0-local" `
  --input "1=kaggle_outputs/worker_1/crawl_worker_1" `
  --input "2=kaggle_outputs/worker_2/crawl_worker_2" `
  --input "3=kaggle_outputs/worker_3/crawl_worker_3" `
  --input "4=kaggle_outputs/worker_4/crawl_worker_4" `
  --input "5=kaggle_outputs/worker_5/crawl_worker_5" `
  --output-dir data/crawl/merged
```

Mặc định merge yêu cầu cả sáu manifest đã commit mọi `doc_id`. Để tạo một snapshot tạm khi worker vẫn còn `PENDING`, thêm `--allow-partial` và dùng một `--output-dir` mới.

Merge đọc Parquet theo batch, xác minh URL gốc với từng manifest, chọn một record duy nhất cho mỗi `doc_id` và giữ nguyên tất cả field của record đó. Thứ tự ưu tiên là `SUCCESS > EMPTY_CONTENT > NEEDS_OCR > NEEDS_JS > TOO_LARGE > UNSUPPORTED > FAILED > ROBOTS_DENIED`. Trùng ID được ghi vào `merge_conflicts.jsonl`; URL gốc khác nhau cho cùng ID làm merge dừng. Output gồm `corpus/part-*.parquet` và `merge_summary.json`. Script từ chối ghi đè thư mục output đã có.
