import html
import json
from collections.abc import Sequence
from pathlib import Path

from curation.audit.sampler import AuditSamplePair


class AuditHtmlPresenter:
    """Generates an interactive, standalone HTML audit report for curated documents."""

    def __init__(self, samples: Sequence[AuditSamplePair]) -> None:
        self.samples = list(samples)

    def render_to_file(self, output_path: Path | str) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        html_content = self.render()
        out.write_text(html_content, encoding="utf-8")
        return out

    def render(self) -> str:
        # Precompute summary statistics
        total_samples = len(self.samples)
        domains = sorted({s.domain for s in self.samples})
        total_raw = sum(s.raw_chars for s in self.samples)
        total_clean = sum(s.clean_chars for s in self.samples)
        total_removed = sum(s.removed_chars for s in self.samples)
        pct_removed = (total_removed / total_raw * 100) if total_raw > 0 else 0.0

        # Prepare JSON payload safely
        payload = [s.to_dict() for s in self.samples]
        json_data = json.dumps(payload, ensure_ascii=False)

        # Build domain filter options
        domain_buttons = ['<button class="domain-btn active" data-domain="ALL">Tất cả</button>']
        for dom in domains:
            esc = html.escape(dom)
            domain_buttons.append(f'<button class="domain-btn" data-domain="{esc}">{esc}</button>')
        domain_buttons_html = "\n".join(domain_buttons)

        return f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ViBioMIR — Curation Audit Dashboard</title>
<style>
  :root {{
    --bg: #0f172a;
    --card-bg: #1e293b;
    --card-hover: #334155;
    --border: #334155;
    --text-primary: #f8fafc;
    --text-muted: #94a3b8;
    --accent: #38bdf8;
    --accent-dark: #0284c7;
    --danger: #f43f5e;
    --danger-bg: rgba(244, 63, 94, 0.1);
    --success: #10b981;
    --success-bg: rgba(16, 185, 129, 0.1);
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text-primary);
    line-height: 1.6;
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }}
  header {{
    background: var(--card-bg);
    border-bottom: 1px solid var(--border);
    padding: 12px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-shrink: 0;
  }}
  header h1 {{
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--accent);
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .stat-badges {{
    display: flex;
    gap: 12px;
  }}
  .badge {{
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.8rem;
  }}
  .badge span {{ font-weight: 700; color: var(--accent); }}
  .badge.success span {{ color: var(--success); }}
  .badge.danger span {{ color: var(--danger); }}
  
  .controls-bar {{
    background: #182234;
    border-bottom: 1px solid var(--border);
    padding: 10px 24px;
    display: flex;
    gap: 16px;
    align-items: center;
    flex-shrink: 0;
  }}
  .search-box {{
    flex: 1;
    position: relative;
    max-width: 600px;
  }}
  .search-box input {{
    width: 100%;
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 14px;
    color: var(--text-primary);
    font-size: 0.9rem;
    outline: none;
    transition: border 0.2s;
  }}
  .search-box input:focus {{
    border-color: var(--accent);
  }}
  .domain-filters {{
    display: flex;
    gap: 6px;
    overflow-x: auto;
    padding-bottom: 2px;
  }}
  .domain-btn {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    color: var(--text-muted);
    border-radius: 4px;
    padding: 5px 10px;
    font-size: 0.78rem;
    cursor: pointer;
    white-space: nowrap;
    transition: all 0.2s;
  }}
  .domain-btn:hover {{
    background: var(--card-hover);
    color: var(--text-primary);
  }}
  .domain-btn.active {{
    background: var(--accent-dark);
    color: white;
    border-color: var(--accent);
  }}

  .main-layout {{
    flex: 1;
    display: flex;
    overflow: hidden;
  }}
  .sidebar {{
    width: 380px;
    border-right: 1px solid var(--border);
    background: #131c2e;
    overflow-y: auto;
    flex-shrink: 0;
  }}
  .doc-item {{
    padding: 12px 16px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    cursor: pointer;
    transition: background 0.15s;
  }}
  .doc-item:hover {{
    background: rgba(255, 255, 255, 0.03);
  }}
  .doc-item.active {{
    background: rgba(56, 189, 248, 0.1);
    border-left: 3px solid var(--accent);
  }}
  .doc-title {{
    font-size: 0.88rem;
    font-weight: 600;
    margin-bottom: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .doc-meta {{
    font-size: 0.75rem;
    color: var(--text-muted);
    display: flex;
    justify-content: space-between;
  }}
  .tag-domain {{
    background: rgba(255, 255, 255, 0.08);
    padding: 1px 6px;
    border-radius: 3px;
    font-family: monospace;
  }}
  .tag-removed {{
    color: var(--danger);
    font-weight: 600;
  }}

  .viewer {{
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    background: var(--bg);
  }}
  .viewer-header {{
    padding: 14px 24px;
    background: var(--card-bg);
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .viewer-title {{
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 4px;
  }}
  .viewer-url {{
    font-size: 0.8rem;
    color: var(--accent);
    text-decoration: none;
    word-break: break-all;
  }}
  .viewer-url:hover {{ text-decoration: underline; }}
  
  .diff-container {{
    flex: 1;
    display: flex;
    overflow: hidden;
  }}
  .pane {{
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    border-right: 1px solid var(--border);
  }}
  .pane:last-child {{ border-right: none; }}
  .pane-header {{
    padding: 8px 16px;
    background: #152033;
    border-bottom: 1px solid var(--border);
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    display: flex;
    justify-content: space-between;
  }}
  .pane-header.raw {{ color: #fb7185; }}
  .pane-header.clean {{ color: #34d399; }}
  .pane-body {{
    flex: 1;
    padding: 20px;
    overflow-y: auto;
    font-size: 0.92rem;
    white-space: pre-wrap;
    font-family: "Consolas", "Monaco", "Courier New", monospace;
    background: #0b1120;
    line-height: 1.65;
  }}
  .clean-pane-body {{
    background: #0d1527;
  }}
  .no-selection {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: var(--text-muted);
  }}
</style>
</head>
<body>

<header>
  <h1>🔬 ViBioMIR Curation Audit Dashboard</h1>
  <div class="stat-badges">
    <div class="badge">Đã lấy mẫu: <span id="stat-samples">{total_samples}</span> bài</div>
    <div class="badge">Số domain: <span>{len(domains)}</span></div>
    <div class="badge danger">
      Rác gọt: <span>{total_removed:,} chars ({pct_removed:.2f}%)</span>
    </div>
    <div class="badge success">Text sạch: <span>{total_clean:,} chars</span></div>
  </div>
</header>

<div class="controls-bar">
  <div class="search-box">
    <input
      type="text"
      id="search-input"
      placeholder="🔍 Nhập hoặc dán link URL, Doc ID hoặc Tiêu đề..."
      autofocus
    />
  </div>
  <div class="domain-filters" id="domain-filters">
    {domain_buttons_html}
  </div>
</div>

<div class="main-layout">
  <div class="sidebar" id="doc-list">
    <!-- Populated by JS -->
  </div>
  <div class="viewer" id="doc-viewer">
    <div class="no-selection" id="empty-state">
      <h3>Chọn một bài viết ở danh sách bên trái hoặc dán URL vào ô tìm kiếm</h3>
    </div>
    <div id="active-viewer" style="display: none; height: 100%; flex-direction: column;">
      <div class="viewer-header">
        <div>
          <div class="viewer-title" id="view-title"></div>
          <a class="viewer-url" id="view-url" href="" target="_blank"></a>
        </div>
        <div style="text-align: right; font-size: 0.85rem;">
          <div>Doc ID: <strong id="view-doc-id" style="color: var(--accent);"></strong></div>
          <div style="color: var(--danger); font-weight: 600;" id="view-removed"></div>
        </div>
      </div>
      <div class="diff-container">
        <div class="pane">
          <div class="pane-header raw">
            <span>🔴 RAW CRAWLED TEXT</span>
            <span id="view-raw-chars"></span>
          </div>
          <div class="pane-body" id="view-raw-text"></div>
        </div>
        <div class="pane">
          <div class="pane-header clean">
            <span>🟢 CLEAN CURATED TEXT</span>
            <span id="view-clean-chars"></span>
          </div>
          <div class="pane-body clean-pane-body" id="view-clean-text"></div>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
  const DOCS = {json_data};
  let activeDomain = "ALL";
  let activeIndex = 0;
  let filteredDocs = [...DOCS];

  const searchInput = document.getElementById("search-input");
  const docList = document.getElementById("doc-list");
  const emptyState = document.getElementById("empty-state");
  const activeViewer = document.getElementById("active-viewer");
  const domainBtns = document.querySelectorAll(".domain-btn");

  function filterDocs() {{
    const query = searchInput.value.trim().toLowerCase();
    filteredDocs = DOCS.filter(doc => {{
      const matchDomain = (activeDomain === "ALL" || doc.domain === activeDomain);
      if (!matchDomain) return false;
      if (!query) return true;
      return doc.url.toLowerCase().includes(query) ||
             doc.doc_id.toString().includes(query) ||
             (doc.title && doc.title.toLowerCase().includes(query));
    }});
    renderList();
    if (filteredDocs.length > 0) {{
      selectDoc(0);
    }} else {{
      emptyState.style.display = "flex";
      activeViewer.style.display = "none";
    }}
  }}

  function renderList() {{
    docList.innerHTML = "";
    filteredDocs.forEach((doc, idx) => {{
      const item = document.createElement("div");
      const remTag = `-${{doc.removed_chars.toLocaleString()}} (-${{doc.removed_pct}}%)`;
      item.innerHTML = `
        <div class="doc-title">${{doc.title ? doc.title : doc.url}}</div>
        <div class="doc-meta">
          <span class="tag-domain">${{doc.domain}}</span>
          <span class="tag-removed">${{remTag}}</span>
        </div>
      `;
      item.onclick = () => selectDoc(idx);
      docList.appendChild(item);
    }});
  }}

  function selectDoc(idx) {{
    activeIndex = idx;
    const doc = filteredDocs[idx];
    if (!doc) return;

    // Highlight active in list
    document.querySelectorAll(".doc-item").forEach((el, i) => {{
      el.classList.toggle("active", i === idx);
    }});

    emptyState.style.display = "none";
    activeViewer.style.display = "flex";

    document.getElementById("view-title").textContent = doc.title || "No Title";
    const urlEl = document.getElementById("view-url");
    urlEl.textContent = doc.url;
    urlEl.href = doc.url;
    document.getElementById("view-doc-id").textContent = doc.doc_id;

    const remCount = doc.removed_chars.toLocaleString();
    const remInfo = `Loại bỏ: -${{remCount}} ký tự (-${{doc.removed_pct}}%)`;
    document.getElementById("view-removed").textContent = remInfo;

    const rawCharsStr = `${{doc.raw_chars.toLocaleString()}} chars`;
    document.getElementById("view-raw-chars").textContent = rawCharsStr;
    const cleanCharsStr = `${{doc.clean_chars.toLocaleString()}} chars`;
    document.getElementById("view-clean-chars").textContent = cleanCharsStr;

    document.getElementById("view-raw-text").textContent = doc.raw_text;
    document.getElementById("view-clean-text").textContent = doc.clean_text;
  }}

  searchInput.addEventListener("input", filterDocs);

  domainBtns.forEach(btn => {{
    btn.addEventListener("click", () => {{
      domainBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      activeDomain = btn.getAttribute("data-domain");
      filterDocs();
    }});
  }});

  // Initial render
  filterDocs();
</script>
</body>
</html>"""
