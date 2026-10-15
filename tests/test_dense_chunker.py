"""Unit tests for SentenceAwareChunker and multilingual sentence splitting.

Tests all edge cases: short docs, missing metadata, CJK, long sentences,
long titles, token budgets, overlap, abbreviations, and deterministic IDs.
"""

from retrieval.dense.chunker import (
    ChunkerRegistry,
    SentenceAwareChunker,
    split_sentences_multilingual,
)
from retrieval.dense.config import ChunkingConfig
from retrieval.dense.tokenizer_helper import FallbackTokenCounter


def test_split_sentences_abbreviations_and_decimals():
    """Ensure medical abbreviations, decimal numbers, and units are not split falsely."""
    text = (
        "Bệnh nhân sốt cao 39.5 độ C. "
        "Uống paracetamol 500 mg theo chỉ định của BS. Hùng và TS. Lan. "
        "Bệnh viện đa khoa quốc tế khám lúc 8.30 sáng. "
        "Các triệu chứng gồm ho, sốt, v.v. cần theo dõi."
    )
    sents = split_sentences_multilingual(text)

    assert len(sents) >= 3
    # Check decimal 39.5 not broken
    assert any("39.5 độ C" in s for s in sents)
    # Check BS. Hùng and TS. Lan not broken
    assert any("BS. Hùng" in s for s in sents)
    assert any("TS. Lan" in s for s in sents)
    # Check v.v. not broken
    assert any("v.v." in s for s in sents)


def test_split_sentences_chinese_cjk():
    """Ensure Chinese text is properly split on CJK punctuation without whitespace."""
    text = "外阴炎常表现为局部红肿和瘙痒。建议患者及时前往正规医院就诊！避免辛辣刺激性食物；保持局部清洁干燥。"
    sents = split_sentences_multilingual(text)

    assert len(sents) == 4
    assert sents[0] == "外阴炎常表现为局部红肿和瘙痒。"
    assert sents[1] == "建议患者及时前往正规医院就诊！"
    assert sents[2] == "避免辛辣刺激性食物；"
    assert sents[3] == "保持局部清洁干燥。"


def test_chunker_short_document():
    """A short document should produce exactly one chunk with chunk_index=0."""
    cfg = ChunkingConfig(target_tokens=400, max_total_tokens=512)
    chunker = SentenceAwareChunker(cfg, token_counter=FallbackTokenCounter())

    doc_id = 12345
    title = "Cảm cúm thông thường"
    text = "Cảm cúm là bệnh truyền nhiễm do virus gây ra. Nghỉ ngơi và uống nhiều nước."

    chunks = chunker.chunk_document(doc_id, title=title, text=text)

    assert len(chunks) == 1
    c = chunks[0]
    assert c.doc_id == doc_id
    assert c.chunk_index == 0
    assert c.chunk_id == f"{doc_id}_c0_a1"
    assert c.title == title
    assert "Title: Cảm cúm thông thường" in c.embed_text
    assert c.text == text
    assert c.token_count > 0


def test_chunker_missing_title_and_heading():
    """Document without title or heading should chunk cleanly without Title prefix."""
    cfg = ChunkingConfig(target_tokens=400, max_total_tokens=512)
    chunker = SentenceAwareChunker(cfg, token_counter=FallbackTokenCounter())

    doc_id = 999
    text = "Bài viết y khoa không có tiêu đề. Triệu chứng lâm sàng bao gồm đau đầu và mệt mỏi."

    chunks = chunker.chunk_document(doc_id, title=None, text=text, heading=None)

    assert len(chunks) == 1
    c = chunks[0]
    assert c.title is None
    assert c.heading is None
    assert "Title:" not in c.embed_text
    assert c.embed_text == text
    assert c.text == text


def test_chunker_excessively_long_title_capped():
    """An excessively long title must be capped so it does not starve body token budget."""
    cfg = ChunkingConfig(target_tokens=400, max_total_tokens=512, max_title_tokens=20)
    chunker = SentenceAwareChunker(cfg, token_counter=FallbackTokenCounter())

    long_title = "Tiêu đề " + "rất dài ".join([f"từ_{i}" for i in range(100)])
    body_text = "Nội dung bài viết y tế quan trọng."

    chunks = chunker.chunk_document(doc_id=1, title=long_title, text=body_text)

    assert len(chunks) >= 1
    c = chunks[0]
    # Check that embed_text contains body
    assert body_text in c.embed_text
    # Check that title tokens inside embed prefix were bounded
    assert len(c.embed_text) < len(long_title) + len(body_text)


def test_chunker_sentence_exceeding_budget():
    """A single sentence larger than max_total_tokens must be split by token window fallback."""
    cfg = ChunkingConfig(target_tokens=100, max_total_tokens=150, overlap_tokens=20)
    chunker = SentenceAwareChunker(cfg, token_counter=FallbackTokenCounter())

    # Build a single sentence with 300 words without any period
    giant_sentence = "Bệnh nhân có biểu hiện " + " ".join([f"triệu_chứng_{i}" for i in range(250)])

    chunks = chunker.chunk_document(doc_id=777, title="Ca bệnh khó", text=giant_sentence)

    # Must be split into multiple chunks
    assert len(chunks) >= 2
    for idx, c in enumerate(chunks):
        assert c.doc_id == 777
        assert c.chunk_index == idx
        assert c.chunk_id == f"777_c{idx}_a1"
        # Each chunk must respect token budget
        assert c.token_count <= 180  # bounded by max_total_tokens + margin


def test_chunker_preserves_content_no_body_loss():
    """When a multi-sentence document is split into multiple chunks, all text content must be retained."""
    cfg = ChunkingConfig(target_tokens=50, max_total_tokens=80, overlap_tokens=0)
    chunker = SentenceAwareChunker(cfg, token_counter=FallbackTokenCounter())

    sentences = [
        f"Câu số {i}: Thông tin y khoa cụ thể và chi tiết về phác đồ điều trị bệnh số {i}."
        for i in range(10)
    ]
    full_text = " ".join(sentences)

    chunks = chunker.chunk_document(doc_id=888, title="Phác đồ", text=full_text)

    assert len(chunks) > 1
    # Check that every sentence appears in at least one chunk
    for sent in sentences:
        assert any(sent in c.text for c in chunks), f"Missing sentence in chunks: {sent}"


def test_chunker_deterministic_ids():
    """Running chunker twice on the same document must produce identical chunk_ids."""
    cfg = ChunkingConfig(chunking_version="a1")
    chunker = SentenceAwareChunker(cfg, token_counter=FallbackTokenCounter())

    doc_id = 555
    text = "Câu một. Câu hai. Câu ba. Câu bốn."

    run1 = chunker.chunk_document(doc_id, title="Test", text=text)
    run2 = chunker.chunk_document(doc_id, title="Test", text=text)

    assert len(run1) == len(run2)
    for c1, c2 in zip(run1, run2, strict=False):
        assert c1.chunk_id == c2.chunk_id
        assert c1.doc_id == c2.doc_id == doc_id
        assert c1.chunk_index == c2.chunk_index
        assert c1.text == c2.text


def test_chunker_registry_extensibility():
    """Registry must instantiate registered strategies and raise error for unknown ones."""
    cfg_a = ChunkingConfig(strategy="sentence_aware")
    chunker_a = ChunkerRegistry.get(cfg_a)
    assert isinstance(chunker_a, SentenceAwareChunker)

    # Unknown strategy should raise ValueError
    cfg_unknown = ChunkingConfig(strategy="non_existent_strategy")
    try:
        ChunkerRegistry.get(cfg_unknown)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown chunking strategy" in str(e)
