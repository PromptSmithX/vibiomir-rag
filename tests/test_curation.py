import tempfile
import unicodedata

from curation.models import RawDoc
from curation.pipeline import CurationPipeline
from curation.reader import ParquetCorpusReader
from curation.rules.baidu import BaiduHealthRule
from curation.rules.baoquangninh import BaoQuangNinhRule
from curation.rules.generic import GenericDomainRule
from curation.rules.hanoimoi import HanoiMoiRule
from curation.rules.longchau import LongChauRule
from curation.rules.net39_article import Net39ArticleRule
from curation.rules.net39_qa import Ask39QARule
from curation.rules.nhandan import NhanDanRule
from curation.rules.registry import RuleRegistry
from curation.rules.tamanh import TamAnhHospitalRule
from curation.rules.vietnamnet import VietNamNetRule
from curation.rules.vov2 import VOV2Rule
from curation.stages.markdown_fmt import MarkdownFormatStage
from curation.stages.quality_gate import QualityGateStage
from curation.stages.unicode_norm import UnicodeNormalizeStage
from curation.writer import ParquetCorpusWriter


def test_unicode_normalize_stage():
    stage = UnicodeNormalizeStage()

    # Composed vs decomposed
    decomposed = unicodedata.normalize("NFD", "Bệnh viện đa khoa quốc tế")
    raw_text = f"{decomposed}\u200b có nhiều khoa.\u00a0Bác sĩ giỏi.\r\nĐiều trị tốt.\ufeff"

    doc = RawDoc(doc_id=1, url="http://test.com", text=raw_text)
    processed = stage.process(doc)

    assert "\u200b" not in processed.text
    assert "\ufeff" not in processed.text
    assert "\u00a0" not in processed.text
    assert "\r\n" not in processed.text
    assert unicodedata.is_normalized("NFC", processed.text)
    assert "Bệnh viện đa khoa quốc tế" in processed.text


def test_tamanh_rule():
    rule = TamAnhHospitalRule()
    raw = (
        "# Triệu chứng viêm gan B\n\n"
        "Bệnh viêm gan B do virus HBV gây ra.\n\n"
        "## ĐỐI TÁC BẢO HIỂM\n"
        "Bước 1: Bấm vào nút ‘Thêm Bệnh viện Đa khoa Tâm Anh trên Google’\n"
        "Bước 2: Chọn làm nguồn tìm kiếm ưu tiên.\n"
        "Khi ô vuông chuyển thành dấu tích xanh là hoàn thành."
    )
    cleaned = rule.clean(raw)
    assert "# Triệu chứng viêm gan B" in cleaned
    assert "Bệnh viêm gan B do virus HBV gây ra." in cleaned
    assert "ĐỐI TÁC BẢO HIỂM" not in cleaned
    assert "Bước 1" not in cleaned
    assert "dấu tích xanh" not in cleaned


def test_longchau_rule():
    rule = LongChauRule()
    raw = (
        "# Hướng dẫn điều trị ngộ độc paracetamol\n\n"
        "Paracetamol là thuốc giảm đau phổ biến.\n\n"
        "### Sinh thiết phôi là gì? Quy trình và những điều cần biết\n"
        "0 lượt xem\n"
        "### U xơ tử cung có làm IVF được không?\n"
        "0 lượt xem"
    )
    cleaned = rule.clean(raw)
    assert "# Hướng dẫn điều trị ngộ độc paracetamol" in cleaned
    assert "Paracetamol là thuốc giảm đau phổ biến." in cleaned
    assert "Sinh thiết phôi" not in cleaned
    assert "0 lượt xem" not in cleaned


def test_hanoimoi_rule():
    rule = HanoiMoiRule()
    raw = (
        "# Phân cấp lĩnh vực y tế\n\n"
        "Bộ trưởng Bộ Y tế Đào Hồng Lan phát biểu tại hội nghị.\n\n"
        "- Đào Hồng Lan\n"
        "- Trách nhiệm\n"
        "(*) Không sao chép dưới mọi hình thức khi chưa có sự đồng ý bằng văn bản "
        "của Cơ quan Báo và Phát thanh, truyền hình Hà Nội."
    )
    cleaned = rule.clean(raw)
    assert "# Phân cấp lĩnh vực y tế" in cleaned
    assert "Bộ trưởng Bộ Y tế Đào Hồng Lan phát biểu" in cleaned
    assert "Không sao chép dưới mọi hình thức" not in cleaned
    assert "- Đào Hồng Lan" not in cleaned


def test_vietnamnet_rule():
    rule = VietNamNetRule()
    raw = (
        "- Tuần Việt Nam\n"
        "- Sức khỏe\n"
        "# Phát hiện ca nhiễm mới\n\n"
        "Bệnh nhân được cách ly theo dõi y tế.\n\n"
        "## Tin cùng chuyên mục\n"
        "- Bài viết 1\n"
        "- Bài viết 2"
    )
    cleaned = rule.clean(raw)
    assert cleaned.startswith("# Phát hiện ca nhiễm mới")
    assert "- Tuần Việt Nam" not in cleaned
    assert "Tin cùng chuyên mục" not in cleaned
    assert "Bệnh nhân được cách ly theo dõi y tế." in cleaned


def test_vov2_rule():
    rule = VOV2Rule()
    raw = (
        "# Tạm ngưng hoạt động bệnh viện dã chiến\n\n"
        "Tình hình dịch bệnh đã được kiểm soát tốt.\n\n"
        "#### Việt Nam trắng tay trước Malaysia, khép lại FIFA ASEAN Cup 2026\n"
        "#### Phân cấp thủ tục hành chính trong lĩnh vực xuất nhập khẩu\n"
        "#### Văn hóa đọc: Nền tảng xây dựng xã hội học tập"
    )
    cleaned = rule.clean(raw)
    assert "# Tạm ngưng hoạt động bệnh viện dã chiến" in cleaned
    assert "Tình hình dịch bệnh đã được kiểm soát tốt." in cleaned
    assert "FIFA ASEAN Cup" not in cleaned
    assert "Văn hóa đọc" not in cleaned


def test_baoquangninh_rule():
    rule = BaoQuangNinhRule()
    raw = (
        "# Sức khỏe bệnh nhân bạch hầu ổn định\n\n"
        "CDC ghi nhận bệnh nhân đã hồi phục tốt.\n\n"
        "**Hoàng Quý - Hùng Sơn**\n"
        "### từ khóa:\n"
        "- bạch hầu\n"
        "- CDC Quảng Ninh"
    )
    cleaned = rule.clean(raw)
    assert "# Sức khỏe bệnh nhân bạch hầu ổn định" in cleaned
    assert "CDC ghi nhận bệnh nhân đã hồi phục tốt." in cleaned
    assert "Hoàng Quý - Hùng Sơn" not in cleaned
    assert "từ khóa" not in cleaned


def test_nhandan_rule():
    rule = NhanDanRule()
    raw = (
        "# Nỗ lực phòng chống dịch sốt xuất huyết\n\n"
        "Các ổ dịch đang được phun thuốc khử khuẩn triệt để.\n\n"
        "Theo Báo Nhân Dân\n"
        "## Tin liên quan\n"
        "- Tin 1"
    )
    cleaned = rule.clean(raw)
    assert "# Nỗ lực phòng chống dịch sốt xuất huyết" in cleaned
    assert "Các ổ dịch đang được phun thuốc" in cleaned
    assert "Theo Báo Nhân Dân" not in cleaned
    assert "Tin liên quan" not in cleaned


def test_net39_qa_rule():
    rule = Ask39QARule()
    raw = (
        "# 外阴炎症状有哪些\n\n"
        "## 医生回答\n"
        "外阴炎常表现为局部红肿和瘙痒。\n\n"
        "- 就医助手\n"
        "- 药品通\n"
        "- 疾病百科\n"
        "- 药品"
    )
    cleaned = rule.clean(raw)
    assert "# 外阴炎症状有哪些" in cleaned
    assert "外阴炎常表现为局部红肿和瘙痒。" in cleaned
    assert "就医助手" not in cleaned
    assert "药品通" not in cleaned


def test_net39_article_rule():
    rule = Net39ArticleRule()
    raw = (
        "# 养血 女性漂亮的关键所在\n\n"
        "血为人之本，女性应多调理气血。\n\n"
        "- 开滦嘉盛医院 二级甲等 综合医院 公立\n"
        "- 首都医科大学附属北京妇产医院 三级甲等 专科医院 公立"
    )
    cleaned = rule.clean(raw)
    assert "# 养血 女性漂亮的关键所在" in cleaned
    assert "血为人之本，女性应多调理气血。" in cleaned
    assert "开滦嘉盛医院" not in cleaned
    assert "北京妇产医院" not in cleaned


def test_baidu_rule():
    rule = BaiduHealthRule()
    raw = (
        "百度健康医典原创内容，三审三校\n"
        "- 概述\n"
        "- 病因\n"
        "# 玻璃体混浊\n\n"
        "玻璃体混浊是指玻璃体透明度降低。\n"
        "建议每年做一次眼科检查。"
    )
    cleaned = rule.clean(raw)
    assert "三审三校" not in cleaned
    assert "- 概述" not in cleaned
    assert "# 玻璃体混浊" in cleaned
    assert "玻璃体混浊是指玻璃体透明度降低。" in cleaned


def test_generic_rule_fallback():
    rule = GenericDomainRule()
    raw = (
        "# Bài viết y khoa trên trang lạ\n\n"
        "Nội dung phân tích phác đồ điều trị bệnh.\n\n"
        "## Tin cùng chuyên mục\n"
        "Nội dung không liên quan."
    )
    cleaned = rule.clean(raw)
    assert "# Bài viết y khoa trên trang lạ" in cleaned
    assert "Nội dung phân tích phác đồ điều trị bệnh." in cleaned
    assert "Tin cùng chuyên mục" not in cleaned


def test_rule_registry():
    registry = RuleRegistry()
    assert isinstance(registry.get("tamanhhospital.vn"), TamAnhHospitalRule)
    assert isinstance(registry.get("tiemchunglongchau.com.vn"), LongChauRule)
    assert isinstance(registry.get("ask.39.net"), Ask39QARule)
    assert isinstance(registry.get("woman.39.net"), Net39ArticleRule)
    assert isinstance(registry.get("pf.39.net"), Net39ArticleRule)
    assert isinstance(registry.get("unknown-medical-blog.com"), GenericDomainRule)


def test_markdown_format_stage():
    stage = MarkdownFormatStage()
    raw = "Đoạn 1.\n\n\n\n\nĐoạn 2.\n\n\nĐoạn 3."
    doc = RawDoc(doc_id=1, url="http://test.com", text=raw)
    processed = stage.process(doc)
    assert processed.text == "Đoạn 1.\n\nĐoạn 2.\n\nĐoạn 3."


def test_quality_gate_stage():
    gate = QualityGateStage(min_chars=50)

    good_doc = RawDoc(
        doc_id=1,
        url="http://test.com/1",
        text=(
            "Đây là nội dung y khoa có độ dài đầy đủ vượt qua "
            "ngưỡng 50 ký tự kiểm định chất lượng."
        ),
        domain="test.com",
    )
    curated_good = gate.process(good_doc)
    assert curated_good.curation_status == "CLEAN"
    assert curated_good.text_chars >= 50
    assert len(curated_good.content_hash) == 64

    short_doc = RawDoc(
        doc_id=2,
        url="http://test.com/2",
        text="Quá ngắn",
        domain="test.com",
    )
    curated_short = gate.process(short_doc)
    assert curated_short.curation_status == "LOW_QUALITY"


def test_end_to_end_pipeline():
    pipeline = CurationPipeline()

    raw_text = (
        "## ĐỐI TÁC BẢO HIỂM\n"  # Will be stripped by TamAnh rule
        if False
        else (
            "# Triệu chứng viêm phổi\n\n"
            "Bệnh nhân có biểu hiện sốt cao,\u00a0khó thở,\u200b đau ngực.\n\n\n\n"
            "Cần đưa đến cơ sở y tế ngay lập tức.\n\n"
            "## ĐỐI TÁC BẢO HIỂM\n"
            "Bước 1: Bấm vào nút ‘Thêm BV’..."
        )
    )

    doc = RawDoc(
        doc_id=101,
        url="https://tamanhhospital.vn/viem-phoi",
        domain="tamanhhospital.vn",
        title="Triệu chứng viêm phổi",
        text=raw_text,
    )

    curated = pipeline.curate_doc(doc)
    assert curated.curation_status == "CLEAN"
    assert "ĐỐI TÁC BẢO HIỂM" not in curated.text
    assert "\u200b" not in curated.text
    assert "\u00a0" not in curated.text
    assert "\n\n\n" not in curated.text
    assert "Bệnh nhân có biểu hiện sốt cao" in curated.text
    assert curated.removed_chars > 0


def test_parquet_writer_and_reader():
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = ParquetCorpusWriter(tmpdir)
        pipeline = CurationPipeline()

        doc = RawDoc(
            doc_id=999,
            url="https://vietnamnet.vn/bai-viet-999",
            domain="vietnamnet.vn",
            title="Tin tức y tế",
            text=(
                "- Sức khỏe\n"
                "# Tin tức y tế\n\n"
                "Nội dung bài viết y tế quan trọng dài hơn năm mươi ký tự kiểm định.\n\n"
                "## Tin cùng chuyên mục\n- Rác"
            ),
        )

        curated = pipeline.curate_doc(doc)
        shard_path = writer.write_shard("part-000000.parquet", [curated])

        assert shard_path.exists()

        reader = ParquetCorpusReader(tmpdir, only_success=False)
        docs_read = reader.read_shard(shard_path)
        assert len(docs_read) == 1
        assert docs_read[0].doc_id == 999
        assert "Tin cùng chuyên mục" not in docs_read[0].text
