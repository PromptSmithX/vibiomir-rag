from retrieval.tokenizer import MultilingualMedicalTokenizer


def test_tokenizer_vietnamese_medical() -> None:
    tokenizer = MultilingualMedicalTokenizer(expand_compounds=False)
    text = "Bệnh nhân nam 45 tuổi, chẩn đoán viêm xung huyết dạ dày."
    tokens = tokenizer.tokenize(text)
    assert "bệnh" in tokens
    assert "nhân" in tokens
    assert "viêm" in tokens
    assert "dạ" in tokens
    assert "dày" in tokens
    assert "45" in tokens


def test_tokenizer_chinese_cjk() -> None:
    tokenizer = MultilingualMedicalTokenizer(expand_compounds=False)
    text = "急性扁桃体炎 咽喉肿痛 39.net"
    tokens = tokenizer.tokenize(text)
    assert "急性" in tokens
    assert "扁桃体炎" in tokens
    assert "咽喉" in tokens or "咽喉肿痛" in tokens


def test_tokenizer_icd_and_drug_names() -> None:
    tokenizer = MultilingualMedicalTokenizer(expand_compounds=True)
    text = "Chẩn đoán C00.0 và J45.9. Kê Paracetamol 500mg, liều 10mg/kg."
    tokens = tokenizer.tokenize(text)
    assert "c00.0" in tokens
    assert "c00" in tokens
    assert "0" in tokens
    assert "j45.9" in tokens
    assert "paracetamol" in tokens
    assert "500mg" in tokens
    assert "10mg" in tokens
    assert "kg" in tokens


def test_tokenizer_numbers_and_fractions() -> None:
    tokenizer = MultilingualMedicalTokenizer(expand_compounds=False)
    text = "Viêm 1/3 dưới thực quản, chỉ số Beta-HCG là 0.10 U/L."
    tokens = tokenizer.tokenize(text)
    assert "1" in tokens
    assert "3" in tokens
    assert "beta-hcg" in tokens or "beta" in tokens
    assert "0.10" in tokens
    assert "u" in tokens
    assert "l" in tokens


def test_query_preparation() -> None:
    tokenizer = MultilingualMedicalTokenizer(expand_compounds=True)
    q = "Trị số Beta-HCG và C00.0 là bao nhiêu?"
    q_str = tokenizer.build_query_string(q)
    assert r"beta\-hcg" in q_str or "beta" in q_str
    assert "c00.0" in q_str or "c00" in q_str
    assert "trị" in q_str
