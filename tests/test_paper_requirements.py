from sophia.paper_requirements import check_paper_requirements, is_paper_request


def test_detects_paper_request():
    assert is_paper_request("请写一篇论文")
    assert is_paper_request("write a paper about AI")


def test_requires_type_and_word_count_before_writing():
    result = check_paper_requirements("请写一篇生成式人工智能论文")

    assert result.is_paper_request
    assert result.requires_clarification
    assert "论文类型" in result.message
    assert "目标正文字数" in result.message
    assert "理论论文" in result.message
    assert "实证论文" in result.message


def test_allows_complete_paper_requirements():
    result = check_paper_requirements("请写一篇理论论文，正文不少于 8000 字")

    assert result.is_paper_request
    assert not result.requires_clarification


def test_allows_explicit_default_bypass():
    result = check_paper_requirements("请写一篇论文，按默认直接写")

    assert result.is_paper_request
    assert not result.requires_clarification
