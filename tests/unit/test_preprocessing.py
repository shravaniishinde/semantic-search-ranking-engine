from app.preprocessing.text import clean_text, is_meaningful, tokenize


def test_clean_text_strips_html_urls_and_whitespace():
    raw = "<p>Hello&nbsp;<b>World</b></p>\n\n see https://example.com/x   now\x07"
    assert clean_text(raw) == "Hello World see now"


def test_clean_text_preserves_case_punctuation_and_stopwords():
    text = "What is the CAP theorem? It's about consistency, availability & partitions."
    assert clean_text(text) == text


def test_clean_text_collapses_repeated_punctuation_and_is_deterministic():
    assert clean_text("Wow!!!!!!! ok") == "Wow! ok"
    assert clean_text("  a  b ") == clean_text("  a  b ") == "a b"


def test_clean_text_handles_empty_and_none():
    assert clean_text(None) == ""
    assert clean_text("   \n\t ") == ""


def test_is_meaningful_rejects_blank_and_tiny_content():
    assert not is_meaningful("")
    assert not is_meaningful("... !!! ---")
    assert is_meaningful("A real sentence about databases.")


def test_tokenize_keeps_technical_tokens_intact():
    tokens = tokenize("Patch CVE-2021-44228 in Log4j 2.17 and node.js")
    assert "cve-2021-44228" in tokens
    assert "2.17" in tokens
    assert "node.js" in tokens


def test_tokenize_removes_stopwords_and_light_stems():
    assert tokenize("What are the databases and queries") == ["database", "query"]
    # Words ending in ss/us/is are not truncated.
    assert tokenize("access status analysis") == ["access", "status", "analysis"]


def test_tokenize_without_stopword_removal_keeps_everything():
    assert tokenize("how to fix", remove_stopwords=False, stem=False) == ["how", "to", "fix"]
