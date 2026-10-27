from app.nlp.ner import EntityExtractor, entity_keys


def _labels(entities):
    return {(e.text.lower(), e.label) for e in entities}


def test_query_ner_finds_aws(ner):
    entities = ner.extract("security architecture for AWS")
    assert ("aws", "ORG") in _labels(entities)


def test_document_ner_extracts_people_dates_and_products(ner):
    text = "Kent Beck popularised test-driven development. Kubernetes was open-sourced by Google in 2014."
    labels = _labels(ner.extract(text))
    assert ("kent beck", "PERSON") in labels
    assert ("kubernetes", "PRODUCT") in labels
    assert ("google", "ORG") in labels
    assert any(label == "DATE" for _, label in labels)


def test_cve_identifiers_are_extracted(ner):
    entities = ner.extract("Patch cve-2021-44228 now")
    assert ("cve-2021-44228", "VULNERABILITY_ID") in _labels(entities)
    # No competing statistical label for the same span.
    assert [e.label for e in ner.extract("CVE-2021-44228")] == ["VULNERABILITY_ID"]


def test_entities_are_deduplicated_and_have_offsets(ner):
    text = "AWS and AWS and AWS"
    entities = [e for e in ner.extract(text) if e.text == "AWS"]
    assert len(entities) == 1
    assert text[entities[0].start : entities[0].end] == "AWS"


def test_text_without_entities_returns_empty_list(ner):
    assert ner.extract("how do i make this faster") == []
    assert ner.extract("") == []


def test_batch_matches_single_extraction(ner):
    texts = ["Redis caching", "Terraform by HashiCorp"]
    assert [entity_keys(e) for e in ner.extract_batch(texts)] == [entity_keys(ner.extract(t)) for t in texts]


def test_gazetteer_fallback_when_spacy_model_missing():
    extractor = EntityExtractor("definitely_not_a_spacy_model")
    assert extractor.backend == "gazetteer"
    entities = extractor.extract("Deploy AWS Lambda with Terraform")
    assert _labels(entities) == {("aws lambda", "PRODUCT"), ("terraform", "PRODUCT")}  # longest match wins
