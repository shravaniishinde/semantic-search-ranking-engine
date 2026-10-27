import pytest

from app.classification.intent import INTENTS, IntentClassifier
from app.config.settings import PROJECT_ROOT

TRAINING = PROJECT_ROOT / "data" / "sample" / "intent_training.jsonl"


@pytest.fixture(scope="module")
def trained(ner):
    queries, labels = IntentClassifier.load_training_data(TRAINING)
    return IntentClassifier(ner).fit(queries, labels)


def test_training_data_covers_all_intents():
    _, labels = IntentClassifier.load_training_data(TRAINING)
    assert set(labels) == set(INTENTS)
    assert min(labels.count(i) for i in INTENTS) >= 20


def test_untrained_classifier_uses_rule_fallback(ner):
    clf = IntentClassifier(ner)
    assert not clf.is_trained
    pred = clf.predict("redis vs memcached")
    assert pred.source == "rule_fallback" and pred.label == "comparison" and pred.confidence == 0.0
    assert clf.predict("nginx error when restarting").label == "troubleshooting"
    assert clf.predict("Kubernetes").label == "entity_lookup"


def test_handcrafted_features_capture_cues(ner):
    clf = IntentClassifier(ner)
    names = IntentClassifier.HANDCRAFTED
    f = dict(zip(names, clf.handcrafted_features("What is the difference between AWS and Azure?")))
    assert f["starts_with_question_word"] == 1.0 and f["has_question_mark"] == 1.0
    assert f["cue_comparison"] > 0 and f["entity_count"] >= 2
    assert len(names) == len(clf.handcrafted_features("x"))


@pytest.mark.parametrize(
    "query, expected",
    [
        ("redis vs memcached", "comparison"),
        ("mysql replication error keeps failing", "troubleshooting"),
        ("what is a load balancer", "informational"),
        ("configure prometheus alerting rules for kubernetes", "technical"),
    ],
)
def test_trained_model_on_unseen_but_similar_queries(trained, query, expected):
    pred = trained.predict(query)
    assert pred.source == "xgboost"
    assert pred.label == expected


def test_probabilities_are_a_distribution(trained):
    pred = trained.predict("compare postgres and mongodb")
    assert set(pred.probabilities) == set(INTENTS)
    assert sum(pred.probabilities.values()) == pytest.approx(1.0, abs=1e-3)
    assert pred.confidence == max(pred.probabilities.values())


def test_save_and_load_roundtrip(trained, ner, tmp_path):
    path = tmp_path / "intent.joblib"
    trained.save(path)
    restored = IntentClassifier(ner)
    assert restored.load(path)
    assert restored.predict("tcp vs udp") == trained.predict("tcp vs udp")
    assert not IntentClassifier(ner).load(tmp_path / "missing.joblib")


def test_cannot_save_untrained(ner, tmp_path):
    with pytest.raises(RuntimeError):
        IntentClassifier(ner).save(tmp_path / "x.joblib")
