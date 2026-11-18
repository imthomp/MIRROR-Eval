"""Checks on the packaged AmanaMT v2 reliability table (no models or network needed)."""
from mirroreval.benchmarks.mqm.mqm_datasets import FLORES_LANG_CODES
from mirroreval.benchmarks.mqm.mqm_metrics import bleu_tokenizer
from mirroreval.benchmarks.mqm.mqm_reliability import (
    EVIDENCE_LEVELS,
    build_reliability_section,
    load_reliability,
    recommendation,
)

METRICS = ["bleu", "chrf", "comet"]


def test_every_supported_language_has_an_entry():
    table = load_reliability()["languages"]
    assert set(FLORES_LANG_CODES) <= set(table)


def test_entries_are_well_formed():
    for lang, entry in load_reliability()["languages"].items():
        assert entry["evidence"] in EVIDENCE_LEVELS, lang
        for m in METRICS:
            pooled = entry["metrics"][m]["pooled"]
            assert -1 <= pooled["spearman_r"] <= 1 and pooled["n"] > 0, (lang, m)
        if entry["evidence"] == "direct_en_to_x":
            assert entry["human_labels"] and "en_to_x" in entry["directions"], lang
            assert all("en_to_x" in entry["metrics"][m] for m in METRICS), lang
        if not entry["human_labels"]:
            assert entry["evidence"] == "none", lang


def test_section_and_recommendation_cover_requested_languages():
    langs = ["de", "zh", "sw", "xx"]
    section = build_reliability_section(langs, ["bleu", "comet"])
    assert set(section["languages"]) == set(langs)
    assert section["languages"]["xx"]["evidence"] == "none"
    assert set(section["languages"]["de"]["metrics"]) == {"bleu", "comet"}
    text = recommendation(langs, ["bleu", "comet"])
    assert "sw" in text and "zh" in text


def test_spbleu_for_unsegmented_scripts():
    assert bleu_tokenizer("zh") == "flores200"
    assert bleu_tokenizer("km") == "flores200"
    assert bleu_tokenizer("de") == "13a"
