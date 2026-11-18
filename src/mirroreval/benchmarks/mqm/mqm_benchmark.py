"""
MQM Benchmark — MT Metric Reliability
======================================
Evaluates a translation model by:

  1. Translating FLORES+ English sentences into configured target languages
     using the model under test (settings.model.model_checkpoint_path).
  2. Computing automated MT metrics (BLEU, ChrF, COMET) on those translations
     against FLORES+ reference translations.
  3. Reporting per-language metric scores alongside per-language reliability
     evidence from AmanaMT v2 (see mqm_reliability.py): how well each metric
     tracked human labels for that language, and whether that evidence is direct
     (en->X human labels), indirect (X->en only), or absent (synthetic or
     uninformative labels).

Output JSON (saved to settings.mqm.output_dir/mqm_results.json) has keys
"model", "languages_evaluated", "flores_split", "metrics_run", "metric_scores"
(per-language corpus averages), "metric_reliability" (per-language evidence from
the packaged AmanaMT v2 table) and "recommendation" (plain-English summary).

The model_checkpoint_path is expected to be an NLLB-200-style seq2seq translation
model (e.g., "facebook/nllb-200-distilled-600M"). The model must support the
`translation` task via a HuggingFace transformers pipeline with src_lang /
tgt_lang arguments.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import pipeline as hf_pipeline

from mirroreval.config import init_settings, settings
from mirroreval.benchmarks.interfaces import METRICS, DATASETS
from mirroreval.logger import logger

# Side-effect imports — trigger @register_dataset / @register_metric decorators
import mirroreval.benchmarks.mqm.mqm_datasets  # noqa: F401
import mirroreval.benchmarks.mqm.mqm_metrics   # noqa: F401

from mirroreval.benchmarks.mqm.mqm_datasets import NLLB_LANG_CODES
from mirroreval.benchmarks.mqm.mqm_reliability import (
    build_reliability_section,
    recommendation,
)


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------
def _translate_with_model(sources: list[str], nllb_tgt_lang: str,
                           model_name: str, batch_size: int = 32) -> list[str]:
    """
    Translate a list of English source sentences to tgt_lang using model_name.

    The model is expected to be an NLLB-200-style seq2seq model that accepts
    src_lang / tgt_lang tokens (e.g., "facebook/nllb-200-distilled-600M").

    Returns a list of translated strings in the same order as sources.
    """
    logger.info(
        f"Translating {len(sources)} sentences to {nllb_tgt_lang} "
        f"with {model_name}..."
    )
    device = 0 if torch.cuda.is_available() else -1
    pipe = hf_pipeline(
        "translation",
        model=model_name,
        src_lang="eng_Latn",
        tgt_lang=nllb_tgt_lang,
        device=device,
        max_length=256,
    )
    results = []
    for i in range(0, len(sources), batch_size):
        batch = sources[i: i + batch_size]
        outputs = pipe(batch)
        results.extend(o["translation_text"] for o in outputs)
    return results


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------
def _corpus_metric_scores(segments: list[dict]) -> dict:
    """
    Average per-segment metric scores across all segments for a single language.
    Returns {"bleu": float, "chrf": float, "comet": float | None}.
    """
    scores = defaultdict(list)
    for seg in segments:
        for key in ("bleu_score", "chrf_score", "comet_score"):
            if key in seg and seg[key] is not None:
                scores[key].append(seg[key])
    return {
        "bleu":  sum(scores["bleu_score"])  / len(scores["bleu_score"])  if scores["bleu_score"]  else None,
        "chrf":  sum(scores["chrf_score"])  / len(scores["chrf_score"])  if scores["chrf_score"]  else None,
        "comet": sum(scores["comet_score"]) / len(scores["comet_score"]) if scores["comet_score"] else None,
    }


# ---------------------------------------------------------------------------
# Main benchmark entry point
# ---------------------------------------------------------------------------
def run_benchmark():
    model_name = settings.model.model_checkpoint_path
    target_langs = list(settings.mqm.languages)
    run_metrics = list(settings.mqm.metrics)
    batch_size = int(settings.mqm.get("translation_batch_size", 32))
    flores_split = str(settings.mqm.get("flores_split", "devtest"))

    output_dir = Path(settings.mqm.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "mqm_results.json"

    if output_file.exists():
        logger.info(f"Results already exist at {output_file}. Skipping.")
        with open(output_file) as f:
            return json.load(f)

    # --- Load FLORES+ dataset ---
    logger.info("Loading FLORES+ dataset...")
    dataset = DATASETS["openlanguagedata/flores_plus"](
        target_langs=target_langs, split=flores_split
    )

    # Group segments by language
    segments_by_lang: dict[str, list[dict]] = defaultdict(list)
    for seg in dataset:
        segments_by_lang[seg["lang"]].append(seg)

    # --- Translate with model under test ---
    for lang, segs in segments_by_lang.items():
        nllb_lang = NLLB_LANG_CODES.get(lang)
        if nllb_lang is None:
            logger.warning(f"No NLLB code for '{lang}', skipping translation.")
            for seg in segs:
                seg["hypothesis"] = ""
            continue
        sources = [s["source"] for s in segs]
        hypotheses = _translate_with_model(
            sources, nllb_lang, model_name, batch_size=batch_size
        )
        for seg, hyp in zip(segs, hypotheses):
            seg["hypothesis"] = hyp

    # Flatten segments back to a list
    all_segments = [seg for segs in segments_by_lang.values() for seg in segs]

    # --- Run metrics ---
    for metric_name in run_metrics:
        if metric_name not in METRICS:
            logger.warning(f"Unknown metric '{metric_name}', skipping.")
            continue
        logger.info(f"Running metric: {metric_name}")
        all_segments = METRICS[metric_name]()(all_segments)

    # Re-group by language for per-language scoring
    segments_by_lang_scored: dict[str, list[dict]] = defaultdict(list)
    for seg in all_segments:
        segments_by_lang_scored[seg["lang"]].append(seg)

    # --- Build results ---
    metric_scores = {
        lang: _corpus_metric_scores(segs)
        for lang, segs in segments_by_lang_scored.items()
    }

    results = {
        "model": model_name,
        "languages_evaluated": target_langs,
        "flores_split": flores_split,
        "metrics_run": run_metrics,
        "metric_scores": metric_scores,
        "metric_reliability": build_reliability_section(target_langs, run_metrics),
        "recommendation": recommendation(target_langs, run_metrics),
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_file}")
    logger.info(f"Metric scores: {metric_scores}")
    return results


# ---------------------------------------------------------------------------
# SLURM entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        init_settings(sys.argv[1])
    else:
        logger.error("No settings file path provided.")
        sys.exit(1)
    run_benchmark()
