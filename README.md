# VedOCR — Vedic Sanskrit Manuscript Line Recognition

A handwritten/printed text recognition (HTR/OCR) pipeline for line-level images of a Vedic
Sanskrit manuscript (Devanagari script, Pada-pāṭha), built as CNN encoder → BiLSTM → CTC
(CRNN+CTC), end to end from raw dataset to a saveable, reusable inference pipeline.

## Project structure

```
vedocr-project/
│
├── data/                     # the VedOCR dataset (images + transcriptions, page-level splits)
├── notebooks/                 # exploratory notebook: EDA -> training -> evaluation -> pipeline demo
├── src/                       # reusable, importable pipeline code (the "real" implementation)
├── models/                    # trained checkpoint + saved end-to-end OCRPipeline artifact
├── app/                       # minimal CLI to run inference with the saved pipeline
├── reports/                   # planning guide + current results/limitations write-up
│
├── README.md
├── requirements.txt
└── config.yaml                # single source of truth for paths & hyperparameters
```

## Dataset

681 line-level manuscript image crops (57 pages), RGB, paired with expert transcriptions.
87 grapheme classes (base letters, conjuncts, Vedic accent/tone marks), 9 of them singletons
(≤1 training example — a known, documented limitation, not a bug). Splits are fixed at the
**page level**: train 539 / val 59 / test 83 lines, so no page (and no handwriting style) appears
in more than one split.

Full data understanding, preprocessing, validation-strategy, and modeling rationale live in
[`reports/VedOCR_planning_guide.md`](reports/VedOCR_planning_guide.md).


## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# continue training from the shipped checkpoint (models/vedocr_crnn_ckpt.pt)
python -m src.train --epochs 50

# evaluate on the held-out test set (CER/WER + singleton-grapheme error breakdown)
python -m src.evaluate

# run inference on a single line image via the saved end-to-end pipeline
python app/infer.py --image data/VedOCR_kaggle_release/hf_format/test/page002_line001.png
```

Or open `notebooks/VedOCR_pipeline.ipynb` for the full walkthrough (EDA, validation checks,
preprocessing/augmentation visualization, training, evaluation, and the pipeline demo) —
already executed once, with real output, against this exact dataset.

## `src/` module map

| file | purpose |
|---|---|
| `config.py` | loads `config.yaml`, resolves paths relative to the project root |
| `tokenizer.py` | character-level vocab, encode/decode, CTC blank handling |
| `preprocessing.py` | image resize/pad/normalize + train-only augmentation |
| `data.py` | split loading + integrity checks (unreadable images, empty labels, vocab coverage, singleton graphemes) |
| `dataset.py` | `LineDataset` + CTC-appropriate `collate_fn` |
| `model.py` | the `CRNN` architecture |
| `metrics.py` | CER / WER via edit distance |
| `train.py` | resumable training loop + `python -m src.train` CLI |
| `evaluate.py` | test-set evaluation + error analysis CLI |
| `pipeline.py` | `OCRPipeline` — the single object that bundles preprocessing + model + decoding, with `.save()` / `.load()` |

## Current status

Test CER ≈ 0.96 with the shipped 7-epoch checkpoint — **this reflects a CPU-only, single-core
training environment, not a ceiling on the approach.** See
[`reports/RESULTS.md`](reports/RESULTS.md) for the honest breakdown and exactly what to change
(more epochs on a GPU; start from pretrained Devanagari/Indic OCR weights; synthetic data) to get
this to a genuinely usable model.

## License note

The underlying dataset is CC BY-NC-SA 4.0 (non-commercial) — keep that in mind before using this
for anything beyond research/personal projects.
