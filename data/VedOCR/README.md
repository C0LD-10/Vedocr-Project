# VedOCR Dataset

**An Expert-Annotated Dataset for Vedic Sanskrit Manuscript OCR**

Accompanies the paper *"VedOCR: An Expert-Annotated Dataset and Transformer OCR Baseline
for Vedic Sanskrit Manuscripts"*, accepted at DALL 2026 @ ICDAR (Workshop on Documents
Analysis of Low-Resource Languages), Vienna, September 2026.

---

## Overview

This dataset covers a *Pada-pāṭha* Vedic Sanskrit manuscript — an analytical form that
decomposes compound words into morphemes separated by spaces. This tradition creates
segmentation challenges absent from modern Devanagari OCR benchmarks.

| Property | Value |
|---|---|
| Annotated pages | 57 |
| Line-level segments | 681 |
| Character instances | 41,286 |
| Unique grapheme classes | 87 (incl. archaic conjunct consonants and Vedic tone marks) |
| Singleton grapheme classes (≤1 example) | 9 |
| Train / Val / Test split (pages) | 45 / 5 / 7 |

Splits are enforced at the **page level** — all lines from a given page appear in exactly
one split, so no page leaks across train/val/test.

---

## Contents

```
line_crops/            681 PNG line-level image crops — the core dataset
sample_pages/          3 example full-page manuscript scans (illustrative only;
                        not all 57 source pages are included)
master.tsv             One row per line crop: path, page/line ids, transcription,
                        bounding box, and the filename of the source page scan
splits/
  train.tsv            540 lines (incl. header) from 45 pages
  val.tsv              60 lines (incl. header) from 5 pages
  test.tsv             84 lines (incl. header) from 7 pages
jsonl/
  train.jsonl / val.jsonl / test.jsonl   same splits, JSON-Lines format
hf_format/
  train/ val/ test/    HuggingFace `datasets`-compatible image folders,
                        each with its own metadata.csv (file_name, text)
vocab/
  char_vocab.txt        87 unique grapheme classes
  char_freq.tsv         per-character frequency, Unicode codepoint, and block
```

### `master.tsv` / `splits/*.tsv` columns

| Column | Description |
|---|---|
| `crop_path` | Relative path to the line-crop image, e.g. `line_crops/page001_line001.png` |
| `page_id` | Internal annotation-tool page identifier |
| `page_num` | Sequential page number (1–57) |
| `line_num` | Line number within the page |
| `label` | Region label (`Sanskrit Text`) |
| `transcription` | Expert ground-truth transcription |
| `source_page_image` | Filename of the original full-page scan this line was cropped from |
| `x0, y0, x1, y1` | Bounding box of the line on the source page (pixels) |

`jsonl` and `hf_format` mirror the same data in formats more convenient for HuggingFace
`datasets` / PyTorch pipelines.

---

## Loading the dataset

**Plain Python / pandas:**
```python
import pandas as pd
df = pd.read_csv("master.tsv", sep="\t")
```

**HuggingFace `datasets`:**
```python
from datasets import load_dataset
ds = load_dataset("imagefolder", data_dir="hf_format")
```

---

## License

Released under **CC BY-NC-SA 4.0** (Attribution–NonCommercial–ShareAlike 4.0
International). See `LICENSE` for full terms. Released with the permission of the
manuscript's custodian for public research use.

---

## Citation

```bibtex
@inproceedings{vedocr2026,
  title     = {VedOCR: An Expert-Annotated Dataset and Transformer OCR Baseline
               for Vedic Sanskrit Manuscripts},
  booktitle = {Proceedings of DALL 2026 @ ICDAR},
  year      = {2026}
}
```

*(Full citation with authors and DOI to be added after camera-ready.)*

---

## Contact

Suyash Kumar Bhagat — Delhi Technological University
