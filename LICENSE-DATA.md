# License for the report, the data and the figures

© 2026 Julien Simon · CC BY 4.0 · [creativecommons.org/licenses/by/4.0](https://creativecommons.org/licenses/by/4.0/)

The report (`moe-router-delta.pdf`), the text (`README.md`, `results/README.md`), the data (`results/`) and the figures
(`figures/`) are licensed under the Creative Commons Attribution 4.0 International license. You may share and adapt
them for any purpose, commercial included, if you credit the author and link to this license.

Suggested credit: "Julien Simon, Solfird AI Technical Report 2026-03, CC BY 4.0."

What this license does not cover:

- The code (`code/`, `analysis/`) is under the Apache License 2.0 in `LICENSE`.
- The Solfird AI name and logo in `assets/`.
- The benchmark datasets. This repository holds no article, question, passage or reference text. The per-item records
  hold item numbers, the model's own output and scores; a command that needs the source text downloads it from the
  Hugging Face Hub by item index. Credit and license of each dataset used for the scored items, as the Hub lists them:

  | Dataset | Version used | License on the Hub |
  |---|---|---|
  | `abisee/cnn_dailymail` (CNN/DailyMail) | 3.0.0, test split | Apache-2.0 |
  | `rajpurkar/squad` (SQuAD) | validation split | CC BY-SA 4.0 |
  | `fancyzhx/ag_news` (AG News) | test split | license unknown on the Hub |
  | `SetFit/emotion` (emotion) | test split | no license field on the Hub card |

  The training prompts were built from other splits and datasets: `EdinburghNLP/xsum`, `mrqa-workshop/mrqa` (with its
  SQuAD part removed), `fancyzhx/dbpedia_14`, `fancyzhx/ag_news` (train split), and for one arm `SetFit/20_newsgroups` and
  `fancyzhx/amazon_polarity`. None of their text is in this repository. Each keeps the license the Hub lists for it.
- The outputs of the models under test (Qwen3.6-35B-A3B and Gemma 4, in 4-bit conversions from `mlx-community`) and of the
  jurors (GPT-5.6 Luna, Gemini 3.5 Flash-Lite, Claude Haiku 5.5). They come from models under their own licenses and
  terms (Qwen3.6 and Gemma 4: Apache-2.0 on the Hub; the three hosted jurors: their providers' terms). This license does
  not decide the rights in those outputs. This repository holds no model file and no trained delta.
