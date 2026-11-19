<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/solfird-ai-on-dark.svg">
  <img src="assets/solfird-ai-on-light.svg" alt="Solfird AI" width="240">
</picture>

# Training Only the MoE Router for a Short List of Experts: Short Answers Recover, Summaries Do Not

Solfird AI Technical Report 2026-03 · November 2026 · Julien Simon

[Report (PDF, 14 pages)](moe-router-delta.pdf) · [GitHub](https://github.com/solfird/moe-router-delta) · DOI [10.5281/zenodo.24362881](https://doi.org/10.5281/zenodo.24362881) (version 1.0) · all versions: [10.5281/zenodo.24362880](https://doi.org/10.5281/zenodo.24362880)

## The result

A small trained correction to the router lets Qwen3.6-35B-A3B run on 12 of its 256 experts per layer (4.7%) and keep question answering at about 88% of the full model's score, up from 72.6% with the untrained list. The jury that reads the summaries says the repair stops there: it accepts 17 to 24% as many trained summaries as full-model summaries, against 3.4% untrained. Classification is unstable: the same recipe scores from 66.7 to 95.0% across six training seeds, so a single classification run proves nothing. The one clear classification gain comes from training on examples of the test task itself (AG News), and it does not reach a second task that was not in training. With the list off, the model's answers are identical to the original on every item tested.

So train the router for short-answer tasks and confirm on your own task. Do not expect good summaries from this recipe, judge generated text with a jury or a person and not with ROUGE-L, and run at least six seeds before you claim a classification result.

The question: after the [first report](https://github.com/solfird/moe-short-lists) showed that a frozen MoE model can run on a short list of experts with no training and loses much at the smallest lists, how much of that damage can training only the router repair, and where does the repair stop? The delta is one matrix per layer (21 million numbers, 84 MB), active only under the short list. Three arms (two training sets, one control), six seeds each, on one model at one budget; section 2 of the report has the setup.

![Three panels, Qwen3.6 with 12 of 256 experts. Left: question answering, kept score of each of six seeds for three training arms, all near 88% against 72.6% untrained, below the 90% bar. Middle: summaries a jury accepts, as a share of the full model's rate, 3.4% untrained, 24.3% and 17.2% for the two deltas. Right: AG News accuracy of each seed, spread from 66.7 to 96.1% depending on the arm and the seed.](figures/fig_teaser.png)

*Left and right: kept score (% of the full model) of each of six seeds (dots) and their median (bar) per training arm, against the untrained list (grey line) and the 90% bar (dashed). Middle: summaries accepted by a jury of three models, as a share of the full model's, with 95% intervals.*

## Main results

| Result | Value | Basis |
|---|---|---|
| Question answering at 12 of 256 experts, items 300 to 599 | 72.6% untrained [67.6, 77.6]; median 87.7% (DBpedia-only arm) and 88.5% (mixed arm) over six seeds each; no seed below about 84% | Measured |
| Question answering on new items 600 to 899, one run per arm | 73.2% untrained; 86.6% and 88.2% | Measured |
| Looping summaries | 26.3% untrained; 3.3 to 8.3% across the trained seeds | Measured |
| Unmasked model | Identical answers on all 1,200 items of every run | Measured |
| Summaries accepted by the jury, as a share of the full model's, items 600 to 899 | 3.4% untrained; 24.3% DBpedia-only delta; 17.2% mixed delta | Measured, one seed per delta |
| Seed range of classification (AG News) | 66.7 to 95.0% for one recipe, a range of 28.3 points | Measured |
| Training on AG News examples | Median 92.2% (mixed arm) against 81.4% (DBpedia-only) and 81.0% (control with random label order); exact one-sided rank tests 0.013 listed order, 0.004 rotated order; 0.130 and 0.052 after the Holm correction | Measured, six seeds per arm |
| Gain on a task not in training (emotion) | No difference detected; seed spread of 9 to 14 points | Measured |
| Cost of the delta | 84 MB; 1.5 to 2.5 hours on one laptop | Measured |

Each table of the report marks a result above the 90% bar, under it, or undecided by its 95% interval. All six-seed results are for one model at one budget.

## Reproduce the numbers and the figures

No model, no GPU, no network, no paid API and no dataset download. You need [uv](https://docs.astral.sh/uv/), poppler (`pdftotext`) and a machine where the `mlx` wheel installs (Apple silicon), because the scripts import the code that wrote the answers.

```sh
sh analysis/reproduce.sh
git diff --stat results/
```

The script rewrites the six `clsprobe` result files, `selrule_h35.json` and `jury_h67.json` from the stored answers and verdicts, writes the four table fragments to `results/tables/`, rebuilds the four figures with matplotlib 3.9.2, and ends with `check_numbers.py`, which recomputes the report's numbers from the run files and looks for each one in the text of the PDF. It prints `all numbers found` and exits 0. The `git diff` command must print nothing for `results/`; the PDF figures in `figures/` carry a build time and show as changed. The measurements themselves (training, generation, the jury) are repeated with the commands under "What you cannot reproduce".

## What is in this repository

| Path | Content |
|---|---|
| [`moe-router-delta.pdf`](moe-router-delta.pdf) | The report |
| [`results/`](results) | Every answer and score behind the report, the jury verdicts, the training loss curves, the command logs; [`results/README.md`](results/README.md) says where to start |
| [`code/`](code) | The training and generation code (`c3b.py`), the short-task suite, the label probes (`clsprobe.py`), the jury (`jury.py`, `jury_d.py`), the overlap check (`overlap_audit.py`, `dup_sens.py`), the shared modules they import, and their tests in `code/tests/` |
| [`analysis/`](analysis) | The scripts that compute the report's numbers, tables and figures: `check_numbers.py`, `tables_gen.py`, `make_figures.py`, `reproduce.sh`, and `strip_dataset_text.py`, which documents how dataset text was removed from the records |
| [`figures/`](figures) | The data figures of the report, as PDF and PNG |
| [`LICENSE-DATA.md`](LICENSE-DATA.md) | Licenses of the data, the datasets and the model outputs |

## Where each result comes from

Section numbers are those of the report.

| Result | Section | Evidence |
|---|---|---|
| Setup, unmasked answers identical | 2 | [`clsprobe_h38.json`](results/clsprobe_h38.json) (`unmasked_identical`), the `_k0_` run files in [`results/`](results) |
| First recipe never learned to stop | 3 | [`selrule_h35.json`](results/selrule_h35.json), [`first_recipe.log`](results/logs/first_recipe.log), [`h35.log`](results/logs/h35.log) |
| Question answering lift, six seeds | 4 | [`clsprobe_h64.json`](results/clsprobe_h64.json), [`tab_qa_seeds.tex`](results/tables/tab_qa_seeds.tex), [`h63_seeds.log`](results/logs/h63_seeds.log), [`h64_seeds.log`](results/logs/h64_seeds.log) |
| Jury on the trained summaries | 5 | [`jury_h67.json`](results/jury_h67.json), [`jury_d/`](results/jury_d), [`jury_d.py`](code/jury_d.py), [`jury_d_luna.log`](results/logs/jury_d_luna.log) |
| Seed noise in classification | 6 | [`clsprobe_h68.json`](results/clsprobe_h68.json), [`clsprobe_h38.json`](results/clsprobe_h38.json), [`clsprobe_h63.json`](results/clsprobe_h63.json), [`tab_allseeds.tex`](results/tables/tab_allseeds.tex) |
| Seen and unseen classification, control | 7 | [`clsprobe_h68.json`](results/clsprobe_h68.json), [`clsprobe_h68_paired.json`](results/clsprobe_h68_paired.json), [`tab_medians.tex`](results/tables/tab_medians.tex), [`tab_gain.tex`](results/tables/tab_gain.tex), [`h68_dbrand.log`](results/logs/h68_dbrand.log) |
| One run on new items | 8 | [`clsprobe_h65.json`](results/clsprobe_h65.json), [`h65_final.log`](results/logs/h65_final.log), the `qwen36_d600_*` run files |
| Jury design and agreement | Appendix A | [`jury_h67.json`](results/jury_h67.json) (`agreement`, `jurors`) |
| Overlap between training and test text | Appendix B | [`overlap_audit.json`](results/overlap_audit.json), [`dup_sens.txt`](results/dup_sens.txt), [`overlap_audit.py`](code/overlap_audit.py), [`dup_sens.py`](code/dup_sens.py) |
| Every classification seed | Appendix C | [`clsprobe_h68.json`](results/clsprobe_h68.json), [`tab_allseeds.tex`](results/tables/tab_allseeds.tex) |
| Numbers stated in the text | Appendix D | [`check_numbers.py`](analysis/check_numbers.py) |

The `hNN` in a result file name is the study's run number. The report does not use it.

## What you cannot reproduce from this repository

- **Training and generation.** They need Qwen3.6-35B-A3B in the 4-bit `mlx-community` conversion and about 1.5 to 2.5 hours per delta on an M3 Max with 48 GB (MLX 0.32.4). The trained deltas (84 MB each) are not here. To repeat them: `python code/c3b.py teach --model-tag qwen36`, `python code/c3b.py teach-cls --model-tag qwen36 --recipe known`, then for example `python code/c3b.py train --model-tag qwen36 --k 12 --s 32 --ends 64 --stop --recipe known --seed 1 --chunked-la`, and `python code/selrule.py run --model-tag qwen36 --arm ends64 --frac 0.046875 --n 300 --start 300 --delta CKPT --label _stop12knownseed1`. The meta file beside each run holds its exact parameters.
- **The teacher data and the dataset text.** The training prompts and the model's own teacher answers hold dataset text, so they are not published; `c3b.py teach` rebuilds them from the Hugging Face Hub. The per-item records hold item numbers, the model's output and scores, and no article, question or reference text. [`LICENSE-DATA.md`](LICENSE-DATA.md) names each dataset.
- **The overlap check.** `overlap_audit.py` and `dup_sens.py` read the training prompts, so they need the teacher data above. Their outputs are in [`results/`](results), and the functions they use are tested in `code/tests/`.
- **The jury.** Re-running it calls hosted models: `export REQUESTY_API_KEY=...` and `python code/jury.py --round d run --juror luna`. It costs about $0.82. The verdicts of all three jurors are stored in [`results/jury_d/`](results/jury_d), so nobody has to pay to check a number. The Claude Haiku 5.5 verdicts were produced from blind batch files (`code/blindbatch.py`).
- **Some command logs.** [`results/logs/`](results/logs) holds the logs of the training and evaluation chains of this report. The evaluation log of the seed-0 DBpedia-only arm sits in a working log that also holds runs of other models, and is left out. Its answers are in [`results/`](results) and its loss curve in [`results/training/`](results/training).

## Limits

One model (Qwen3.6-35B-A3B, a 4-bit conversion), one budget (12 of 256 experts), one machine, 300 items per task and cell. The six-seed results use items 300 to 599, which also chose the recipe; items 600 to 899 were kept out of every choice and run once per arm, so their intervals cover item sampling and not seed variance. The jury is made of models and no person graded the summaries. Gemma 4 appears only in a two-seed comparison. Section 10 of the report gives the limits in full.

## Corrections

Found an error? [Open an issue](https://github.com/solfird/moe-router-delta/issues). A correction is a new release of this
repository, and its release notes say what changed.

## Cite

```bibtex
@techreport{simon2026routerdelta,
  author      = {Simon, Julien},
  title       = {{Training Only the MoE Router for a Short List of Experts: Short Answers Recover, Summaries Do Not}},
  institution = {Solfird AI},
  number      = {2026-03},
  year        = {2026},
  month       = nov,
  doi         = {10.5281/zenodo.24362881},
  url         = {https://github.com/solfird/moe-router-delta},
}
```

GitHub's "Cite this repository" button gives the same reference from [`CITATION.cff`](CITATION.cff).

## Licenses

- The report, the figures and the data: CC BY 4.0, with the exceptions listed in [`LICENSE-DATA.md`](LICENSE-DATA.md).
- The code: Apache 2.0, in [`LICENSE`](LICENSE).
- The Solfird AI name and logo in `assets/` are not under these licenses.

This work had no funding from any model vendor, and none reviewed it. Contact: julien@solfird.ai.
