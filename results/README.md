# results/

Every answer and score behind the report, the jury verdicts, the training loss curves and the command logs. Nothing here
is a model weight, an activation or a copy of a dataset.

## Start here

| Kind | Files | What they are |
|---|---|---|
| Derived numbers | `clsprobe_h38.json`, `clsprobe_h63.json`, `clsprobe_h64.json`, `clsprobe_h65.json`, `clsprobe_h68.json`, `clsprobe_h68_paired.json`, `selrule_h35.json`, `jury_h67.json`, `overlap_audit.json`, `dup_sens.txt`, `*_short_n300_scores.json` | Written by [`code/clsprobe.py`](../code/clsprobe.py), [`code/selrule.py`](../code/selrule.py) and [`code/jury_d.py`](../code/jury_d.py) from the stored answers; [`analysis/reproduce.sh`](../analysis/reproduce.sh) rewrites them. `overlap_audit.json` and `dup_sens.txt` come from the two overlap scripts, which need the training prompts (see the README at the root). The `hNN` in a name is the study's run number |
| Table fragments | `tables/` | Written by [`analysis/tables_gen.py`](../analysis/tables_gen.py): the shaded tables of the report as LaTeX fragments |
| Stored answers | `{tag}_short_n300_k{K'}_s{S}.jsonl`, `clsprobe_*.jsonl` | One record per item, with the model's output and scores. A `.meta.json` beside each run file holds the model snapshot, library versions, command line parameters and start time |
| Jury verdicts | `jury_d/` | The 1,200 summaries judged (answer, source index, copied share) and one verdict file per juror |
| Training | `training/` | The loss of every step of each trained delta (`step`, `loss`, `s` = seconds since the start) |
| Logs | `logs/` | The command logs of the training and evaluation chains of this report. The progress bars of the Hugging Face downloads and the retry lines were removed; every other line is as written. `first_recipe.log` holds the lines of the first recipe's training and evaluation, cut from a log that also held other models |

## Record fields

Short suite (`{tag}_short_n300_k{K'}_s{S}.jsonl`): `task` (summarize, qa, classify, rewrite), `i` (item number in the
split), `out` (the model's answer), `ref_scores` (the item's scores against the dataset reference, computed when the run
was made: `sum_rougeL_ref` for summarize, `qa_em` and `qa_f1` for question answering, `cls_acc` for classify; empty for
rewrite), `n_prompt`, `prompt_ids_len`, `n_gen`, `gen_ids`, and under the mask `loads_per_layer`, `touched_per_layer`,
`reselects`. The reference text is not stored.

Label probes (`clsprobe_*.jsonl`): `i`, `lp` (log-probability of each label's first token), `top`, `hit` (whether the best
label is the dataset label). `_rot` is the rotated label order; `clsprobe_emotion_*` is the emotion task.

Jury items (`jury_d/items.jsonl`): `cell` (unmasked, untrained, dbpedia, mixed), `i` (index in the test split), `id`,
`answer`, `source` (dataset, config, split, index of the article) and `copied` (share of the answer copied from the
article). Verdicts: `id` and `verdict` (three booleans); the hosted jurors' files also keep `raw` and `usage`.

Datasets, versions and licenses: [`LICENSE-DATA.md`](../LICENSE-DATA.md). How the records were made from the raw files:
[`analysis/strip_dataset_text.py`](../analysis/strip_dataset_text.py).

## Tags

`qwen36` and `gemma` are the models. `_d300` and `_d600` put the first item at position 300 and 600; `ends64` is the
both-ends rule for the short list, and no suffix the unmasked model (`k0`). `c3b` marks a run with a trained delta;
`stop12`, `stop16`, `stop24` are the second recipe at that budget (no `stop`: the first recipe). The arm follows: no
suffix is the DBpedia-only arm with fixed label order, `known` the mixed arm, `dbrand` the control, `zero` the
three-dataset recipe; `seedN` is the training seed (`knownseed3` is the mixed arm, seed 3). The paths in the `delta`
field of a meta file point to checkpoints that are not published.
