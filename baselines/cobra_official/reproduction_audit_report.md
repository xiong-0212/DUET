# CoBRA reproduction audit (2026-10-11)

> Trigger: the released CoBRA checkpoint was suspected of having been reproduced incorrectly, so the
> official repository was re-downloaded from scratch (`CoBRA/CoBRA-main`).
> Artefacts: `official_pipeline_results.json` (official pipeline, end to end) · `results.json`
> (our earlier reproduction) · `threshold_sweep.json` · `subset_analysis.json`

---

## 0. Conclusion

**Our earlier reproduction was correct.** Running the freshly downloaded official code verbatim,
end to end (the authors' `embed_fasta` + `run_folder`, including their AMP half-precision
embedding path), gives numbers that agree **digit for digit** with our hand-written reproduction.

We also confirm that **our reproduced values are systematically higher than the values reported in
Table 3 of the CoBRA paper** (+0.015 to +0.036 per benchmark, +0.049 on average). We tested every
protocol we could think of (threshold sweep, overlap-free subsets, per-chain averaging versus
residue pooling, the official code) and **none of them reproduces the published numbers**. The most
likely explanation is that the released weights are not the same training run that produced Table 3
— and that table's M1–M5 were themselves **selected from 60 models by test-set MCC**.

---

## 1. Agreement of three independent paths (test sets de-duplicated, RB9 = 6 chains)

| Path | TE18 | TL12 | JL10 | RB9 | Mean MCC |
|:---|---:|---:|---:|---:|---:|
| 1. Earlier reproduction (hand-written inference, fp32) | 0.212 | 0.568 | 0.329 | 0.671 | **0.445** |
| 2. **Official code end to end** (fresh download, AMP fp16) | 0.212 | 0.568 | **0.329** | **0.671** | **0.445** |
| 3. Paper Table 3 (M3 = RiNALMo + BCE) | 0.190 | 0.546 | 0.293 | 0.557 | 0.396 |

Per-benchmark Precision/Recall from the official pipeline: TE18 0.510/0.389 · TL12 0.804/0.607 ·
JL10 0.543/0.549 · RB9 0.679/0.841 — identical to 1.

> TE18/TL12 for 2 and 3 are recorded in `official_pipeline_results.json`.

## 2. Corroborating checks that the reproduction chain is sound

| Check | Result |
|:---|:---|
| Official architecture versus ours | `predictor.py` MLP head matches layer for layer (1280→1024→256→128→64 + LN/ReLU/Dropout 0.1) |
| Official data files versus our copies | all 8 files `cmp`-identical (the paper's own count, 432 chains = 345+43+44 ✓) |
| Official decision rule | `argmax` is equivalent to `p ≥ 0.5` for two classes, matching ours |
| Trustworthiness of the RiNALMo weights | `_dl/verify_report.json`: masked-position predictions agree with the official model card to ~1e-7 |
| Effect of half precision | the official AMP path and our fp32 path give **identical** metrics |

## 3. Why the reproduced values exceed the published ones: what was ruled out

| Candidate explanation | Test | Result |
|:---|:---|:---|
| Different threshold | swept 0.05–0.95 for the threshold closest to the published MCC | No — the four benchmarks need contradictory thresholds (TE18≈0.62, TL12≈0.87, JL10≈0.20, ...) |
| Overlapping sequences removed at evaluation | evaluated on the subset of 16 chains identical to the training set | No — TE18/JL10 match (0.185/0.269 vs 0.190/0.293), TL12/RB9 diverge badly (0.297/0.427 vs 0.546/0.557) |
| Per-chain averaging instead of residue pooling | computed both ways | No — per-chain averaging is worse still (JL10 0.371 vs the published 0.293) |
| A bug in the official code | ran the **freshly downloaded official code** verbatim | No — identical to the hand-written reproduction |
| A different version of the data files | compared one by one against the fresh download | No — identical |

**Conclusion: the values in Table 3 of the CoBRA paper cannot be reproduced from the released
weights, data and code** (the gaps range over ±0.02–0.11 and are not even consistent in sign).
Given that Table 3 selects the top 5 of 60 models by test-set MCC, its reported M3 is plausibly a
different training run.

## 4. Two environment details encountered (not bugs; documented here)

1. **flash-attn naming mismatch.** This version of the official RiNALMo defaults to
   `use_flash_attn=True`, whose weight keys are the fused `mh_attn.Wqkv`; the local weights were
   converted from the Hugging Face release (split `to_q`/`to_k`/`to_v`, mathematically equivalent).
   The authors' `embedder.load_model()` performs no fallback, so `use_flash_attn=False` must be set
   explicitly (`run_official_pipeline.py` does this and records the reason).
2. **The `gdown` dependency.** `rinalmo.pretrained` imports gdown at module level to download
   weights; an offline environment needs a local weight cache
   (`~/.cache/rinalmo_pretrained/giga-v1.pt`) plus an import stub.

## 5. Consequences for our paper

- **A more conservative comparison.** Our baseline is the **reproduced** value (0.445; 0.420 before
  de-duplication), which is *higher* than the published 0.396 — i.e. we compare against a stronger
  CoBRA.
- **Suggested wording for the Supplementary Information.** Give the reproduction table (Section 1
  of this file), state that the reproduced values are used as the reference with the published
  values listed alongside, and note that the published values could not be reproduced. Do **not**
  claim in the main text that we outperform CoBRA: MCC is a tie under both protocols (0.361 vs
  0.361 after de-leaking) and AUC is uniformly lower (0.735 vs 0.763).

## 6. Reproduction commands

```bash
cd cobra_repro
python run_official_pipeline.py     # official code end to end (needs ~/.cache/rinalmo_pretrained/giga-v1.pt)
python run_cobra_eval.py            # our earlier reproduction, per benchmark against the paper
python threshold_analysis.py        # threshold-protocol sweep
python subset_analysis.py           # leaked / non-leaked subset evaluation
```