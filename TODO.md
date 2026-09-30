# AnomalyCodebook — remaining work for the ICLR 2027 submission

Last updated: 2026-09-09. Paper lives in the nested repo `latex/` (Overleaf-synced:
fetch + rebase before pushing, never force).

**Deadline check:** the ICLR 2027 dates have not been verified. If the usual pattern
holds, abstracts are due mid-September and full papers about a week later. Confirm
this first, since it decides whether items in "Nice to have" get done at all.

---

## 0. The open decision: which configuration the headline table reports

This is the one thing that needs a call before anything else can be finalized.

The adopted configuration (K=150, no image BCE, vocabulary-free, k-means init +
dead-code revival, sigma=4) is more trustworthy than what the paper currently
reports, and on several columns it is *worse*. The previously reported numbers came
from one seed under the random init, and that seed sat at the top of a 4-point spread.

| Dataset | Metric | Old (seed 42, random init) | New (stab42, adopted) | Best baseline |
|---|---|---|---|---|
| VisA | img AUROC | 85.3 | 83.2 | 83.6 MultiADS |
| VisA | img AP | 88.6 | 86.9 | 86.9 MultiADS |
| VisA | AUPRO | 90.0 | 89.6 | 89.7 MultiADS |
| MAD-Sim | px AUROC | 88.1 | 85.5 | 88.0 MultiADS |
| MAD-Sim | AUPRO | 77.2 | 72.1 | 74.2 MultiADS |
| Real-IAD | img AUROC | 81.0 | 78.4 | 78.7 MultiADS |
| Real-IAD | img AP | 81.1 | 79.0 | 79.1 MultiADS |
| MPDD | px AUROC | 96.5 | **96.9** | 96.5 AnomalyCLIP |
| MAD-Real | img AUROC | 67.9 | **71.8** | 78.5 MultiADS-F |

Under the adopted configuration we lose the Real-IAD image-level win, which the
introduction currently leads with, and MAD-Sim drops below MultiADS on pixel AUROC
and AUPRO.

The defensible resolution is to **report mean over seeds rather than one seed**, which
is what `msmain_fast` / `msmain_riad` are computing. Then the claim becomes "matches or
exceeds prior work on the pixel metrics, with image-level parity", which the stability
appendix already supports. Alternatives are to keep the single-seed random-init table
and treat stability purely as a limitation (weaker, and the appendix contradicts it),
or to adopt smoothing without the k-means init (recovers MAD-Sim but keeps the spread).

- [ ] **Decide this**, then rewrite `tab:main` and the §`sec:exp-main` prose, the
      introduction's results sentence, and the abstract in one pass.

## 1. Blocked on the LSF queue

All jobs use conda env `anomalyCB_new`. Check with `bjobs -w`.

| Job | ID | Fills | Output |
|---|---|---|---|
| `ksw_b` | 12800680 | `tab:k` at K = 70, 120, 200 | `results/ksweep_final/fk*/epoch_1/` |
| `ksw_a2` | 12807934 | `tab:k` at K = 50, 100, 150 (resubmit, see §5) | same |
| `finvocab_main` | 12807923 | `tab:composition`, with-vocabulary rows | `results/nosem_final_stab42/` |
| `finvocab_riad` | 12807927 | same, Real-IAD | same |
| `mpddrho_s4` | 12807930 | `tab:topk`, rho sweep at sigma=4 | `results/mpdd_rho_sig4/r*/` |
| `stab_s3` | 12808571 | fourth seed for `tab:app-remedies` | `results/stabtrain/stab3/` |
| `msmain_fast` | 12809445 | seed means for `tab:main` | `results/multiseed/s*/` |
| `msmain_riad` | 12809448 | same, Real-IAD | same |
| `usage_fast` | 12814347 | `tab:app-usage` on the adopted ckpt | `results/usage_final_stab42/` |
| `usage_loco` | 12814348 | same, MVTec-LOCO | same |
| `usage_riad` | 12814349 | same, Real-IAD | same |

**`tab:app-usage` is stale and load-bearing.** It was measured on `exps/retrain_nl150`,
which predates the k-means init, so the prototypes have moved. The two numbers it
produces, "under $0.06\%$ of assignments" and the "$0.59$ cosine margin", are quoted in
the abstract, the introduction twice, the conclusion, §`sec:exp-composition` and
`fig/gap.tex`. The nearest-prototype range "$0.81$ to $0.92$" at
[`3_method.tex:75`](latex/sec/3_method.tex#L75) has the same provenance. Nothing in the
argument should change, but every one of those six call sites needs the re-measured
value before submission.

Already collected and written into the paper:

- [x] §`sec:exp-cali` and `tab:cali`, from `results/nocali_final_stab42/`. Calibration
      gains +0.6 to +1.9 pixel AUROC on four datasets and +21.1 on MAD-Sim.
- [x] `tab:bmad` and `tab:loco`, from `results/stage2_final_stab42/`.
- [x] §`sec:app-stability` "Two remedies" and `tab:app-remedies`.

Still to write once the jobs land:

- [ ] **`tab:main`** from `results/multiseed/`, as mean over seeds (see §0).
- [ ] **`tab:k`** from `results/ksweep_final/fk*/epoch_1/`. Collected so far under the
      adopted config: K=120 gives VisA 95.0 px / 83.6 img, MPDD 96.0 / 69.2,
      BMAD brain 64.3 img, liver 64.7 img.
- [ ] **`tab:topk`** from `results/mpdd_rho_sig4/`. The unsmoothed sweep
      (`results/mpdd_rho_final/`) peaks at rho=0.1 (74.6) with rho=0.05 close behind
      (74.0); the headline main run at rho=0.05 with sigma=4 gives 74.5, so the
      smoothed sweep decides which rho the tuned MPDD row should quote.
- [ ] **`tab:composition`** and the with-vocabulary rows from `results/nosem_final_stab42/`.
- [ ] **`tab:objective`** still reflects the *old* initialization. Needs two retrainings
      (image BCE on and off) under k-means init to be strictly comparable. Not yet queued,
      because §0 may change what the comparison is for.
- [ ] **§`sec:exp-epoch`** with epoch-2 numbers under the adopted config, from
      `results/ksweep_final/fk150/epoch_2/` once `ksw_a2` lands.
- [ ] **Regenerate the per-class appendix tables**:
      `python3 tools/make_perclass_tex.py`, after pointing `RUNS` at
      `results/stage2_final_stab42/`.

## 2. Must do before submission

- [ ] **Compile on Overleaf.** Nothing in this environment can build LaTeX (tectonic has no
      bundle, the node has no network), so none of the three figures has ever been rendered.
      All are hand-checked for syntax; layout is unverified. First compile is the priority.
- [ ] **Check the figures visually.** [`latex/fig/architecture.tex`](latex/fig/architecture.tex),
      [`latex/fig/motivation.tex`](latex/fig/motivation.tex), [`latex/fig/gap.tex`](latex/fig/gap.tex).
      Most likely problems: the diagonal dashed arrow from the global-token box passing close
      to the adapter node; the three-panel motivation row overflowing `\textwidth`; whether
      `\resizebox` shrinks the architecture font too far.
- [ ] **Full coherence read-through.** The story changed twice (prompt-extendable →
      vocabulary-free), the objective changed once (image BCE removed) and the training
      recipe changed once (k-means init). Read start to finish for leftovers of any.
- [ ] **Verify every number against its log** one final time. The main table was silently
      wrong for a day because the full-suite runs predated the default flip, and again
      because the rho sweep was run unsmoothed while the headline was smoothed.

## 3. Known weaknesses a reviewer will find

- [ ] **MPDD and MAD-Real image AUROC are weak** (74.5 tuned, 71.8). `tab:objective`
      documents part of this honestly. Decide whether to pre-empt the objection.
- [ ] **Objective selection was not held out.** `nobce` was chosen on VisA, MPDD, MAD-Sim
      and MAD-Real, then evaluated on Real-IAD, BMAD and LOCO. Say so explicitly if
      challenged; do not present the latter three as independent confirmation.
- [ ] **MAD-Real image AUROC is still unstable** (5.5-point spread under the adopted
      config, worse than the 4.6 of the random init on matched seeds). Reported as an
      open limitation in §`sec:app-stability`; keep it reported.
- [ ] **rho is tuned per dataset for MPDD only.** Already flagged in the text and in
      Limitations; keep it flagged.

## 4. Nice to have, cut first if time runs out

- [ ] Qualitative figure: anomaly maps for a few products across datasets, ideally with the
      assignment-entropy map beside them, since that visualizes the calibration signal.
- [ ] Appendix: prompt vocabulary listing per dataset.
- [ ] A label-free criterion for rho and for the stopping epoch. This is the paper's own
      stated next step in the Conclusion; a working version would remove the last
      dependence on target supervision, but it is a new contribution, not a fix.

## 5. Reference: what changed in the code (already done)

- `--with_semantic_codebook` is now the opt-in ablation; vocabulary-free is the default
  (`test.py`). The old flag was `--no_semantic_codebook`.
- `--semantic_bias` adds an offset to the semantic logits before the argmax (`model.py`),
  used for the forcing sweep. Calibration is deliberately computed from the *raw* cosine
  logits so that this flag cannot silently move the temperature.
- `--image_loss_weight` defaults to `0.0` (`train.py`); `retrain_codebook.bsub` passes
  `0.03` explicitly so the older sweeps still reproduce.
- `--codebook_init warmup_kmeans`, `--codebook_warmup_steps`, `--revive_every` (`train.py`).
- `--smooth_sigma` applies a separable Gaussian to the anomaly map before scoring (`test.py`).
- `test.py` logs codebook usage and the per-half mean best cosine on every run.
- **Fixed 2026-09-09:** `HybridCodebook.init_from_features` crashed with a device-side
  assert (`invalid multinomial distribution`) when the k-means++ distance vector summed
  to zero, which killed the K=100 training in `ksw_a`. Now falls back to a uniform pick.
- **Fixed 2026-09-09:** `withsem_ablation.bsub` had `set -eo pipefail` *before*
  `module load conda`, which writes to stderr and exited the job in two seconds. Every
  other bsub script already ordered these correctly.
