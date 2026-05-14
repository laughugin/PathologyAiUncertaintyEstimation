# Uncertainty Estimation — Digital Pathology Thesis

## What this project is

Diploma thesis implementing and evaluating uncertainty estimation methods for a ViT-based binary tumor classifier on the PatchCamelyon (PCAM) histology dataset.

**Author**: working locally in `/home/ubuntu/diplom`  
**Model**: `google/vit-base-patch16-224` fine-tuned for 2-class PCAM  
**Python venv**: always activate first: `cd /home/ubuntu/diplom && source venv/bin/activate`

---

## Specialist agents — use these by name

| Agent | Invoke as | Does |
|---|---|---|
| **Analyst** | `/analyst` | Audits evaluation JSONs, finds inconsistencies, writes `evaluation/ANALYSIS_REPORT.md` |
| **Thesis Writer** | `/thesis-writer` | Writes/improves LaTeX chapters from results, applies natural writing style |
| **UI/UX** | `/ui-ux` | Improves web UI, adds missing result panels, fixes display issues |
| **Code Reviewer** | `/code-reviewer` | Reviews Python for scientific correctness (dropout, metrics, calibration) |

---

## Skills available

| Skill | Use for |
|---|---|
| `run-experiments` | Running any evaluation script |
| `write-thesis` | Writing thesis sections from JSON results |

---

## Current state of experiments (as of May 2026)

### Done ✓
- PCAM evaluation: confidence, mc_dropout, deep_ensemble, temperature_scaled
- Shift/OOD: blur, noise, jpeg, color × severities 1, 3, 5
- Cross-domain OOD: PCAM (ID) vs NCT-CRC-HE-100K (OOD)
- Conformal prediction: split conformal @ α = 0.05, 0.10, 0.20
- Aleatoric/epistemic decomposition via MC Dropout (T=30)
- ECE under shift: calibration degradation per severity

### Output files (evaluation/)
```
metrics_<method>_test.json          ← per-method full metrics
shift_ood_<method>_test.json        ← per-condition shift results
conformal_prediction__model-*.json  ← conformal coverage
aleatoric_epistemic__model-*.json   ← uncertainty decomposition
cross_domain_ood__<method>__*.json  ← PCAM vs NCT-CRC
ece_under_shift_summary.json        ← ECE table
```

### Known gaps (to address)
- [ ] Thesis chapters not yet written with final results
- [ ] Web UI missing: conformal panel, A/E decomposition panel, ECE-under-shift chart
- [ ] Deep ensemble conformal prediction not evaluated (only confidence + mc_dropout)
- [ ] Temperature scaling not included in cross-domain OOD evaluation

---

## Key results summary

| Method | Accuracy (n=256) | ECE (id_s0) | Error-det AUROC | Notes |
|---|---|---|---|---|
| Confidence | 0.848 | 0.076 | 0.825 | 1-MSP, overconfident on OOD |
| MC Dropout | 0.766* | 0.109 | 0.771 | *model.train() corrupts BatchNorm → 8pp accuracy drop |
| Deep Ensemble | 0.856 | 0.049 | 0.814 | Best calibration; most robust under shift |
| Temp. Scaling | 0.848 | 0.048 | 0.825 | Same rank order as Confidence — ECE +0.015 vs baseline |

> ECE values are from `shift_ood_*_test.json id_s0` (10-bin ECE, authoritative source for shift comparisons).  
> Error-detection AUROC values are from `metrics_*_test.json` using 1-MSP as uncertainty score.

**Aleatoric ratio**: 96.2% of uncertainty is aleatoric (data noise), only 3.8% epistemic  
**Conformal @ α=0.10**: Confidence=0.913 ✅ (set size=1.11), MC Dropout=0.878 🔴 (undercoverage, set size=1.26), Deep Ensemble=0.914 ✅ (set size=1.10)  
**Cross-domain AUROC**: MC Dropout best at 0.620 (PCAM vs NCT-CRC); Deep Ensemble MI inverted (0.529)  
**Temp. Scaling note**: T=1.432 fitted on val split; NLL improves but ECE worsens on test set

---

## Pipeline to run everything from scratch

```bash
cd /home/ubuntu/diplom && source venv/bin/activate

# 1. Full evaluation bundle
python3 experiments/run_evaluation_pipeline.py

# 2. Shift evaluation
for method in confidence mc_dropout deep_ensemble temperature_scaled; do
  python3 experiments/evaluate_shift_ood.py --method $method --split test
done

# 3. New analyses
python3 experiments/run_conformal.py
python3 experiments/run_aleatoric_epistemic.py
python3 experiments/run_ece_under_shift.py
python3 experiments/evaluate_cross_domain_ood.py --method confidence
python3 experiments/evaluate_cross_domain_ood.py --method mc_dropout
```
