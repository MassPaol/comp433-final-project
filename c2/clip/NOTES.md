# C2 CLIP: experiment log

Seeds: tuning 0–4 for every search decision, reporting 10–19 for every reported number.
Metric: validation accuracy on the 700 held-out images per problem instance. Mean ± std over seeds
(std = spread across class pairs). Time = train + inference seconds per seed.
Raw rows: `results/alvin.csv` (challenge = c2).

Write one section per stage, after its results are in.

## Setup decisions (fixed before any search)
- Checkpoint: ViT-B-32, pretrained `openai` (OpenAI WIT web dataset). Approved by the prof.
  Model card link: (add before reporting).
- Preprocessing: open_clip's transform, not `timm.data.resolve_model_data_config` (see README).
- Zero-shot templates: 18 templates in `loadClip.py`. Frozen before any run. Changing them is a new config.
- Probe head: initialized from the text embeddings scaled by 100; AdamW, weight decay 0, full batch,
  fixed steps. No early stopping.
- Selection rule: best = highest mean tuning accuracy; fast = lowest train + inference time among configs
  within **1 point** of best. (Written here before the search; the C1 rule uses different thresholds,
  so state this difference in the report.)

## <date> · zeroshot
**Result on tuning seeds:**

## <date> · probe: learning rate × steps
**Why:** does training the head beat zero-shot on 100 images?
**Results:**
**Decision → PROBE_BASE:** (lr, steps)

## <date> · wiseft: mix weight
**Why:** does interpolating the zero-shot and probe heads beat either alone?
**Results:**
**Decision:**

## <date> · tip: alpha × beta
**Why:** how much does the cache help, and how sharp should its affinity be?
**Results:**
**Decision → TIP_BASE:** (alpha, beta)

## <date> · tipf: key learning rate × steps
**Why:** does fine-tuning the cache keys help over the fixed cache?
**Results:**
**Decision:**

## <date> · Selection (rule above)
- probe_best / probe_fast:
- wiseft_best / wiseft_fast:
- tip_best / tip_fast:
- tipf_best / tipf_fast:
- zeroshot (no search):
- close calls:

## <date> · Reporting seeds 10–19 (frozen)
| config | acc (%) | train (s) | inference (ms / image) | paired diff vs zero-shot (pp) |
|---|---|---|---|---|
| c2_zeroshot | | | | |
| | | | | |

# Source log

| source | used for |
|---|---|
| Radford et al. 2021, *Learning transferable visual models from natural language supervision* (CLIP), ICML | zero-shot classification, prompt templates and ensembling, the data-overlap analysis (cite the section) |
| Zhang et al. 2022, *Tip-Adapter: Training-free adaption of CLIP for few-shot classification*, ECCV | cache formula (affinity, alpha, beta); official code github.com/gaopengcuhk/Tip-Adapter; Tip-Adapter-F |
| Wortsman et al. 2022, *Robust fine-tuning of zero-shot models* (WiSE-FT), CVPR | weight interpolation between zero-shot and fine-tuned models (applied here at the head level) |
| Zhou et al. 2022, *Learning to prompt for vision-language models* (CoOp), IJCV | background on prompt learning; not implemented |
| Caron et al. 2021, *Emerging properties in self-supervised vision transformers* (DINO), ICCV | fallback only; not used |
| open_clip (version pinned in `requirements.txt`) | CLIP model, preprocessing, tokenizer |
