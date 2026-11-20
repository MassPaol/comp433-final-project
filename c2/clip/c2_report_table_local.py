"""C2 results table from results/alvin.csv. Needs only pandas and numpy (no torch, no open_clip).

Run in PyCharm: open this file and press Run, or from a terminal:
    python c2_report_table_local.py
    python c2_report_table_local.py --results path/to/alvin.csv --best c2_wiseft_m0.5 --fast c2_zeroshot

Writes report/tables/c2_clip_results.md and c2_clip_results.tex next to this file.
"""
import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_SEEDS = list(range(10, 20))
N_TEMPLATES = 18
CHALLENGE = "c2"

DESCRIPTIONS = {
    "c2_zeroshot": (f"Zero-shot: {N_TEMPLATES} prompt templates per class, averaged, renormalized. "
                    "Cosine similarity to the class embeddings. No training."),
    "c2_wiseft_m0.5": ("WiSE-FT-style: probe head (300 steps, learning rate 0.001) mixed with the "
                       "zero-shot head at weight 0.5. Head-level interpolation, encoder frozen."),
    "c2_wiseft_m0.75": ("WiSE-FT-style: probe head (300 steps, learning rate 0.001) mixed with the "
                        "zero-shot head at weight 0.75. Head-level interpolation, encoder frozen."),
    "c2_probe_lr0.001_s300": ("Linear probe: head initialized from the text embeddings, trained "
                              "300 steps at learning rate 0.001."),
    "c2_probe_lr0.01_s100": ("Linear probe: head initialized from the text embeddings, trained "
                             "100 steps at learning rate 0.01."),
    "c2_tip_a0.5_b1": "Tip-Adapter: zero-shot logits plus a cache of the 100 training embeddings (alpha 0.5, beta 1). No training.",
    "c2_tip_a0.5_b5.5": "Tip-Adapter: zero-shot logits plus a cache of the 100 training embeddings (alpha 0.5, beta 5.5). No training.",
    "c2_tipf_lr0.0001_s100": "Tip-Adapter-F: cache keys fine-tuned for 100 steps at learning rate 0.0001 (alpha 0.5, beta 1).",
    "c2_tipf_lr0.001_s100": "Tip-Adapter-F: cache keys fine-tuned for 100 steps at learning rate 0.001 (alpha 0.5, beta 1).",
}


def load_results(path):
    df = pd.read_csv(path)
    # Tolerate the mangled header (e.g. 'notesc2') left by the earlier commit
    bad = [c for c in df.columns if c.startswith("notes") and c != "notes"]
    if bad:
        df = df.rename(columns={bad[0]: "notes"})
    if "notes" not in df.columns:
        raise SystemExit(f"{path}: no notes column; columns are {list(df.columns)}")
    df["notes"] = df["notes"].fillna("")
    return df


def summarize(d):
    acc = d.val_acc * 100
    return {
        "acc_mean": acc.mean(),
        "acc_std": acc.std(ddof=0),
        "train_mean": d.train_time_s.mean(),
        "train_median": d.train_time_s.median(),
        "infer_700": d.infer_time_s.mean(),
        "per_image_ms": d.infer_time_s.mean() / 700 * 1000,
    }


def to_markdown(rows):
    out = ["| Config | Pipeline | Val. accuracy (%) | Train time (s, mean) | Train time (s, median) "
           "| Inference time (s / 700 img) | Per image (ms) |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| *{r['label']}* | {r['pipeline']} | {r['acc_mean']:.2f} ± {r['acc_std']:.2f} | "
                   f"{r['train_mean']:.3f} | {r['train_median']:.3f} | {r['infer_700']:.3f} | "
                   f"{r['per_image_ms']:.2f} |")
    return "\n".join(out) + "\n"


def tex_escape(s):
    return (s.replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
             .replace("_", r"\_").replace("#", r"\#").replace("±", r"$\pm$"))


def to_latex(rows):
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Config & Pipeline & Val. acc. (\%) & Train (s, mean) & Train (s, median) "
             r"& Inference (s / 700) & Per image (ms) \\", r"\midrule"]
    for r in rows:
        lines.append(f"\\emph{{{r['label']}}} & {tex_escape(r['pipeline'])} & "
                     f"{r['acc_mean']:.2f} $\\pm$ {r['acc_std']:.2f} & {r['train_mean']:.3f} & "
                     f"{r['train_median']:.3f} & {r['infer_700']:.3f} & {r['per_image_ms']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Final C2 CLIP configurations on report seeds 10--19 "
              r"(mean $\pm$ std over 10 class pairs). Times measured on a Colab T4. "
              r"The first call of each run includes model loading, so the mean train time is "
              r"inflated; the median is the more representative value.}",
              r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description="Build the C2 results table from the CSV.")
    p.add_argument("--results", default=os.path.join(HERE, "results", "alvin.csv"))
    p.add_argument("--hw", default="colab-t4", help="hardware tag used in the notes column")
    p.add_argument("--best", default="c2_wiseft_m0.5")
    p.add_argument("--fast", default="c2_zeroshot")
    p.add_argument("--outdir", default=os.path.join(HERE, "report", "tables"))
    a = p.parse_args()

    df = load_results(a.results)
    df = df[(df.challenge == CHALLENGE) & df.seed.isin(REPORT_SEEDS)
            & df.notes.str.contains(f"hw={a.hw}")]

    rows = []
    for label, cid in [("best", a.best), ("fast", a.fast)]:
        d = df[df.config_id == cid]
        if len(d) != len(REPORT_SEEDS):
            raise SystemExit(f"{cid}: {len(d)} of {len(REPORT_SEEDS)} reporting seeds found for hw={a.hw}")
        pipeline = DESCRIPTIONS.get(cid, f"See c2/clip/configs.py for the settings of {cid}.")
        rows.append({"label": label, "pipeline": pipeline, **summarize(d)})

    os.makedirs(a.outdir, exist_ok=True)
    md, tex = to_markdown(rows), to_latex(rows)
    with open(os.path.join(a.outdir, "c2_clip_results.md"), "w") as f:
        f.write(md)
    with open(os.path.join(a.outdir, "c2_clip_results.tex"), "w") as f:
        f.write(tex)
    print(md)
    print("wrote", os.path.join(a.outdir, "c2_clip_results.md"), "and c2_clip_results.tex")


if __name__ == "__main__":
    main()
