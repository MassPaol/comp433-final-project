"""C2 results table. Needs pandas and common/sweep.py.

Reads results/alvin.csv (accuracy, train and total inference time) and
results/alvin_timing.csv (preprocess / encode split).

    python c2_report_table_local.py --best <best_id> --fast <fast_id>

Writes report/tables/c2_clip_results.md and c2_clip_results.tex next to this file.
"""
import argparse
import os

import pandas as pd

from common.sweep import hw_of

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_SEEDS = list(range(10, 20))
N_TEMPLATES = 18
CHALLENGE = "c2"
N_IMAGES = 700  # validation images per problem instance

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
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found.")
    df = pd.read_csv(path)
    if "notes" not in df.columns:
        raise SystemExit(f"{path}: no notes column; columns are {list(df.columns)}.")
    df["notes"] = df["notes"].fillna("")
    return df


def load_timing(path):
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found. Rerun the stages to create it.")
    t = pd.read_csv(path)
    t["hw"] = t["hw"].astype(str)
    # If a run was interrupted and redone, keep the latest row per (config, seed, hw).
    return t.drop_duplicates(["config_id", "seed", "hw"], keep="last")


def summarize(d, t):
    acc = d["val_acc"] * 100
    return {
        "acc_mean": acc.mean(),
        "acc_std": acc.std(ddof=0),
        "train_mean": d["train_time_s"].mean(),
        "train_median": d["train_time_s"].median(),
        "infer_700": d["infer_time_s"].mean(),
        "per_image_ms": d["infer_time_s"].mean() / N_IMAGES * 1000,
        "encode_ms": t["infer_encode_s"].astype(float).mean() / N_IMAGES * 1000,
    }


def to_markdown(rows):
    out = ["| Config | Pipeline | Val. accuracy (%) | Train time (s, mean) | Train time (s, median) "
           "| Inference time (s / 700 img) | Per image (ms) | Encode per image (ms) |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| *{r['label']}* | {r['pipeline']} | {r['acc_mean']:.2f} ± {r['acc_std']:.2f} | "
                   f"{r['train_mean']:.3f} | {r['train_median']:.3f} | {r['infer_700']:.3f} | "
                   f"{r['per_image_ms']:.2f} | {r['encode_ms']:.2f} |")
    return "\n".join(out) + "\n"


def tex_escape(s):
    return (s.replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
             .replace("_", r"\_").replace("#", r"\#").replace("±", r"$\pm$"))


def to_latex(rows):
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"Config & Pipeline & Val. acc. (\%) & Train (s, mean) & Train (s, median) "
             r"& Inference (s / 700) & Per image (ms) & Encode (ms / image) \\", r"\midrule"]
    for r in rows:
        lines.append(f"\\emph{{{r['label']}}} & {tex_escape(r['pipeline'])} & "
                     f"{r['acc_mean']:.2f} $\\pm$ {r['acc_std']:.2f} & {r['train_mean']:.3f} & "
                     f"{r['train_median']:.3f} & {r['infer_700']:.3f} & {r['per_image_ms']:.2f} & "
                     f"{r['encode_ms']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Final C2 CLIP configurations on report seeds 10--19 "
              r"(mean $\pm$ std over 10 problem instances). Times measured on a Colab T4. "
              r"Train time includes loading CLIP inside \texttt{fit} for every method, so train "
              r"times are comparable. Encode time is the CLIP forward pass only; per-image "
              r"inference also includes CPU preprocessing.}",
              r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description="Build the C2 results table from the CSVs.")
    p.add_argument("--results", default=os.path.join(HERE, "results", "alvin.csv"))
    p.add_argument("--timing", default=os.path.join(HERE, "results", "alvin_timing.csv"))
    p.add_argument("--hw", default="colab-t4", help="hardware tag used in the notes column")
    p.add_argument("--best", required=True, help="best config id from --select")
    p.add_argument("--fast", required=True, help="fast config id from --select")
    p.add_argument("--outdir", default=os.path.join(HERE, "report", "tables"))
    a = p.parse_args()

    df = load_results(a.results)
    df = df[(df["challenge"] == CHALLENGE) & df["seed"].isin(REPORT_SEEDS)
            & (df["notes"].map(hw_of) == a.hw)]
    timing = load_timing(a.timing)
    timing = timing[(timing["challenge"] == CHALLENGE) & timing["seed"].isin(REPORT_SEEDS)
                    & (timing["hw"] == a.hw)]

    rows = []
    for label, cid in [("best", a.best), ("fast", a.fast)]:
        d = df[df["config_id"] == cid]
        t = timing[timing["config_id"] == cid]
        if len(d) != len(REPORT_SEEDS):
            raise SystemExit(f"{cid}: {len(d)} of {len(REPORT_SEEDS)} reporting seeds found for hw={a.hw}")
        if len(t) != len(REPORT_SEEDS):
            raise SystemExit(f"{cid}: {len(t)} of {len(REPORT_SEEDS)} timing rows found for hw={a.hw}")
        pipeline = DESCRIPTIONS.get(cid, f"See c2/clip/configs.py for the settings of {cid}.")
        rows.append({"label": label, "pipeline": pipeline, **summarize(d, t)})

    os.makedirs(a.outdir, exist_ok=True)
    md, tex = to_markdown(rows), to_latex(rows)
    md_path = os.path.join(a.outdir, "c2_clip_results.md")
    tex_path = os.path.join(a.outdir, "c2_clip_results.tex")
    with open(md_path, "w") as f:
        f.write(md)
    with open(tex_path, "w") as f:
        f.write(tex)
    print(md)
    print("wrote", md_path, "and", tex_path)


if __name__ == "__main__":
    main()