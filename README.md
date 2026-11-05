# COMP 433 Final Project — Learning from Limited Data

**Team:** Massimo · Ryan · Ammar · Alvin
**Deadline:** Sun Dec 6, 2026
**Full plan:** [`docs/team_plan.md`](docs/team_plan.md) · **Instructions:** [`docs/project_instructions.pdf`](docs/project_instructions.pdf) · **Original testbed:** [`testbed/`](testbed/)
**Overleaf:** TODO (Alvin) · **TA email thread:** TODO

---

## Layout

```
common/                 shared code — changes go through a PR reviewed by Massimo
  data.py               load_instance(seed), TUNE_SEEDS, REPORT_SEEDS (splits = testbed, don't change)
  method.py             Method interface: fit(X, y) / predict(X)
  timing.py             GPU-synchronized Timer, gpu_name()
  harness.py            run(...) over seeds -> results CSV        (Massimo, Nov 9)
  plots.py              shared figure style                       (Massimo)
  augment.py            shared augmentation                       (Ryan, Nov 15)
c1/                     Challenge 1: no external data or models
  classical/            Massimo
  cnn/                  Ryan
  vit_scratch/          Ammar
  ssl/                  Alvin
c2/                     Challenge 2: external pretraining allowed
  frozen_features/      Massimo
  finetune/             Ryan
  peft/                 Ammar
  clip/                 Alvin
final_eval/             runs final methods on the instructor's split files (Massimo)
results/<name>.csv      one results file per person (no merge conflicts)
notebooks/<name>/       your own Colab driver notebooks
report/figures/         figures that go in the report (commit PNG/PDF here)
docs/                   team plan, project instructions
testbed/                original course notebook, untouched, for reference
```

## Rules (short version, details in the plan)

1. Get data **only** through `common.data.load_instance(seed)`. Tune on `TUNE_SEEDS` (0–4); report on `REPORT_SEEDS` (10–19).
2. Never train on validation images. No early stopping on validation; fix epochs in advance.
3. **C1:** no pretrained models, no external data, no other CIFAR-10 images. **C2:** no CIFAR-10-trained checkpoints; record each model's pretraining data.
4. Every method subclasses `common.method.Method` and does all its preprocessing inside `fit`/`predict`.
5. Real code lives in `.py` files. Notebooks are thin drivers: import, configure, call `harness.run`. Clear notebook outputs before committing.
6. Log every run to `results/<your-name>.csv`, including bad configs.

---

## Setup — local (for editing code)

```bash
git clone https://github.com/OWNER/comp433-final-project.git
cd comp433-final-project
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install torch torchvision                         # CPU build is fine for editing / small tests
pip install -e . -r requirements.txt
python -c "from common.data import load_instance; X, y, Xv, yv, c = load_instance(0); print(X.shape, Xv.shape, c)"
```

Expected output: `(100, 32, 32, 3) (700, 32, 32, 3) [...]` (first run downloads CIFAR-10, ~170 MB).

## Setup — Colab (for running experiments)

One-time: create a GitHub token (Settings → Developer settings → Personal access tokens → **Tokens (classic)**, scope `repo`). In Colab, open the 🔑 **Secrets** panel, add it as `GH_TOKEN`, and turn on notebook access.

First cell of every notebook:

```python
from google.colab import userdata
token = userdata.get("GH_TOKEN")
!git clone -q https://{token}@github.com/OWNER/comp433-final-project.git /content/repo
%cd /content/repo
!pip install -q -e . -r requirements.txt
!git config user.name "YOUR NAME" && git config user.email "YOUR_EMAIL"
```

Already cloned this session? Just `!git pull`.

Optional: mount Google Drive for things too big for git (feature caches, checkpoints):

```python
from google.colab import drive
drive.mount("/content/drive")
CACHE = "/content/drive/MyDrive/comp433_cache"   # add the team's shared folder as a shortcut in My Drive
```

## Git workflow

- `main` always works. Nobody pushes to `main` directly; open a pull request.
- Branch names: `<name>/<topic>`, e.g. `ryan/cnn-augment-sweep`, `alvin/clip-zero-shot`.
- PRs that touch `common/` need Massimo's review. PRs that only touch your own folders can be merged by you once they run.
- Commit your results file with the code that produced it, so `git_commit` in each row points to real code.

```bash
git checkout -b ammar/vit-spt-lsa
# ...edit, run...
git add c1/vit_scratch results/ammar.csv
git commit -m "c1/vit_scratch: SPT + LSA, patch size sweep"
git push -u origin ammar/vit-spt-lsa
# then open a PR on GitHub
```
