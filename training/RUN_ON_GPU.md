# Running the Krishimitra training on your GPU

Thanks for helping! This trains a rice-disease photo classifier. You'll mostly click **Run All** and wait about **1–2 hours** on an 8 GB card. Nothing in this folder touches anything outside it, except that it writes the finished model to `../models/`.

> Background reading (optional): `GUIDE.md` explains everything in plain language.

---

## What you received
```
Krishimitra/
├── training/                          ← you work in here
│   ├── train_disease_classifier.ipynb   the notebook to run
│   ├── krishi_data.py                   helper module (must stay next to the notebook)
│   ├── splits.csv                       fixed train/val/test split - DO NOT regenerate or edit
│   ├── requirements.txt
│   ├── RUN_ON_GPU.md                    this file
│   └── GUIDE.md / make_split.py         (not needed for the run)
└── models/                            ← created by the notebook; the result lands here
```
Keep this structure: the notebook saves the result to `../models/disease_classifier_v1/`.

---

## 1. Check the GPU (1 min)
```bash
nvidia-smi
```
Note two things:
- **the memory total** (for example `8192MiB` = 8 GB), which you need in step 5;
- **the "CUDA Version"** in the top-right corner, which is the highest CUDA your driver supports.

If `nvidia-smi` isn't found, install or update the NVIDIA driver first.

## 2. Create a Python environment (5 min)
Use Python **3.10–3.12**. From inside `training/`:

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```
**Linux:**
```bash
python3 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip
```
(Conda works too, if you prefer it.)

## 3. Install PyTorch **with CUDA**, then the rest (5–10 min)
1. Go to <https://pytorch.org/get-started/locally/> and select *Stable / your OS / Pip / Python / CUDA*. Pick a CUDA version **≤ the one `nvidia-smi` showed**. Run the command it gives you, which looks like this:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```
2. Install the rest:
   ```bash
   pip install -r requirements.txt
   ```
3. **Verify the GPU is visible.** This must print `True`:
   ```bash
   python -c "import torch; print(torch.cuda.is_available(), torch.__version__)"
   ```
   If it prints `False`, you installed the CPU build. Uninstall torch and torchvision (`pip uninstall torch torchvision`) and repeat step 3.1.

## 4. Get the dataset (~1 GB)
Choose **one** of these:
- **A. Let the notebook download it.**
  1. Make a free <https://www.kaggle.com> account.
  2. Open <https://www.kaggle.com/competitions/paddy-disease-classification>, go to *Data* and click **accept the rules**.
  3. Go to *Settings → API → Create New Token*. This downloads a file called `kaggle.json`.
  4. Put it in `~/.kaggle/kaggle.json` (Windows: `C:\Users\<you>\.kaggle\kaggle.json`). On Linux, also run `chmod 600 ~/.kaggle/kaggle.json`.

  The notebook downloads and unzips the data on its first run.
- **B. Copy it.** If you were given the data folder, put it at `training/data/paddy/` so that `training/data/paddy/train.csv` and `training/data/paddy/train_images/` exist.

## 5. Set the config for your VRAM
Open the notebook:
```bash
jupyter notebook train_disease_classifier.ipynb      # or open it in VS Code
```
In the **first code cell (§0 Config)**, set these three values from your VRAM, and **change nothing else**:

| VRAM | `IMG_SIZE` | `BATCH` | `ACCUM_STEPS` |
|---|---|---|---|
| 6 GB | 320 | 12 | 3 |
| **8 GB (default)** | **384** | **16** | **2** |
| 12 GB | 384 | 24 | 1 |
| 16 GB+ | 448 | 32 | 1 |

Make sure `SMOKE_TEST = False`.

> **Optional 2-minute rehearsal:** set `SMOKE_TEST = True`, Run All, check that it reaches the end without errors, then set it back to `False`. The rehearsal writes only to `outputs/smoke/`.

## 6. Run All
Use *Kernel → Restart & Run All*. What you'll see:
- **§1** prints your GPU name. If you see a big `NO CUDA GPU FOUND` warning, **stop** and go back to step 3.
- **§2** downloads the data (first run only) and shows class counts and sample photos.
- **§5** prints one line per epoch, for example:
  ```
  [B  7/25] train loss 0.712 acc 0.861 | val loss 0.301 top1 0.948 top3 0.991 F1 0.941 | 3.2 min *BEST*
  ```
  - About 2 + up to 25 epochs.
  - It stops early when val stops improving; that's normal.
  - Train accuracy is *meant* to look lower than val accuracy, because the training images are deliberately distorted.
- **§9** ends with:
  ```
  SEND BACK: .../models/disease_classifier_v1.zip + this executed notebook
  ```

You can leave it running. Make sure the computer won't **sleep**.

## 7. Send back
1. **`models/disease_classifier_v1.zip`** (about 80 MB)
2. **The executed notebook.** Save it (Ctrl+S) *after* it finishes, so the outputs and plots are included.
3. `training/outputs/history.csv` (tiny)

Please **don't** send `outputs/checkpoints/` (large and not needed) or `kaggle.json` (it's a password).

---

## Troubleshooting
| Problem | Fix |
|---|---|
| `CUDA out of memory` | Use the next smaller row of the VRAM table. Restart the kernel, then Run All. |
| The session crashed, the PC rebooted, or you stopped it | Just **Run All again**. With `RESUME = True` it continues from the last finished epoch (`outputs/checkpoints/last.pt`). |
| `Loss became nan ...` (training diverged) | Halve `LR_HEAD`, `LR_BACKBONE` and `LR_STAGE_A`, then Run All (it resumes). Tell us it happened. |
| `Kaggle API not set up` or `403 Forbidden` | Check that `kaggle.json` is in the right place **and** that you accepted the competition rules (step 4A). |
| `... images in splits.csv are missing` | The data folder is in the wrong place. You need `training/data/paddy/train_images/<class>/<id>.jpg`. |
| `ModuleNotFoundError: krishi_data` | Start Jupyter from inside `training/`, and keep `krishi_data.py` next to the notebook. |
| Windows: the first epoch hangs or there are worker errors | In the config, set `NUM_WORKERS = 0`. It will be slower, but it always works. |
| Each epoch takes more than 15 min | Probably running on the CPU. Check that §1 printed your GPU name. |
| You want to start completely fresh | Delete `training/outputs/checkpoints/`, then Run All. |

If anything else goes wrong, send a screenshot of the error and the **last ~30 lines** of the cell output.
