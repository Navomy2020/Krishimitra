# Training the Krishimitra Paddy Disease Classifier: A Beginner's Guide

This guide is for someone who has **never trained a neural network**. It explains what we do and *why* we do it. Read it alongside the code in this folder.

The plan behind this work is in `../crop_health_cnn_plan.md`. This guide is the "learn it" companion to that plan.

**Goal:** a model that looks at a phone photo of a rice plant and says which of 10 conditions it most likely has, together with a confidence that we can trust. The target is at least 93% top-1 accuracy, up from 79.5% for the old model in `../paddy_model/`.

---

## Contents
1. [What a CNN classifier is](#1-what-a-cnn-classifier-is)
2. [Finding and downloading the dataset](#2-finding-and-downloading-the-dataset)
3. [Looking at the data (EDA)](#3-looking-at-the-data-eda)
4. [Splitting into train / val / test](#4-splitting-into-train--val--test)
5. [Preprocessing and augmentation](#5-preprocessing-and-augmentation)
6. [Training](#6-training)
7. [Evaluation](#7-evaluation)
8. [Calibration](#8-calibration)
9. [Export and use](#9-export-and-use)
10. [Next improvements](#10-next-improvements)
11. [Common mistakes to avoid](#11-common-mistakes-to-avoid)
12. [Implementation log](#12-implementation-log)

---

## 1. What a CNN classifier is

### Image → numbers → probabilities
A computer sees a photo as a grid of numbers. A 384×384 colour photo is 384 × 384 × 3 (red, green, blue) ≈ 442,000 numbers between 0 and 255.

A **classifier** is a function that takes those numbers and returns **10 probabilities**, one per class, that add up to 1:

```
photo ──► [ neural network ] ──► bacterial_leaf_blight 0.02
                                  bacterial_leaf_streak 0.01
                                  bacterial_panicle_blight 0.00
                                  blast                 0.81  ◄── prediction
                                  brown_spot            0.12
                                  dead_heart            0.00
                                  downy_mildew          0.01
                                  hispa                 0.01
                                  normal                0.02
                                  tungro                0.00
```

Our 10 classes are the same as the old model's (`../paddy_model/class_names.json`):
`bacterial_leaf_blight, bacterial_leaf_streak, bacterial_panicle_blight, blast, brown_spot, dead_heart, downy_mildew, hispa, normal, tungro`.

### What makes it "convolutional"
A **CNN (Convolutional Neural Network)** slides small filters (for example 3×3 pixels) across the image.
- The **early layers** learn simple patterns such as edges, colour blobs and textures.
- The **middle layers** combine these into shapes such as lesion borders, stripes and spots.
- The **late layers** combine those into concepts such as "a diamond-shaped grey lesion with a brown border", which suggests blast.

The final layer is called the **head**. It is a small layer that turns the learned features into the 10 class scores (called **logits**). A **softmax** then turns the logits into probabilities.

### Training = adjusting millions of numbers
The network has millions of adjustable numbers called **weights**. Training works like this:
1. Show it a batch of photos with known labels.
2. Measure how wrong it is. This number is the **loss**: lower is better.
3. Use calculus (**backpropagation**) to work out which direction to nudge each weight to reduce the loss.
4. Nudge the weights a little. The size of the nudge is set by the **learning rate**.
5. Repeat thousands of times.

### Transfer learning: why we don't start from scratch
Training a CNN from scratch needs millions of images. We have about 10,000.

Instead we use **transfer learning**. We start from a network that someone already trained on **ImageNet** (more than a million everyday photos). It already knows edges, textures and shapes, so we only need to teach it *rice diseases*.

We use **EfficientNetV2-S** (`tf_efficientnetv2_s.in21k_ft_in1k`). It is accurate for its size, runs fast enough on a CPU server, and is available in the `timm` library. Its name means:
- pretrained on ImageNet-21k (14 million images, 21,000 classes),
- then fine-tuned on ImageNet-1k.

> **Lesson from a reference project:** another project on this same dataset found that a **frozen** backbone (only the head trained) "did not converge" well. When they fine-tuned the **whole** network, they reached about 93.7%. Our old model probably under-performed for this reason. So we train the head briefly first and then unfreeze everything. See §6.

### Why PyTorch + timm (and not Keras like the old model)?
- `timm` gives us hundreds of strong pretrained backbones through a single line of code.
- PyTorch exports cleanly to **ONNX** (§9). Our FastAPI server can run ONNX on a CPU without installing PyTorch or TensorFlow.

---

## 2. Finding and downloading the dataset

### The dataset: Paddy Doctor
We use the Kaggle competition **Paddy Doctor: Paddy Disease Classification** (`paddy-disease-classification`).

| What | Detail |
|---|---|
| Images | 10,407 labelled field photos (480×640, taken with phones in Tamil Nadu) |
| Classes | 10 (9 diseases or pests + `normal`) |
| `train.csv` | Columns `image_id, label, variety, age` (`variety` = rice variety, `age` = crop age in days) |
| `train_images/<label>/` | One folder per class |
| `test_images/` | About 3,469 images **with no labels**. This is the competition's hidden test set, so it is **useless to us** for measuring accuracy. |

Because the official test set has no labels, **we make our own labelled test set** by holding back 15% of `train_images` (§4).

### Step by step: get the data
1. **Create a Kaggle account** at <https://www.kaggle.com> (free).
2. **Accept the competition rules**. Open <https://www.kaggle.com/competitions/paddy-disease-classification>, go to the *Data* tab and click **"Join competition / I understand and accept"**. Without this step the download returns a `403 Forbidden` error.
3. **Create an API token**. Go to *Kaggle → profile picture → Settings → API → "Create New Token"*. This downloads a file called `kaggle.json`.
4. **Put the token where the `kaggle` tool looks for it**:
   - Windows: `C:\Users\<your-user>\.kaggle\kaggle.json`
   - Linux/macOS: `~/.kaggle/kaggle.json`, then run `chmod 600 ~/.kaggle/kaggle.json`

   ⚠️ `kaggle.json` is a password. Never commit it (`.gitignore` already blocks it) and never paste it into chat.
5. **Install the tools and download the data** (from the `D:\Krishimitra\training` folder):
   ```powershell
   pip install -r requirements.txt
   kaggle competitions download -c paddy-disease-classification -p data
   ```
   This downloads a single zip file of about 1 GB into `data/`.
6. **Unzip it**:
   ```powershell
   Expand-Archive data\paddy-disease-classification.zip -DestinationPath data\paddy
   ```
   The resulting layout is:
   ```
   training/data/paddy/
       train.csv
       train_images/bacterial_leaf_blight/100023.jpg ...
       test_images/ ...          (unlabelled, ignored)
       sample_submission.csv     (ignored)
   ```

The notebook can also do steps 5–6 automatically if `data/paddy` is missing and the token is in place.

> ⚠️ **Two unzip gotchas (both happened on our first try):**
> 1. **Wait for the unzip to finish.** It writes about 13,900 files and takes several minutes. If you start too early, whole classes are missing; on our first try `normal` and `tungro` hadn't appeared yet. Check that `train_images/` has **10 folders** and **10,407 images** in total.
> 2. **Check for an extra folder level.** Some unzip tools create `data/paddy/paddy-disease-classification/train.csv`. Move the contents up so that the file is at `data/paddy/train.csv`.
>
> Quick check (Git Bash): `find data/paddy/train_images -type f | wc -l` should print `10407`.

### Other data sources for later
These are for when we add Kerala-specific classes (for example sheath blight, false smut or leaf folder):
- **Kerala Agricultural University (KAU) / KVK** extension photos. Ask about licence and permission.
- **Mendeley Data** rice disease datasets. Check each dataset's licence (CC-BY is fine).
- **Our own Kerala field photos.** These are the most valuable for testing, because Paddy Doctor photos come from Tamil Nadu.

---

## 3. Looking at the data (EDA)
**EDA (Exploratory Data Analysis)** means looking at the data *before* training. Most "the model is bad" problems are really data problems. Always check these things:

| Check | Why | Where we do it |
|---|---|---|
| **Class counts** | If one class has 2,000 images and another has 300, the model may learn to ignore the small class. That's why we use macro-F1 (§7) and a *stratified* split (§4). | `make_split.py` prints them; the notebook plots them. Paddy Doctor ranges from about 330 (`bacterial_panicle_blight`) to about 1,750 (`normal`) images. |
| **Sample grid** | Look at about 16 random photos per class. Can *you* tell the classes apart? Are some photos blurry, mostly soil, or wrongly labelled? | Notebook §2. |
| **Image sizes** | Mixed sizes are fine because we resize anyway, but a few huge or tiny images can be a sign of junk. | `make_split.py` prints the most common sizes. Almost all are 480×640. |
| **Unreadable files** | A single corrupt JPEG crashes training an hour in. | `make_split.py` opens every image and drops the broken ones. |
| **Duplicates and near-duplicates** | The same photo (or the same leaf shot twice) in both train and test makes test accuracy look better than it is. | `make_split.py` (see below). |
| **Metadata** | `variety` and `age` hint at how the data was collected. For example, if almost all `dead_heart` photos come from one variety, the model might learn the variety instead of the disease. | Notebook §2 cross-tabs. |

### How near-duplicates are found: perceptual hashing
An ordinary file hash (such as MD5) changes completely if a single pixel changes. A **perceptual hash (pHash)** is designed so that *visually similar images get similar hashes*:
1. Shrink the image to 32×32 greyscale.
2. Take a DCT, a frequency transform similar to the one JPEG uses.
3. Keep the 8×8 lowest frequencies, which describe the overall shape and light pattern.
4. For each of the 64 values, write 1 if it is above the median, else 0. This gives a **64-bit fingerprint**.

Two images are compared by their **Hamming distance**, the number of bits that differ:
- **0** means the images are identical or nearly so (a re-saved or slightly brightened copy).
- **≤ 4** means a near-duplicate (the same shot re-compressed, slightly cropped or resized). This is our default `--threshold 4`.
- **About 32** is typical for two unrelated photos.

The script compares all pairs (about 54 million for roughly 10k images) using a fast matrix trick: `distance = |a| + |b| − 2·(a·b)`.

## 4. Splitting into train / val / test

### Why three sets?
| Set | Share | Used for | Analogy |
|---|---|---|---|
| **train** | 70% | The model learns its weights from these images | Textbook exercises |
| **val** (validation) | 15% | Choosing *settings*: which epoch to keep, when to stop, the calibration temperature | Mock exams |
| **test** | 15% | **One final, honest score**, measured once at the very end | The real exam |

Every decision we make after looking at val results slowly "tunes" us to val, so the val score becomes a little optimistic. That's why the **test set is touched exactly once**. If you look at the test score, change something and re-test, it has become a second val set and its score no longer means anything.

### Data leakage
**Leakage** is any way information from val or test sneaks into training. The classic case here is a near-duplicate pair split across train and test. The model has effectively *seen* the test photo, so it gets it right for the wrong reason. In the field, the model will only ever see new plants, so leaked accuracy is accuracy you will not get in practice.

### What `make_split.py` does, step by step
1. **Reads `train.csv`**. The path of each image is built as `train_images/<label>/<image_id>`.
2. **Hashes every image** in parallel and caches the result in `data/paddy/phash_cache.csv`, so re-runs are instant. Unreadable images are dropped.
3. **Groups near-duplicates** with *union-find*: if A≈B and B≈C, then A, B and C are one group, even if A and C are slightly further apart.
4. **Cleans the groups:**
   - **Conflicting labels:** if a group contains *different* labels (the same photo labelled `blast` in one place and `brown_spot` in another), every image in it is dropped. We can't know which label is right, and training on both teaches the model nonsense.
   - **Exact duplicates** (distance 0, same label): only the first image is kept. Extra copies would just give that photo extra weight.
   - **Other near-duplicates** are kept, but the **whole group always goes into the same split**.
5. **Stratified group split:** for *each class separately*, the groups are shuffled (with seed 42) and assigned to test until about 15% of that class's images are there, then to val until about 15%, and the rest go to train. *Stratified* means every class has the same proportions in every split.
6. **Sanity checks:** the script fails if an image_id appears twice, if any duplicate group spans more than one split, or if any class's share in a split differs from its overall share by more than 1%.
7. **Writes two files:**
   - `splits.csv` with columns `image_id, label, variety, age, path, group_id, split`.
   - `duplicate_report.csv`, which lists every removed image with the reason (`unreadable`, `label_conflict_in_duplicate_group` or `exact_duplicate`) and, for an exact duplicate, which image was kept instead.

### How to run it (once, on your laptop)
```powershell
cd D:\Krishimitra\training
python make_split.py
```
It takes about 2–3 minutes to hash about 10k images on a CPU. Then:
- Open `duplicate_report.csv` and look at a few of the "label conflict" images. This is real EDA: you'll see how noisy field labels can be.
- **Commit `splits.csv` and `duplicate_report.csv`.** Your friend must train on *exactly* this split. Never re-generate it after training has started, or the test set changes.

Useful options:
- `--threshold 4` sets how strict the near-duplicate test is. Higher catches more but risks merging genuinely different photos.
- `--seed 42` controls which images land in which split.
- `--workers 4` sets how many CPU processes hash images in parallel.

## 5. Preprocessing and augmentation
All of this lives in `krishi_data.py`, in `build_transforms()` and `PaddyDataset`.

> **Why a `.py` module and not notebook cells?** On Windows, PyTorch's `DataLoader` starts its worker processes with "spawn": each worker re-imports the code it needs. Anything defined *inside a notebook* can't be re-imported, so you get a confusing `AttributeError: Can't get attribute 'PaddyDataset'` error. Keeping the dataset and transforms in a real file avoids that.

### Preprocessing: what *every* image gets (train, val, test and later the app)
1. **Resize and crop to `IMG_SIZE`** (384 by default). The network needs a fixed size, and bigger images show more lesion detail but cost more memory.
2. **Scale pixels from 0–255 to 0–1.**
3. **Normalise** by subtracting a mean and dividing by a std, per colour channel. We must use **the same numbers the backbone was pretrained with**, so that we feed it the same "units" it learned on.
   - Many models use the ImageNet statistics, mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`.
   - **Our EfficientNetV2-S (`tf_…in21k_ft_in1k`) uses mean = std = `(0.5, 0.5, 0.5)`.** Using the wrong statistics doesn't crash anything; it just quietly costs accuracy.
   - So the notebook reads the correct values from timm with `kd.model_norm_stats(MODEL_NAME)` instead of hard-coding them.

These exact numbers are written to `preprocess.json` at export. The FastAPI server **must** apply the same steps, or the model sees images unlike anything it was trained on and its accuracy collapses silently.

### Augmentation: random changes, **training images only**
The model sees each training photo about 25 times (once per epoch). Without augmentation it starts **memorising** those exact pixels, which is overfitting. Augmentation shows a slightly different version each time, so the model has to learn the *disease* rather than the photo. We choose changes that **mimic how farmers' phone photos really vary**:

| Transform | Real-world variation it simulates |
|---|---|
| `RandomResizedCrop(scale 0.6–1)` | Standing closer or further away; the leaf is off-centre |
| `RandomHorizontalFlip`, `RandomVerticalFlip` | Leaves have no "up". A lesion is a lesion either way up. |
| `RandomRotation(15°)` | A tilted phone |
| `ColorJitter` (brightness, contrast, saturation, slight hue) | Midday sun vs shade vs overcast; different phone cameras |
| `GaussianBlur` (20% of images) | Out-of-focus or shaky shots |
| `JPEG(quality 40–95)` (50% of images) | Photos forwarded over **WhatsApp**, which compresses them heavily |

What we deliberately **don't** do:
- **Strong hue shifts.** Colour *is* a symptom: yellow-orange means tungro, brown spots mean brown spot. Changing the hue a lot would erase real information.
- **Augmenting val or test.** Eval transforms are *only* resize, centre-crop and normalise, so scores are stable and comparable between epochs (see the common mistakes in §11).

### Mixup and CutMix (in the notebook, training only)
On 30% of batches (via `timm.data.Mixup`), two images are blended (**Mixup**) or a patch of one is pasted into another (**CutMix**). The label becomes a matching mix, for example "70% blast, 30% brown_spot". This discourages over-confident predictions and is a strong regulariser.

Because the labels become *soft* (fractions instead of one class), training uses `SoftTargetCrossEntropy`. This is also why the training loader uses `drop_last=True`: Mixup needs full, even-sized batches.

## 6. Training

### Vocabulary
| Term | Meaning |
|---|---|
| **Batch** | How many images go through the network at once (`BATCH=16`). It is limited by GPU memory (VRAM). |
| **Gradient accumulation** | Add up the gradients of `ACCUM_STEPS=2` batches before updating the weights. This acts like a batch of 32 without needing 32 images' worth of VRAM. |
| **Step** | One weight update. |
| **Epoch** | One full pass over all training images. |
| **Loss** | How wrong the model is (cross-entropy). It should go down. |
| **Learning rate (LR)** | The size of each weight nudge. Too high and training explodes (NaN loss); too low and it learns very slowly. |

### The two-stage recipe
1. **Stage A: head only** (`EPOCHS_HEAD=2`, backbone frozen with `set_backbone_trainable(model, False)`).

   The new 10-class head starts with random weights. If we trained everything at once, its large, random gradients would flow back and damage the carefully pretrained backbone. So first the head learns, cheaply, to use the existing features.
2. **Stage B: fine-tune everything** (`EPOCHS_FT=25`, all layers unfrozen).

   Now the backbone adapts its features to rice leaves. This is the step the old 79.5% model was missing.

### The ingredients (and why each one)
| Ingredient | Code | Why |
|---|---|---|
| **AdamW** optimiser | `build_optimizer()` | Adam adapts the step size per weight. The "W" (decoupled weight decay 0.05) gently pulls weights toward zero, which reduces overfitting. Biases and normalisation layers are excluded from decay, which is standard practice. |
| **Two learning rates** | head 3e-4, backbone 1e-4 | The head is new and needs to move a lot. The backbone is already good and should only be *adjusted*. |
| **Warmup + cosine schedule** | `cosine_with_warmup()` | The LR rises from about 0 over the first steps, so early updates are small while optimiser statistics settle. It then smoothly decays to 1% of its peak, so later epochs make fine, careful adjustments. |
| **Label smoothing 0.1** | `CrossEntropyLoss(label_smoothing=0.1)` | The target becomes "90% correct class, 10% spread over the others". This stops the model from becoming absurdly over-confident, and some labels are wrong anyway. |
| **Dropout 0.3 / drop-path 0.2** | `timm.create_model(...)` | Randomly switches off parts of the network during training, so it can't rely on any single feature. This is another regulariser. |
| **Mixed precision (AMP)** | `torch.autocast` + `GradScaler` | On the GPU, does most of the maths in 16-bit floats. That's about 2× faster with about half the VRAM, at no accuracy cost. It is off automatically on the CPU. |
| **Gradient clipping (1.0)** | `clip_grad_norm_` | Caps very large gradient spikes so one bad batch can't wreck the weights. |
| **NaN guard** | `train_one_epoch()` | If the loss becomes NaN or infinite, training has diverged. We stop immediately with a clear message instead of wasting an hour producing garbage. |
| **Early stopping** (patience 6) | `EarlyStopping` | After each epoch we measure **val macro-F1**. If it hasn't improved for 6 epochs, we stop: more training would only overfit. |
| **Best checkpoint** | `save_checkpoint()` | We keep the weights from the epoch with the best val macro-F1, not the last epoch. |
| **Resumable checkpoint** | `last.pt` every epoch | If the friend's session crashes or the laptop sleeps, training continues from the last epoch instead of restarting. Files are written to a temp file and then renamed, so a crash mid-save can't corrupt them. |

### Why monitor **macro-F1** and not accuracy?
Accuracy is dominated by the big classes (`normal` and `blast` each have about 1,750 images). **Macro-F1** computes the F1 score *for each class separately* and averages them, so a weak rare class (such as `bacterial_panicle_blight`, about 330 images) pulls the score down visibly. See §7.

### Reading the training curves
The notebook plots train loss and val loss, and train accuracy and val macro-F1, per epoch:

```
 loss                                 loss                                 loss
  │╲                                   │╲                                   │╲___________
  │ ╲___ train                         │ ╲____ train                        │            train
  │  ╲__ val                           │  ╲   ___/ val ← rising!            │ ‾‾‾‾‾‾‾‾‾‾ val (high)
  └──────── epoch                      └──────── epoch                      └──────── epoch
   GOOD: both fall, small gap          OVERFITTING: val rises while         UNDERFITTING: both stay high
                                       train keeps falling                  → train longer / higher LR /
                                       → more augmentation, early stop,       bigger model or image size
                                         more data
```

The old model had a **train–val gap of about 10 points**, a clear sign of overfitting. Ours should be much smaller.

- **Train accuracy will look lower than you expect.** It is measured *on augmented, Mixup-blended* images, which are deliberately harder than real ones. That's normal; judge by val.
- **Loss suddenly jumps to NaN:** the LR is too high. Lower `LR` and resume.

## 7. Evaluation
This is notebook §7–8, run **once**, on the test split.

### The metrics, in plain words
| Metric | Question it answers | Our target |
|---|---|---|
| **Top-1 accuracy** | How often is the single best guess right? | ≥ 93% (the old model scored 79.5% on val) |
| **Top-3 accuracy** | How often is the right answer *among the top 3*? This matters because the Q&A engine can ask the farmer questions to choose between the top few candidates. | ≥ 98% |
| **Precision** (per class) | When the model says "blast", how often is it really blast? | — |
| **Recall** (per class) | Of all the real blast photos, how many did the model catch? A missed disease is the costly mistake for a farmer. | no class < 85% |
| **F1** (per class) | A single number that combines precision and recall (their harmonic mean) | — |
| **Macro-F1** | The average of the per-class F1 scores, with every class counting equally whatever its size | Used to choose the best epoch |
| **ECE** | Calibration quality (§8) | < 0.03 |

**Why macro and not overall?** Imagine a model that is 99% right on `normal` (about 1,750 photos) but only 50% right on `bacterial_panicle_blight` (about 330). Overall accuracy still looks great because the big class dominates. Macro-F1 exposes the weak class.

### The confusion matrix
This is a 10×10 table. **Rows are the true class and columns are the predicted class.** The diagonal holds the correct predictions; everything off the diagonal is a mistake, and its position tells you *which* mistake. The notebook shows both counts and row percentages, and each row percentage on the diagonal is that class's recall.

Expected look-alike pairs include:
- `bacterial_leaf_blight` ↔ `bacterial_leaf_streak` (both bacterial, with streaky lesions)
- `blast` ↔ `brown_spot` (both cause spots)

The most-confused pairs are printed in §8. **These are exactly the cases the follow-up Q&A engine needs good questions for**, such as "Do the lesions have a grey centre?" or "Is there bacterial ooze in the morning?"

### Error analysis
§8 shows the **16 most confident mistakes**. For each one, ask:
- *Is the label wrong?* Field datasets always have some mislabels. Note them, but don't edit the test set to make the score look better.
- *Is it a genuine look-alike?* That's a job for the Q&A engine.
- *Is the photo bad* (blurry, mostly soil, several diseases at once)? That's a reason for image-quality checks in the app.

### TTA (test-time augmentation)
At test time we predict on the image **and its mirror image** and average the two logits. This gives slightly more stable predictions for free. Flip-TTA is **baked into the exported ONNX** (§9), so the server automatically gets exactly the same numbers.

### Rules for the test set
- Look at the test score **once**. If you then change something and re-run, report it honestly as "tuned after seeing test".
- The old model's 79.5% was measured on a val set that was *also* used for early stopping, so it is slightly optimistic. The notebook prints this caveat next to the comparison.

## 8. Calibration
This is notebook §6.

### Why it matters for Krishimitra
The **Bayesian Q&A engine** starts from the classifier's probabilities, which act as the *prior*, and updates them with the farmer's answers. If the classifier says "95% blast" but is really right only 75% of the time at that confidence, the engine will:
- stop asking questions too early, and
- give wrong advice with false certainty.

Modern CNNs, especially after long fine-tuning, are typically **over-confident**.

### Temperature scaling
Divide all the logits by a single number **T** before the softmax: `p = softmax(logits / T)`.
- **T > 1** softens over-confident probabilities. T < 1 sharpens under-confident ones.
- **The prediction (argmax) never changes**, so accuracy is untouched. Only the confidence changes.
- T is fitted on the **val** set by minimising NLL (`kd.fit_temperature`, which uses the LBFGS optimiser on `log T` so that T stays positive). It must *not* be fitted on test, for the same reason as always.
- A typical result is T ≈ 1–2. The notebook warns you if T is outside 0.5–5, which suggests something is wrong. In the smoke test a huge T is expected, because an untrained model is meaningless.

### Reading the reliability diagram and ECE
- All predictions are grouped into **bins by confidence** (0–6.7%, …, 93–100%).
- For each bin we plot the **actual accuracy** (bar height) against the confidence. **Perfect calibration means the bars sit on the diagonal.**
- If the bars are *below* the diagonal, the model is over-confident; above it, under-confident.
- **ECE (Expected Calibration Error)** is the average gap between bar and diagonal, weighted by how many predictions fall in each bin. For example, ECE = 0.02 means confidence is off by about 2 percentage points on average. Our target is < 0.03 after scaling.

## 9. Export and use
This is notebook §9.

### What ONNX is
**ONNX** (Open Neural Network Exchange) is a file format that stores the network's structure *and* weights in a framework-neutral way.
- The FastAPI server loads it with **onnxruntime**, a small, fast CPU inference library. **PyTorch, timm and CUDA are not needed on the server.**
- Our model is exported at opset 17 with a *dynamic batch* size, so the server can send 1 image or 32.
- EfficientNetV2-S takes up about 80 MB.

The notebook **verifies the export**: it runs 8 real test images through both PyTorch and onnxruntime and requires the outputs to match within 1e-4. (The smoke test gave about 2e-5.)

### The deliverable: `models/disease_classifier_v1/`
| File | Contents |
|---|---|
| `model.onnx` | The network with flip-TTA baked in. Input `image` [N,3,H,W] float32; output `logits` [N,10]. |
| `class_names.json` | The class order. Output index *i* is `class_names[i]`. |
| `preprocess.json` | The exact preprocessing: resize the shorter side to `IMG_SIZE`, centre-crop, divide by 255, subtract `mean` and divide by `std`, using RGB and NCHW. |
| `temperature.json` | T, and the val ECE before and after scaling |
| `metrics.json` | The test metrics, per-class recall, versions, GPU and the old baseline |
| `classification_report.txt`, `confusion_matrix.png`, `reliability_diagram.png`, `training_curves.png` | For humans to review |

### How the server will use it (sketch for `/health/diagnose`)
```python
import json, numpy as np, onnxruntime as ort
from PIL import Image

D = "models/disease_classifier_v1/"
sess  = ort.InferenceSession(D + "model.onnx", providers=["CPUExecutionProvider"])
pre   = json.load(open(D + "preprocess.json"))
names = json.load(open(D + "class_names.json"))
T     = json.load(open(D + "temperature.json"))["temperature"]

def diagnose(img: Image.Image) -> dict[str, float]:
    img = img.convert("RGB")
    s = pre["resize_shorter_side_to"]; w, h = img.size; r = s / min(w, h)
    img = img.resize((round(w * r), round(h * r)), Image.BILINEAR)
    w, h = img.size; c = pre["center_crop"]; l, t = (w - c) // 2, (h - c) // 2
    x = np.asarray(img.crop((l, t, l + c, t + c)), dtype=np.float32) / 255.0
    x = (x - pre["mean"]) / pre["std"]
    x = x.transpose(2, 0, 1)[None].astype(np.float32)        # HWC -> NCHW
    logits = sess.run(None, {"image": x})[0][0] / T
    p = np.exp(logits - logits.max()); p /= p.sum()
    return dict(sorted(zip(names, p.tolist()), key=lambda kv: -kv[1]))   # prior for the Q&A engine
```
PIL's resize and torchvision's resize differ very slightly, so probabilities can shift a little in the last decimals. That's harmless, but the final sanity check (Execution order, step 5) compares the server-style predictions with the notebook on 5 test images to confirm it.

## 10. Next improvements
Try these roughly in this order, and **only after** the first full run, so we know what is actually holding the model back. Change **one thing at a time** and compare on **val**. The test set is for the final number only.

### If the targets were missed
| Symptom (from curves / report) | Try |
|---|---|
| Val still improving when training ended (underfitting) | More epochs (`EPOCHS_FT=40`, `PATIENCE=8`) |
| One or two classes below 85% recall | Look at their errors in §8. If they are mislabels, clean the train set; if they are look-alikes, increase `IMG_SIZE` (lesion detail) or add more photos of that class |
| Big train–val gap (overfitting) | Stronger augmentation, `MIXUP_PROB=0.5`, higher `drop_path_rate` |
| Everything plateaus at about 90% | A bigger or different backbone (below) |

### Bigger or different models (same notebook, just change `MODEL_NAME`)
- `convnext_tiny.fb_in22k_ft_in1k`: a strong modern CNN at about 28 M parameters. Try `IMG_SIZE` 384.
- `tf_efficientnetv2_m.in21k_ft_in1k`: larger. It needs about 12 GB VRAM at 384, and it is slower on the CPU server.
- **Larger images** (448–512) often help the most for small lesions such as early blast or hispa scars.
- **Remember the server:** inference runs on a **CPU**, so we must check the latency of any bigger model (aim for under 1 s per photo) before adopting it.

### Kerala field test set (the most important next step)
Paddy Doctor photos come from **Tamil Nadu**, with particular varieties, lighting and phones. Our test score measures *in-distribution* accuracy only. Before trusting the app in Kerala:
1. Collect about 30–50 photos per class from Kerala fields, through KVK/KAU or our own field visits, labelled by an agronomist.
2. Evaluate the exported model on them **without retraining**. The drop from 93% tells us the size of the "domain gap".
3. Only then mix some Kerala photos into training, keeping a separate Kerala test set.

### New classes for Kerala
Paddy Doctor lacks several diseases and pests that are common in Kerala, for example **sheath blight**, **false smut**, **sheath rot**, **leaf folder**, **BPH (brown planthopper) damage** and **gall midge**.

Adding a class means:
- collecting photos from licensed sources (§2),
- adding the class to `CLASS_NAMES` (append it at the end, so existing indices stay the same),
- re-running `make_split.py` to produce a **new** `splits.csv` for a new model version (`disease_classifier_v2`),
- retraining.

Never silently change v1's split or class list.

### Other ideas (later)
- **An "unknown / not a rice leaf" option.** The classifier always picks one of the 10 classes, even for a photo of a cow. Options include a confidence threshold, a separate out-of-distribution check, or an extra "other" class trained on non-paddy photos.
- **Several diseases in one photo.** This is deferred (see the MVP scope); for now the model is single-label.
- **Smaller or faster export:** INT8 quantisation of the ONNX model (`onnxruntime.quantization`) gives about 4× smaller files and faster CPU inference. Check that accuracy and calibration don't drop.

---

## Your next steps (hands-on)
1. **Set up Kaggle** (§2: account, accept the rules, `kaggle.json`) and download and unzip the data into `training/data/paddy/`.
2. **Make the split:** `python make_split.py`. It should end with *All checks passed*. Look at `duplicate_report.csv`.
3. **Smoke test on your laptop:**
   - Open the notebook, set `SMOKE_TEST = True` and Run All. It should take about 5–10 minutes on the CPU.
   - Every cell should finish, and §9 should print an ONNX diff below 1e-4.
   - Then set `SMOKE_TEST = False` again.
4. **Commit** `training/` (including `splits.csv` and `duplicate_report.csv`). Then send the `training/` folder to your friend, together with `RUN_ON_GPU.md`.
5. When you get the zip and the executed notebook back, unzip into `models/disease_classifier_v1/` and review the results using §7–8.

---

## 11. Common mistakes to avoid
These are taken from reviewing a public Keras notebook on the same dataset. We take ideas from it only. It has no licence, so we copy no code.

| Mistake | Why it's bad | What we do instead |
|---|---|---|
| Validation images go through the same *augmenting* generator as training | Val scores become noisy and don't reflect real photos | Eval transforms are strictly resize + center-crop |
| No separate test set: the best checkpoint is picked on val *and* the score is reported on val | The reported accuracy is optimistic (we "tuned to" val) | Separate **test** split, touched exactly once at the end |
| Near-duplicate photos land in different splits | The model "recognises" a photo it has nearly seen, which inflates accuracy | Perceptual-hash dedup; each duplicate group stays in one split |
| Only overall accuracy is reported | A rare class can be 50% wrong while overall accuracy still looks great | Per-class recall, macro-F1, confusion matrix |
| No calibration | "95% confident" may really be right only 80% of the time | Temperature scaling + ECE (§8) |
| A frozen backbone | Under-fits (the old model reached 79.5%) | Fine-tune the whole network |
| Using disease facts from random READMEs | That README named the wrong pathogens for blast and downy mildew | Treatment / POP info comes only from KAU/ICAR sources |

---

## 12. Implementation log
A record of what was built, in order, and why.

### Step 1: Scaffolding (2026-11-16)
- Created the `training/` folder.
- `requirements.txt`: all dependencies **except PyTorch**. PyTorch is installed separately because the CPU laptop and the friend's GPU machine need different builds. torchvision ≥ 0.19 is needed for the JPEG-compression augmentation.
- `.gitignore`: excludes `data/` (raw images, about 1 GB), `outputs/checkpoints/` (intermediate weights), notebook checkpoints and `kaggle.json`.
- `GUIDE.md`: wrote §1 (what a CNN is, transfer learning, choice of backbone), §2 (dataset and Kaggle download) and §11 (common mistakes).

### Step 2: `make_split.py` (2026-11-16)
- Wrote `make_split.py`, which runs pHash dedup, then union-find grouping, then a stratified group split of 70/15/15, then sanity checks. It writes `splits.csv` and `duplicate_report.csv`.
- **Design choices:**
  - The Hamming distance is computed as a chunked matrix product (fast, about 40 MB of memory per chunk) instead of a Python double loop.
  - Hashes are cached in `data/paddy/phash_cache.csv`, which is git-ignored.
  - Groups with conflicting labels are dropped instead of guessing a label.
  - Hashing runs in a `ProcessPoolExecutor` guarded by `if __name__ == "__main__"`, which is required on Windows.
  - Exit codes: `0` = OK, `1` = data missing, `2` = a sanity check failed.
- **Tested** on a synthetic 872-image dataset made of 10 imbalanced classes with planted problems: 1 corrupt file, 20 exact duplicates, 1 near-duplicate and 1 cross-label duplicate.
  - All planted problems were caught: 1 unreadable image, 2 conflict images and 19 exact duplicates removed. (The 20th planted copy belonged to the cross-label pair, so it was dropped as a conflict instead.)
  - Every class proportion was within ±1%, and no group spanned splits.
  - A re-run from the hash cache produced a byte-identical `splits.csv`.
- **Not yet run on the real data** at this point. See "Real data: split created" below.
- `GUIDE.md`: wrote §3 (EDA and pHash) and §4 (why train/val/test, leakage, how the script works).

### Step 3: `krishi_data.py` (2026-11-16)
- Wrote `krishi_data.py`, an importable module that the notebook will use. It contains:
  - `CLASS_NAMES` (same order as the old model)
  - `PaddyDataset`
  - `build_transforms`, using torchvision v2 (train = augmenting, eval = resize + centre-crop only)
  - `make_loaders` (Windows-safe worker count, `drop_last` for train)
  - `set_backbone_trainable` (Stage A/B)
  - `build_optimizer` (AdamW, two LRs, no decay on bias/norm)
  - `cosine_with_warmup` (stepped once per *optimizer* step)
  - `train_one_epoch` (AMP, gradient accumulation, Mixup support, grad clipping, NaN guard)
  - `predict` (with optional hflip TTA)
  - `compute_metrics` (NLL, top-1, top-3, macro-F1, optional temperature)
  - `EarlyStopping`
  - `save_checkpoint` / `load_checkpoint` (atomic temp-file and rename)
- **Environment change on the user's laptop:** installed the CPU builds of `torchvision` (0.29.2) and `timm` (1.0.31) to test the module. pip also upgraded `torch` from 2.9.0 to **2.15.0+cpu** as a dependency.
- **Tested** on the synthetic dataset with `efficientnet_b0` (random init), 64 px and `num_workers=2`:
  - transform shapes and dtypes are correct
  - DataLoader workers start correctly on Windows
  - Stage A updates only the head and the backbone stays unchanged
  - Stage B with Mixup + `SoftTargetCrossEntropy` updates the backbone
  - LR schedule: 0.1 → 1.0 after warmup → 0.0103 at the end
  - `predict` with TTA and the metrics both run
  - `EarlyStopping` triggers after its patience runs out
  - a checkpoint save and load round-trips the weights exactly
  - the NaN guard raises `FloatingPointError`
  - Accuracy is near chance, as expected with random weights and 6 batches. This test checks that the plumbing works, not how well the model learns.
- `GUIDE.md`: wrote §5 (preprocessing, augmentation, Mixup/CutMix) and §6 (vocabulary, the two-stage recipe, each training ingredient and why, macro-F1, reading the curves).

### Step 4: The notebook (2026-11-16)
- Wrote `train_disease_classifier.ipynb` with 26 cells, matching plan sections 0–9:
  - **§0:** the config cell (VRAM table, `SMOKE_TEST`)
  - **§1:** environment check (loud warning when there's no CUDA)
  - **§2:** data (Kaggle auto-download, `splits.csv` integrity check, class counts, variety cross-tab, sample grid)
  - **§3:** transforms and the augmentation preview
  - **§4:** the model
  - **§5:** two-stage training with best/last checkpoints, resume and early stopping, plus the curves
  - **§6:** temperature scaling and the reliability diagram
  - **§7:** a one-time test evaluation (TTA, report, confusion matrix, comparison with the old model)
  - **§8:** error analysis
  - **§9:** ONNX export with verification, the JSON sidecars and a zip
- **Additions to `krishi_data.py`:**
  - `model_norm_stats`
  - `mean`/`std` parameters on `build_transforms` and `make_loaders`
  - `fit_temperature`, `reliability_bins`, `expected_calibration_error`
  - `TTAWrapper`
- **Bug found and fixed before it could bite:** `tf_efficientnetv2_s.in21k_ft_in1k` was pretrained with mean = std = 0.5, *not* the ImageNet statistics. The normalisation is now read from timm's pretrained config, and GUIDE §5 has been corrected.
- **Design choices:**
  - **Flip-TTA is baked into the ONNX file** through `TTAWrapper`, so the server needs no TTA code. Val (for calibration) and test both use TTA, so T matches what is deployed.
  - **The ONNX export uses the classic exporter** (`dynamo=False`, opset 17). The newer dynamo exporter crashed on the Windows console encoding under torch 2.15, so it is kept only as a fallback.
  - **Smoke mode writes to `outputs/smoke/`** (git-ignored) and never touches `models/`.
  - Environment variables `KRISHI_SMOKE`, `KRISHI_DATA_DIR` and `KRISHI_SPLITS` allow headless testing without editing the config.
- **Environment change:** installed `onnx`, `onnxruntime`, `onnxscript`, `nbclient` and `ipykernel` on the laptop. `onnxscript` was added to `requirements.txt`.
- **Tested:** ran the actual notebook headlessly (nbclient) in smoke mode on the synthetic dataset.
  - **All 26 cells ran without errors in 78 s.**
  - The ONNX vs PyTorch max difference was 1.8e-5 (< 1e-4).
  - All 9 deliverable files and the zip were produced.
  - **A second run resumed correctly:** "Training already finished - skipping to evaluation".
  - The metrics were near chance and T was huge, as expected for random-noise images. This test only checks the pipeline.
  - The test outputs were deleted afterwards, so the real smoke test starts clean.
- **Not tested:** resuming *partway through* stage B (only resuming after a finished run was tested), and the real GPU/AMP path. Both get exercised on the friend's machine.
- `GUIDE.md`: wrote §7 (metrics, confusion matrix, error analysis, TTA, test-set rules), §8 (calibration) and §9 (ONNX, the deliverable files, a server usage sketch).

### Step 5: `RUN_ON_GPU.md` and wrap-up (2026-11-16)
- Wrote `RUN_ON_GPU.md` for the friend with the GPU:
  - checking the GPU with `nvidia-smi`
  - creating a virtual environment
  - installing the **CUDA** build of PyTorch, with a `torch.cuda.is_available()` check
  - getting the data (Kaggle download *or* copying the folder)
  - choosing the config from the VRAM table
  - an optional 2-minute rehearsal with `SMOKE_TEST`
  - what to expect during Run All, and exactly what to send back (and what *not* to send)
  - a troubleshooting table: out of memory, crash and resume, NaN, Kaggle 403, missing images, Windows worker hangs, accidentally running on the CPU
- `GUIDE.md`: wrote §10 (what to try if the targets are missed, alternative backbones, the Kerala field test set, adding new classes as v2, unknown-input handling, INT8 quantisation) and a "Your next steps" checklist.
- **All files from Execution order step 1 are now written.** The user-side steps (download, split, smoke test, hand-off) come next.

### Real data: split created (2026-11-16)
**Dataset setup**
- The user downloaded Paddy Doctor. `make_split.py` was first attempted **while the unzip was still running**: only 6,632 of 10,407 images were present, and `normal` and `tungro` were missing.
- We waited until all 13,878 files existed, then moved them up out of an extra `paddy-disease-classification/` folder level.
- Verified: 10 classes, and 10,407 images matching the 10,407 rows of `train.csv`. GUIDE §2 now warns about both gotchas.

**Bug found by the real data, and fixed in `make_split.py`**
- In `stratified_group_split`, `groups = sizes.index.to_numpy(); rng.shuffle(groups)` shuffled a **view** of the pandas index in place. That scrambled the group-to-size mapping, so splits were filled using the wrong group sizes.
- The result was a deviation of up to 2.58% (for example, `bacterial_leaf_streak` had 22% of its images in test). The sanity check caught it and the script exited with "SOME CHECKS FAILED".
- It now uses `rng.permutation(...)`, which returns a new array.
- The synthetic test missed this because nearly all of its groups had size 1.
- *Lesson: always keep self-checks in data scripts, and never trust a split that fails them.*

**Results** (`python make_split.py`, threshold 4, seed 42)
| | |
|---|---|
| Images in `train.csv` | 10,407 (10 classes, from 337 `bacterial_panicle_blight` to 1,764 `normal`) |
| Unreadable | 0 |
| Image sizes | 10,403 at 480×640 and 4 at 640×480 |
| Near-duplicate pairs (≤ 4 bits) | 2,961, of which 229 are exact |
| Dropped: conflicting labels | **152 images in 23 groups**. Most common: `blast` + `downy_mildew` (7 groups), `hispa` + `normal` (4). |
| Removed: exact duplicates | 194 |
| Kept near-duplicate groups | 902 groups covering 2,544 images. Mostly **burst shots of the same field scene**; the largest group has 18 photos. |
| **Final split** | **train 7,035 / val 1,512 / test 1,514** (10,061 images) |
| Class proportion check | Every class is within **0.11%** of its overall share in every split ✅ |
| Leakage check | No image_id is in two splits, and no duplicate group spans splits ✅ |

**What the duplicates tell us about the dataset.** A visual check of the conflict groups confirmed that they really are the same scene photographed several times, but with different disease labels (for example `bacterial_panicle_blight` vs `normal`, and `blast` vs `tungro`).
- About a quarter of the dataset consists of burst-shot near-duplicates. **Without the group-aware split, these would have leaked between train and test** and inflated the test score.
- The label conflicts are a reminder that Paddy Doctor's labels are probably *per field visit, not per photo*. Expect some label noise in the remaining images too. Label smoothing (§6) helps with this.

**Outputs:** `splits.csv` and `duplicate_report.csv`, both to be committed. `data/paddy/phash_cache.csv` is git-ignored.

### CPU smoke test on the real data (2026-11-16)
**The run**
- Ran the notebook headlessly with `KRISHI_SMOKE=1`: 200 real images (14/3/3 per class), `efficientnet_b0`, 128 px, 1 + 1 epochs.
- **All 26 cells passed in 112 s.**
- The ONNX vs PyTorch max difference was **3.2e-5** (< 1e-4 ✅). All 9 deliverable files and the zip were produced in `outputs/smoke/`.
- The executed notebook is saved as `outputs/smoke/executed_smoke.ipynb`. Open it to see every cell's output and plot.
- The scores were near chance (val top-1 13%) and T = 10. **That is expected**: 17 training steps on 140 tiny images can't learn anything.

**Extra sanity check, because a near-chance score could also hide a bug.** A *linear probe*: a logistic regression on frozen pretrained `efficientnet_b0` features, 224 px, 40 train images per class.
- It reached **52% val accuracy** (chance is 10%), which proves that the images, labels and normalisation line up.
- The high starting loss (about 5, against 2.3 for uniform guessing) comes from timm's head initialisation, which gives logits with a std of about 2.5. It is not a bug.
- It also confirms §1's lesson: frozen features alone fall far short of the target, so stage B fine-tuning is essential.

**Note:** `outputs/smoke/` is git-ignored and is separate from the real run's `outputs/checkpoints/`, so it cannot interfere with the friend's training.
