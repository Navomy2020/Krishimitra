# Plan: Train the paddy disease classifier (step 1 of the MVP)

## Context
Step 1 of the Krishimitra MVP (`crop_health_agent_plan.md` §7) is to retrain the photo disease classifier. The current `paddy_model/mobilenetv2_paddy_final.keras` reaches only 79.5% val top-1. It is overfitting, uses a small input size, has no held-out test set and no calibration, and its training script is missing.

The user has never trained a CNN and wants guidance from the very start. Their laptop has no GPU (PyTorch 2.9 is the CPU build, and there is no `nvidia-smi`). They will hand the notebook to a friend who has a GPU, and the friend's VRAM is unknown.

Decisions taken: **PyTorch + timm**, a **self-contained notebook that the friend runs locally**, safe defaults for 6–8 GB VRAM, and one config cell to raise them.

Outcome: a reproducible notebook plus a beginner guide. It should produce a calibrated EfficientNetV2-S (target ≥ 93% test top-1, ≥ 98% top-3) exported to ONNX, ready for `/health/diagnose`.

## Reference repo reviewed (prashanth-gajula/Paddy-Disease-Classification-Using-Deep-Learning)
The repo is one Colab notebook (Keras) plus a README, trained on the same Paddy Doctor dataset. It has **no LICENSE file**, so we take ideas only and copy no code.
- **Worth taking:**
  1. Their MobileNetV2, **fully fine-tuned** at 256px with Adam 3e-4 and label smoothing 0.1, reached ~93.7% val, compared with our old model's 79.5%. This suggests the old model was held back by its training setup (frozen or under-trained backbone), not by the data. It confirms our ≥93% target and the plan to unfreeze the whole backbone. Their note that "frozen backbone did not converge" is a useful lesson for `GUIDE.md`.
  2. Vertical flip and small rotation are fine for leaf photos. Both are already in or added to our train transforms.
  3. Stopping on a NaN loss is a useful safety check, so we add a NaN guard to the training loop.
- **Not worth taking (flaws to avoid; `GUIDE.md` will explain these as "common mistakes"):**
  - Validation used the same augmenting `ImageDataGenerator` (`validation_split=0.3`), so val images were augmented.
  - There is no separate test set: checkpoint selection and the reported score both use val, so the reported numbers are optimistic.
  - No near-duplicate check, no per-class metrics and no calibration.
  - The "ViT" is a tiny model trained from scratch at 72px, a copy of the Keras example. It's not useful.
  - Several disease facts in the README are wrong. For example, it gives the blast pathogen as "Auricularia oryzae" (it is *Magnaporthe/Pyricularia oryzae*) and the rice downy mildew pathogen as "Peronospora parasitica" (it is *Sclerophthora macrospora*). **Never use it as a source for the POP/treatment DB.**
- Net effect on this plan: no change of backbone or approach. We add the NaN guard and small rotation, and keep eval transforms strictly un-augmented.

## What gets created (all new, under `D:/Krishimitra/training/`)
| File | Purpose |
|---|---|
| `GUIDE.md` | The learning guide for the user, in plain language. Explains every step and why it is done (see below). |
| `requirements.txt` | timm, torch/torchvision (installed separately per CUDA), pandas, scikit-learn, matplotlib, pillow, imagehash, onnx, onnxruntime, kaggle, tqdm. |
| `make_split.py` | Run **once on the user's CPU**. Removes near-duplicate images (perceptual hash) and writes a stratified 70/15/15 `splits.csv`. Duplicates are kept in the same split to avoid leakage. The CSV is committed, so the friend uses exactly the same split. |
| `krishi_data.py` | Dataset class, transforms and the train/eval loops as an importable module. This is **required on Windows**: DataLoader workers cannot pickle classes defined inside a notebook. |
| `train_disease_classifier.ipynb` | The main notebook (sections below). |
| `RUN_ON_GPU.md` | Step-by-step instructions for the friend: install, Kaggle token, "Run All", what to send back. |
| `.gitignore` entries | `data/` and `outputs/checkpoints/`. Raw images and intermediate weights are never committed. |

Final artifacts land in `D:/Krishimitra/models/disease_classifier_v1/`:
- `model.onnx`
- `class_names.json`
- `preprocess.json` (size, mean, std)
- `temperature.json`
- `metrics.json`
- `classification_report.txt`
- `confusion_matrix.png`
- `reliability_diagram.png`
- `training_curves.png`

`paddy_model/` is kept as the old baseline for comparison.

## Learning path (`GUIDE.md` outline)
1. **What a CNN classifier is.** Image → numbers → probabilities over the 10 classes. What "transfer learning" means and why we start from an ImageNet-pretrained network instead of training from scratch.
2. **Finding a dataset.** Use the Kaggle *Paddy Doctor* competition (`paddy-disease-classification`): 10,407 labelled field photos in 10 classes, plus `train.csv` with `variety` and `age` columns.
   - The `test_images/` folder is **unlabelled**, so we make our own test split.
   - How to accept the competition rules, create `kaggle.json`, and run `kaggle competitions download -c paddy-disease-classification`.
   - Other sources for later Kerala classes: KAU/KVK and Mendeley rice datasets.
3. **Looking at the data (EDA).** Class counts, sample grids, image sizes, and spotting mislabels and duplicates.
4. **Splitting.** Why we need train/val/test, why test is touched only once, and what data leakage and near-duplicates are.
5. **Preprocessing and augmentation.** Resize and normalise to ImageNet mean/std. Augmentation simulates real phone photos: random crop, flips, colour/brightness jitter, blur and JPEG compression (WhatsApp-like images).
6. **Training.**
   - Epochs, batches, loss and learning rate.
   - Two stages: train only the head, then fine-tune the whole network.
   - AdamW with a cosine schedule, label smoothing, mixed precision.
   - Early stopping on val macro-F1.
   - How to read the training curves to spot overfitting and underfitting.
7. **Evaluation.** Accuracy vs top-3 vs per-class recall, the confusion matrix, and why macro-F1 matters with imbalance.
8. **Calibration.** Why "90% confident" should mean right 90% of the time (the Bayesian Q&A engine uses these probabilities as its prior). Temperature scaling, and reading ECE and the reliability diagram.
9. **Export and use.** What ONNX is and how FastAPI will load it.
10. **Next improvements.** More epochs or larger images, ConvNeXt, a Kerala field test set, and new classes.

## Notebook sections (`train_disease_classifier.ipynb`)
0. **Config cell.** The only cell the friend may edit.
   - `DATA_DIR`, `OUT_DIR`, `MODEL_NAME="tf_efficientnetv2_s.in21k_ft_in1k"`, `IMG_SIZE=384`, `BATCH=16`, `ACCUM_STEPS=2`, `EPOCHS_HEAD=2`, `EPOCHS_FT=25`, `LR`, `SEED`.
   - `SMOKE_TEST=False`.
   - A VRAM table in a comment: 6 GB → 320/12, 8 GB → 384/16, 12 GB → 384/24, 16 GB+ → 448/32.
1. **Environment check.** GPU name and VRAM, torch/timm versions, and a seed for reproducibility. It warns loudly if CUDA is not available.
2. **Data.** Download if missing (Kaggle API), read `splits.csv`, show class counts and a sample grid.
3. **Transforms.** torchvision v2 transforms:
   - Train: RandomResizedCrop(scale 0.6–1), HFlip/VFlip, ColorJitter, GaussianBlur(p=0.2), JPEG(quality 40–95), Normalize.
   - Eval: Resize + CenterCrop.
   - Show a grid of augmented samples.
4. **Model.** `timm.create_model(MODEL_NAME, pretrained=True, num_classes=10, drop_rate=0.3, drop_path_rate=0.2)`.
5. **Training.**
   - Stage A trains the head with the backbone frozen.
   - Stage B fine-tunes everything with AdamW (lr 3e-4 head / 1e-4 backbone), cosine schedule with warmup, CrossEntropy(label_smoothing=0.1), AMP and gradient accumulation.
   - Light Mixup/CutMix via `timm.data.Mixup` (prob 0.3).
   - Save the best checkpoint by val macro-F1, early stopping (patience 6), and a resumable checkpoint every epoch in case the friend's session stops.
   - Plot the curves.
6. **Calibration.** Fit temperature T on the val logits (LBFGS on NLL). Report ECE before and after and draw the reliability diagram.
7. **Test evaluation (run once).** Use TTA (original + hflip) and calibrated softmax. Report top-1, top-3, macro-F1, the per-class report and the confusion matrix, and compare with the old model's 79.5/95.2.
8. **Error analysis.** A grid of the worst misclassifications (true vs predicted, with confidence). Note which look-alike pairs the Q&A engine must resolve.
9. **Export.** ONNX (opset 17, dynamic batch). Check that the onnxruntime outputs match PyTorch (max abs diff < 1e-4). Write the JSON sidecars and zip `models/disease_classifier_v1/` for the friend to send back.

## Execution order
1. Write `GUIDE.md`, `requirements.txt`, `krishi_data.py`, `make_split.py`, the notebook and `RUN_ON_GPU.md`.
2. User: create a Kaggle account and token, download the dataset to `D:/Krishimitra/training/data/` (~1 GB), and run `python make_split.py` on the CPU (about 2–3 minutes).
3. **CPU smoke test on the user's laptop:** set `SMOKE_TEST=True`, which uses 200 images, 128 px, 1 epoch and `efficientnet_b0`, then Run All. This proves that every cell runs and the export works before anything is sent to the friend.
4. Send the friend the `training/` folder (code + `splits.csv` + `RUN_ON_GPU.md`; the data is downloaded on their side). The friend installs the CUDA build of PyTorch and runs all cells, which should take about 1–2 h on an 8 GB card. They send back the zipped `models/disease_classifier_v1/` and the executed notebook.
5. Review the results together using `GUIDE.md` §7–8 and commit the scripts, `splits.csv` and the model folder (ONNX is about 80 MB, under GitHub's 100 MB limit).

## Verification
- Smoke test passes end to end on the CPU: every cell runs, an ONNX file is produced, and the onnxruntime vs torch diff is < 1e-4.
- `make_split.py`: no image_id appears in two splits, no near-duplicate cluster spans two splits, and the class proportions in each split are within ±1% of the overall proportions.
- Full GPU run, on the held-out test set:
  - top-1 ≥ 93%, top-3 ≥ 98%
  - no class with recall < 85%
  - ECE after temperature scaling < 0.03
  - val–train gap clearly smaller than the old model's 10 points
- Sanity check: load `model.onnx` with onnxruntime on the user's CPU and predict on 5 test images. The predictions must match the notebook.
