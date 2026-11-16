# Krishimitra – Crop Health Agent: Feasibility Review & Build Plan

## Context
Krishimitra is a two-agent advisory system for Kerala paddy farmers (crop health + market/scheme). The repo currently holds only the architecture diagram and a trained MobileNetV2 (`paddy_model/`). Goal of this step: check what is realistic in the crop-health design, what to change, and how to build it. Market agent is out of scope for now.

## 1. Diagnosis of the current model (from `paddy_model/`)
- 10 classes = the Kaggle **Paddy Doctor** dataset (Tamil Nadu field images, ~10.4k images). Val: **top-1 79.5%, top-3 95.2%**.
- **Class imbalance is not the main problem.** Ratio is only ~5:1 (71 vs 343) and the small classes (BLS, BPB) actually score well. The real problems:
  1. **Weak capacity / low resolution.** MobileNetV2 at (probably) 224px loses small lesions. Published Paddy Doctor baselines with ResNet/EfficientNet at higher resolution reach the high 90s.
  2. **Overfitting.** Train is 0.89 vs val 0.79, and val loss flattens at ~0.6 → needs stronger augmentation and regularisation.
  3. **No real test set.** Val was also used for early stopping (see `metrics.json` note), so the reported numbers are optimistic.
  4. Main confusions: hispa↔normal (36), hispa→brown_spot/downy_mildew, BLB→brown_spot, dead_heart→BPB. This fits the "looks the same early on" problem, and Q&A can fix it.
- **Top-3 = 95%** is the key number. The right disease is almost always in the top 3, so **CNN top-k + clarifying questions** is a sound design.
- The training script is not in the repo. Add it, so results can be reproduced.

**Decision: retrain (still transfer learning). Do not look for a ready-made "paddy model".** No reliable off-the-shelf paddy model exists. PlantVillage models use lab images and do not transfer. General VLMs (GPT/Claude/Gemini) are poor at telling fine-grained rice diseases apart, so do not use them as the classifier.
- Backbone: EfficientNetV2-S or ConvNeXt-Tiny (ImageNet-pretrained), input 384–512px. Optional: a frozen DINOv2 feature extractor + linear head as a quick baseline.
- A fixed 70/15/15 train/val/test split, stratified (by source/field if metadata allows, to avoid leakage).
- Augmentation: RandomResizedCrop, flips, colour jitter/brightness (sun vs shade), blur, JPEG compression (WhatsApp-like). Mixup/CutMix.
- Loss: label-smoothed CE with class weights (or focal). Imbalance is mild, so weights are enough.
- **Calibration (temperature scaling)** on val. This is essential because the Bayesian engine uses these probabilities as its prior.
- Test-time augmentation (TTA). Report per-class recall on the held-out test set.
- Expected: ~93–97% on the Paddy Doctor test set. **Real-world accuracy on Kerala phone photos will be lower (domain shift).** Plan to collect 50–100 Kerala field images per class (KAU/KVK/pilot farmers) as a separate "field test set".

## 2. Feasibility of each feature

| Feature | Feasible? | Notes |
|---|---|---|
| Photo → disease + confidence | Yes | After the retrain. Needs calibration and a "not rice / bad photo / unknown" rejection path. |
| Clarifying Q&A to resolve look-alikes | **Yes, the strongest part of the design** | Bayesian update over the CNN top-k is simple and explainable. The hard part is the *likelihood table*, not the code. |
| Disease details from Package of Practices (POP) in DB | Yes | Manual curation of the KAU POP (rice chapter) into structured rows. About 1–2 weeks of careful data entry. Must be expert-reviewed. |
| Personalised treatment plan | Yes, if deterministic | Compute doses from POP rate × farmer's field area, growth stage, organic/chemical preference and pre-harvest interval. **The LLM must not invent chemicals or doses.** |
| Follow-up photos → "improving or not" | **Partly. This is the most overclaimed feature.** | Lesions on treated leaves do **not** disappear, so the CNN will still say "blast" after a successful spray. Two photos of different leaves, angles and lighting cannot be compared directly. See §4. |
| KVK / expert fallback | Yes, as an async hand-off | Needs a real partner. Otherwise show the Kisan Call Centre number (1800-180-1551) and the local Krishi Bhavan. |

## 3. Design corrections to the diagram
1. **Add an input gate before the CNN.** Check image quality (blur/dark) and "is this a rice plant/leaf". A small binary classifier or an OOD score is enough. Ask for a retake on failure.
2. **Coverage gap: important Kerala problems are missing from the 10 classes.** Sheath blight, sheath rot, false smut, neck blast, brown planthopper (hopper burn), gall midge, leaf folder and stem borer adult signs are all missing. Handle this with:
   - an **"Other / unknown" path**: low max-probability or high energy/OOD score → symptom-only Q&A over the *full* POP disease list → KVK;
   - a **symptom-first entry** (the diagram already has "select symptoms") that works with no photo;
   - adding classes later from collected Kerala images.
3. **Add context priors to the Bayesian engine**, not just the photo: growth stage (from sowing date), season (Virippu/Mundakan/Puncha), recent weather (rain, humidity, night temperature, from a weather API using the farmer's location) and variety. Example: BLB after floods, blast with cool humid nights. These priors are cheap and work well.
4. **Add persistent stores the diagram is missing:** Farmer/Field profile (location, area, variety, sowing date), **Case store** (images, Q&A answers, posteriors, plan, follow-ups), Image store. Progress tracking cannot work without these.
5. **Treat the crop health "agent" as a deterministic state machine. The LLM works only at the edges.** The LLM (a) phrases questions and advice in Malayalam and (b) parses free-text or voice answers into structured answers. Diagnosis logic, the treatment lookup and dose maths stay in code and the DB. This keeps the system safe and auditable.
6. **Translation layer:** chemical names, doses and dates come from **pre-verified Malayalam templates**. The LLM must not translate them freely. Use an LLM only for the surrounding explanation.
7. Confidence thresholds should decide three outcomes: **confirm** (posterior ≥ ~0.85), **ask** (up to 3 questions), **escalate** (still unsure, rare disease, or OOD).

## 4. How to build each component

### 4.1 Photo Scanner
FastAPI endpoint `POST /health/diagnose`. It runs gate → classifier (ONNX/TF-Lite on the server) → calibrated softmax → top-k with probabilities.

### 4.2 Bayesian Q&A engine
- Data: `symptom_questions` (id, Malayalam/English text, answer options, optional image hint), `likelihoods` (disease × question × answer → P). Elicit P with qualitative levels from POP text and an expert (always 0.9 / often 0.6 / sometimes 0.3 / rarely 0.05).
- Algorithm: prior = CNN probs × context priors (normalised). Each turn, pick the question with the **maximum expected information gain** over the current candidates. Update the posterior. Stop when the posterior is ≥ threshold, after 3 questions, or when no question separates the remaining candidates.
- Example discriminators: lesion shape (spindle = blast, oval brown with grey centre = brown spot), wavy yellow margin from leaf tip (BLB), scraped white streaks + beetle/grubs (hispa), bacterial ooze test (cut leaf in water), central shoot dries and pulls out easily (dead heart), orange-yellow leaves + leafhoppers + stunting (tungro).
- Also ask for a **second photo** (stem/panicle/whole hill) when the top candidates differ by plant part.

### 4.3 Treatment DB (from KAU POP)
Tables: `diseases`, `symptoms`, `treatments` (type: cultural/biological/chemical; product; rate per litre and per ha; timing; repeat interval; pre-harvest interval; growth-stage limits; organic flag), `preventive_measures`. Every row has a POP page reference and a reviewed-by field.

### 4.4 Personalised plan generator (deterministic)
Inputs: diagnosis, severity (from Q&A: % of hills affected), field area, growth stage, preference, and days to harvest. Output: a dated schedule (Day 0 action, Day 7–10 re-check, repeat spray if needed) with total quantities computed for the farmer's area. Then the LLM renders it into simple Malayalam.

### 4.5 Follow-up / progress tracking (redesigned)
Do not rely on photo-to-photo comparison. Each follow-up (Day 7, Day 14) combines:
- **Structured check-in questions:** "Are new leaves showing spots?", "Has the affected patch grown?", "% hills affected now?", "Did you apply the treatment? When?"
- **A guided photo of the *newest* leaves / the same marked patch.** Use the CNN's diseased-vs-normal probability on the new growth as a signal.
- Rule-based status: **Improving / Stable / Worsening / Not treated** → next action (continue, repeat, switch to the alternative treatment, escalate to KVK).
- Later (v2): a lesion segmentation model for a % severity score. This needs pixel-level labels, so it is not for the MVP.

### 4.6 Expert escalation
Build a case packet (photos, answers, posterior, plan) → queue/dashboard or WhatsApp to a partner officer. Return the reply to the farmer.

## 5. Key challenges / risks
- **Domain shift** (Tamil Nadu dataset vs Kerala phones and conditions). This is the biggest real-world risk, and only local data fixes it.
- **Expert time** for likelihood tables and POP validation. It is the bottleneck, not the code.
- **Missing classes** (sheath blight etc.) are common in Kerala. The unknown path is mandatory.
- **Liability:** pesticide advice must come only from POP, with a disclaimer and KVK fallback.
- **Malayalam UX:** low literacy means voice input/output and picture-based answer options help a lot.
- Connectivity: compress images on the device. The server model is fine. Use an on-device TF-Lite model only if offline use is required.

## 6. Suggested build order
1. Retrain the classifier: new split, backbone, calibration, test report. Commit the training script.
2. Structure the POP rice chapter into the DB (start with the 9 diseases/pests in the model plus sheath blight).
3. Build the Bayesian Q&A engine with the likelihood table, plus a simulation harness (sample a disease → answer questions per likelihoods → measure accuracy and number of questions).
4. FastAPI endpoints: diagnose, answer, plan, follow-up, plus the case store.
5. LLM Malayalam rendering and answer parsing.
6. Follow-up flow, then the KVK escalation.

Architecture diagram: done → `crop_health_agent_diagram/crop-health-agent.html` (it shows the full target design; the MVP builds a subset of it).

## 7. MVP scope (decided: build this first)
Goal: one farmer, one field, one open case at a time, end to end: photo → diagnosis → plan → one follow-up.

**In the MVP**
- Retrained classifier (EfficientNetV2-S, calibrated, top-3) served from FastAPI (ONNX).
- Simple input gate: blur/brightness check (OpenCV Laplacian variance + mean brightness) and "low max-probability → retake or symptom path". A dedicated rice-or-not model comes later.
- Context prior: **growth stage + season only** (from sowing date). The Weather API comes later.
- Bayesian Q&A engine over the top-3 with ≤ 3 questions. Likelihood table for the 10 model classes only.
- POP DB with the 9 diseases/pests in the model plus `normal` advice. Tables: `diseases`, `symptom_questions`, `likelihoods`, `treatments`.
- Deterministic plan generator: rate × area, stage filter, pre-harvest interval, dated schedule.
- One follow-up (Day 7): check-in questions + photo → Improving / Stable / Worsening / Not treated → rule-based next action.
- LLM only for Malayalam rendering of fixed templates and parsing free-text answers. Templates first, LLM optional.
- Escalation = show the KVK / Kisan Call Centre contact and save the case with `ESCALATED`. No officer dashboard yet.
- Storage: one PostgreSQL (or SQLite for dev) with `farmers`, `fields`, `cases`, `case_events`, `plans`, `followups`. Photos go on local disk or MinIO.

**MVP rule for multiple problems:** a field can have only **one open case**. If a new photo arrives while a case is open, ask "Is this a follow-up of the current problem?" Yes → attach it as follow-up evidence. No → the farmer closes the current case (or the system marks it `SUPERSEDED`) and a new case starts. This keeps the data model ready for multiple cases later without building the router or coordinator now.

**Deferred (post-MVP)**
- Multiple diseases per field: Case Router (new / follow-up / revision), Field Treatment Coordinator (merged sprays, tank-mix compatibility, nutrient conflicts), combined field visits.
- Weather API priors, OOD energy score, rice-or-not gate model, second-photo requests by plant part.
- New classes for Kerala (sheath blight, false smut, BPH, ...) and the Kerala field test set.
- Ask-a-question fallback (LLM answers only from POP), a KVK officer dashboard, Day 14+ follow-ups, a severity segmentation model.
- LLM-led conversational agent (only if pilot farmers find the fixed flow too rigid).

**MVP build order**
1. Retrain + calibrate the classifier; commit the training script and test report.
2. POP DB seed for the 10 classes and the likelihood table (expert-reviewed spreadsheet → seed script).
3. Bayesian Q&A engine + simulation harness.
4. FastAPI: `/health/diagnose`, `/health/answer`, `/health/plan/{case_id}`, `/health/followup` + case state machine + storage.
5. Plan generator + Malayalam templates.
6. Day-7 follow-up flow; escalation contact screen.
7. Minimal client (Android or a simple web/PWA page) for the pilot.

## Verification
- Classifier: per-class recall and confusion matrix on the **held-out test set**, plus the Kerala field set. Reliability diagram / ECE before and after temperature scaling.
- Q&A engine: simulation shows top-1 after Q&A ≥ CNN top-1 + 10 points, with ≤ 3 questions on average. Expert review of 20–30 real cases.
- Treatment plans: an expert checks generated plans against POP for 10 scenarios (different areas, stages and preferences).
- End-to-end: `uvicorn` + a scripted session (upload → questions → plan → day-7 follow-up) produces the correct state transitions.
- MVP one-case rule: uploading a second photo while a case is open triggers the follow-up question; "No" supersedes the old case and opens a new one.
- After approval: also copy this updated plan to `D:/Krishimitra/crop_health_agent_plan.md` so the repo copy stays in sync.
