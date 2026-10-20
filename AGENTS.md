# AGENTS.md

Instructions and current-state notes for AI agents working on this repository.

Repository root: `Nuclear-Test-Video-Generator/`
Project: parameter-controlled archival nuclear-test video generator (SD 1.5 + AnimateDiff two-phase LoRA).

---

## 0. Read this first

The codebase is a **scaffold with one real data run**, not a trained model. Before changing architecture or claiming results, verify the data stage.

### Verified state (2026-10-20)

| Check | Current result |
|---|---|
| `data/manifest.csv` | **100 rows** = 50 base clips × `orig`/`flip` (standardized 16-frame sequences @ 8 fps across 3 verified shots) |
| Distinct shots in manifest | **3** (`Dominic / Sunset`, `Crossroads / Able`, `Crossroads / Baker`) — Gate: **FAIL** (blocked from Phase 1/2 LoRA training until multi-shot ingest reaches ≥30 distinct shots) |
| Split distribution | **66 train, 34 val, 0 test** — Gate: **FAIL** (`validate_manifest.py` and `check_split_leakage.py` enforce the gate by default: requires test split and ≥30 shots) |
| Cached latents | **14 / 100** cached (`4, 16, 64, 64`); `04 --assert-100` is now the default and fails on partial coverage |
| Phase labels | **12 early-fireball, 18 expansion, 70 stabilized-cloud** (stabilized sampled across 8–118 s post-detonation) |
| Source reels | **4 downloaded of 27 registry rows** (`Dominic Sunset`, `Crossroads`, `Castle`, `Redwing`); compilation reels are rejected unless a verified `shot_segments.csv` row exists |
| Stage 2 table | **211 training-eligible detonations** (206 atmospheric + 5 underwater) across 24 operations; the 220-row alias registry also tracks 5 craters + 4 Roller Coaster safety tests, which are excluded from training |
| Official NNSA tally | **216 detonations** (206 atmospheric + 5 underwater + 5 crater); dataset corpus = **211** because crater events are not one of the four canonical burst types. Roller Coaster (4) are non-nuclear HE dispersal tests and are excluded entirely. |
| Terrain classes | **2 only**: `desert-flats`, `tropical-ocean`. `low-mountains` was removed because no atmospheric test in the registry is at a mountain site (all are NTS/NTTR desert or Pacific atolls). |
| Dataset batch test | Raw-frame and synthetic multi-split smoke tests pass; latent-only mode correctly fails while coverage is 14/100 (no shape mixing) |
| Web demo | 5 archival demo videos labeled as demo reference; canonical schema tokens (`black-and-white`, etc.); UI claims "100 Standardized Clips"; 211 training records in the archive database |

**Do not start or extend Phase 1/2 LoRA training until the Stage 2/3 gates in §4 pass.** With 3 shots (and 0 test shots), the model cannot learn the conditioning axes or generalize.


---

## 1. Non-negotiable rules

1. **Do not silently fall back.** No fallback to the full manifest when a split is empty, no raw-frame fallback that changes tensor shape, no silent simulation in inference, no silent default burst type.
2. **Never commit media or weights.** `raw_videos/`, `processed_clips/`, `latents/`, `samples/`, `models/`, `checkpoints/`, `web/videos/` stay gitignored. Use Kaggle/Hugging Face datasets.
3. **All paths written to CSVs must use `/`**, not Windows `\`. Kaggle runs Linux.
4. **Splits are shot-level and shot-disjoint.** All clips from one shot stay in one split. No random clip-level validation.
5. **One schema only.** Every prompt token produced by Stage 3 must be a valid inference token and classifier class. Validate or fail.
6. **Do not claim generated results from real archival samples.** `samples/` and `web/videos/` are demo footage until a real inference run produces labeled outputs.
7. **Provenance is part of the data.** Do not add a source without explicit license/rights evidence; do not default missing licenses to public domain.

---

## 2. P0 blockers

### P0.1 — Insufficient shots in manifest (3 of 30 required)
The manifest contains only 3 verified shots (`Dominic Sunset`, `Crossroads Able`, `Crossroads Baker`) across 2 operations. Multi-axis conditioning cannot be learned from 3 shots.

**Required:** ingest and segment enough reels to reach ≥30 distinct shots before Phase 1/2 LoRA training. See §3.


### P0.2 — Stage 2 parser corrupts operations and shot names
File: `scripts/02_parse_doe_metadata.py`

- Lines 236–247: any line containing `Operation ` replaces `current_op`. The DOE PDF contains annotation lines such as:
  - `Test is also considered under Operation Project 57`
  - `These tests were conducted as Operation Newsreel`
  
  Result: 19 Plumbbob shots labeled `Project 57`; 21 Hardtack I shots labeled `Newsreel`.

- Line 270: `raw_name.split()[0]` truncates multi-word names and appends notes:
  `Coulomb-A` → `Coulomb-ARadioactivity`, `Hood` → `HoodHighest-yield`, `Mike` → `MikeExperimental`, `De Baca` → `De`.

- Line 171: unknown burst types default to `surface-tower`, hiding parse failures.

**Required:** assign operations from an explicit official operation table/test-number ranges; preserve full shot names; strip note text; maintain `data/doe_shot_aliases.csv`; fail on `unknown`; reconcile 206 parsed rows against the official 216.

### P0.3 — Latent/raw fallback breaks batching
File: `training/dataset.py:145–155`

A batch with one cached latent and one raw clip raises:

```text
RuntimeError: stack expects each tensor to be equal size,
but got [4, 64, 64] at entry 0 and [3, 512, 512] at entry 1
```

**Required:** fail fast or filter to valid latents. `scripts/04_preencode_latents.py:194` defaults to 10 clips; encode all before training and assert coverage.

### P0.4 — Empty-split leakage
Files:
- `training/dataset.py:54–59` returns the entire manifest when `split` has no rows.
- `scripts/05_train_classifier.py:249–254` uses test as validation, then random 80/20 of the same data.

**Required:** raise on empty split; never fall back to `df`; ensure val/test contain real shots.

### P0.5 — Schema drift
Three incompatible schemas exist among `plan.md`, `README.md`, Stage 3 output, inference, and the classifier.

| Axis | `plan.md` | README/web/inference | Stage 3 output |
|---|---|---|---|
| Burst | 3 classes | 4 incl. `surface-barge` | 3 (barge collapsed) |
| Setting | `desert-day` / `desert-night` / `tropical-ocean` / `tropical-dawn` | `desert-flats` / `tropical-ocean` / `low-mountains` | 4 DOE settings |
| Phase | `flash-fireball` / `stem-rise` / `stabilized-cloud` | `early-fireball` / `expansion` / `stabilized-cloud` | `flash-fireball` / `stem-rise` / `stabilized-cloud` |
| Film | `black-and-white` / `colour-restored` | `high-contrast-monochrome` / `faded-technicolor` / `colour-restored` | `black-and-white` / `colour-restored` |

Training therefore uses tokens inference cannot request. Also, `02:154–171` collapses `Barge` into `surface-tower`, leaving classifier class `surface-barge` with zero data.

**Required:** create `schema.py` (or `schema.yaml`), validate all prompts, and update Stage 2/3, config, classifier, inference, and web together.

Recommended canonical schema (confirm before a breaking migration):
- `burst_type`: `air-burst`, `surface-tower`, `surface-barge`, `underwater`
- `yield_tier`: `sub-kiloton`, `tactical-yield`, `intermediate-yield`, `megaton-class`
- `terrain`: `desert-flats`, `tropical-ocean` (low-mountains removed: no supporting footage)
- `lighting`: `day`, `night`, `dawn`
- `phase`: `early-fireball`, `expansion`, `stabilized-cloud`
- `film_look`: `black-and-white`, `faded-technicolor`, `colour-restored`
- `smoke_trails`: bool

Splitting terrain and lighting avoids mixing two independent visual factors into one token.

### P0.6 — Phase labels collapsed by motion heuristic
File: `scripts/03_cut_and_filter.py:231–241`

The motion/luminance heuristic yields 198/200 `stabilized-cloud`.

**Required:** define phase by time since detonation onset: detect the flash, then `early-fireball` ≈ 0–0.5 s, `expansion` ≈ 0.5–2 s, `stabilized-cloud` later. Store `detonation_start_sec` and `offset_sec` in the manifest. Cap clips per `(shot, phase)`.

### P0.7 — Phase 2 LoRA trains spatial attention too
File: `training/02_train_motion_lora.py:55–74`, `123–144`

- PEFT `target_modules=["attn1.to_q", ...]` suffix-matches **spatial** `attn1` modules as well as `motion_modules.*`.
- Spatial freeze happens before PEFT injection, so the injected spatial LoRA weights remain trainable.
- Phase 1 is loaded with `load_attn_procs`, which likely cannot read a PEFT `save_pretrained` directory; the exception is swallowed.

**Required:** target only `motion_modules.*`; load Phase 1 with the matching loader (`load_lora_adapter` / diffusers format); add a test asserting no spatial parameter has `requires_grad=True`.

### P0.8 — Kaggle notebook cannot access the dataset
File: `scripts/build_kaggle_notebook.py:60–78`

The notebook clones the repo only. `processed_clips/`, `latents/`, and `models/` are gitignored; there is no dataset mount or path configuration. Manifest paths also contain Windows backslashes.

**Required:** mount a versioned Kaggle/HF dataset, parameterize paths, write CSV paths with `.as_posix()`, and assert manifest/latent coverage before training.

### P0.9 — Stage 6 gate is not implemented
File: `scripts/05_train_classifier.py`

- `--eval` is parsed but ignored (`05:387–393`).
- `evaluate_synthetic_clips` only dumps predictions (`05:367–370`); no expected labels, no balanced accuracy, no gate.
- `samples/` contains real archival demo clips, not labeled synthetic outputs.

**Required:** implement test-set evaluation, per-class recall/confusion, class-balanced sampling or weights, and a synthetic audit reading a JSON sidecar with expected attributes. Compute the advertised `balanced_accuracy ≥ majority_baseline + 25%` gate.

### P0.10 — Inference silently simulates
File: `inference/generate.py:67`, `117–177`

Without CUDA or a loadable pipeline, the script draws an OpenCV animation and prints `[SUCCESS]`.

**Required:** remove the silent fallback; add `--mode simulation` and label outputs. When loading two adapters, call `pipe.set_adapters([...])` explicitly.

---

## 3. Data acquisition plan

### 3.1 Fix labels before downloading more
`scripts/03_cut_and_filter.py:117–133` matches by filename and returns `shot_df.iloc[0]` on failure. This would label every Teapot reel as the first Teapot shot.

Introduce:

```text
data/shot_segments.csv
archive_id, operation, shot_name, start_sec, end_sec,
detonation_start_sec, confidence, label_source, reviewer
```

Clips must inherit `(operation, shot_name, date)` from this table, never from the filename.

### 3.2 Discover sources programmatically
Working Internet Archive advanced-search queries:

| Query | Hits |
|---|---|
| `title:("nuclear test film")` | 60 |
| `creator:("Department of Energy") AND mediatype:movies` | 75 |
| `title:("Operation Hardtack") OR title:("Operation Dominic") OR title:("Operation Plumbbob")` | 82 |
| `identifier:gov.doe.08000*` | 20 |

Add missing DOE items to `CORE_ARCHIVE_ITEMS`:
`gov.doe.0800000`, `0800004`, `0800005`, `0800006`, `0800007`, `0800014`, `0800016`.

Search per-shot names directly (`title:("Grable")`, `title:("Swordfish")`, `title:("Castle Bravo")`). Prefer per-shot items over compilation reels. `CompositeNo.1-SwordfishSailorHatConventionalTestASROCSUBROC` covers the rare `Swordfish` underwater shot.

Other sources:
- **NARA Catalog API** (`catalog.archives.gov`, RG 374/434) for operation/shot-level reels and digital objects.
- **Wikimedia Commons API** for individual PD-USGov `.ogv`/`.webm` files.
- **LLNL “Declassified” films**: use YouTube Data API metadata only; respect ToS and prefer official/IA/NARA downloads.

Do not bulk-scrape YouTube.

### 3.3 Auto-label shots in compilations
1. PySceneDetect for cuts.
2. OCR title cards/slates (`pytesseract` or PaddleOCR).
3. Fuzzy-match OCR text to DOE shot/operation names (`rapidfuzz`).
4. Optional Whisper transcription for narrated shot names.
5. Detect the detonation flash to set `detonation_start_sec`.
6. Produce `shot_segments.csv` plus a review page; manually confirm boundaries.
7. Use only high-confidence segments for val/test.

Pilot on 3 reels before scaling. Ten reels × ten shots × ~20 clips ≈ 2,000 clips.

### 3.4 Balance and gate the data
Current extraction uses 50%-overlapping windows (`03:307`), so long stabilized clouds dominate.

Required:
- cap clips per `(shot, phase)` (e.g. 8–12);
- shot-level oversampling for rare classes, not clip duplication;
- shot-disjoint splits stratified by `burst_type × yield_tier` (plus terrain/lighting);
- at least one val and one test shot per class where possible;
- for underwater, accept only 5 shots exist and report uncertainty.

Targets:
- ≥30 distinct shots;
- ≥8–10 shots per burst class except underwater (all 5);
- ≥8 per yield tier;
- ≥8 per terrain/lighting bucket;
- ≥2,000 clips;
- 100% latent coverage.

Use stills from every shot for Phase 1 spatial LoRA; spatial adaptation does not need clips.

### 3.5 Licensing and provenance
`data/sources.csv` currently contains five **CC BY-NC-ND** items, including the main Hardtack underwater reel and Dominic compilation. The README claims public domain; reconcile before publishing.

- Remove `gov.dod.dimoc.23521` — it resolves to “FEW GOOD MEN, A”, not Crossroads.
- Do not default missing licenses to public domain.
- Add `rights_status`, `rights_basis`, `evidence_url`, `allow_training`, `reviewer`.
- Prefer NARA/DOE/LLNL public-domain copies over restrictive IA uploads.

### 3.6 Download robustness
File: `scripts/01_download_archive.py:225–254`

- Handle HTTP 200 when a Range was requested: do not append a full response to a partial file.
- Verify content length/size and retry transient failures.
- Preserve manual provenance/rights columns; do not overwrite `sources.csv` wholesale (`01:200–201`).
- Verify resolution/fps/duration with OpenCV/ffprobe after download.

---

## 4. Definition of done for the data gate

- ≥30 distinct shots; ≥8 per burst class except underwater.
- 100% of manifest rows have a valid latent.
- Val/test contain real, shot-disjoint clips.
- Every prompt token validates against `schema.py`.
- A smoke test loads one batch from `train`, `val`, and `test` in both `spatial` and `motion` modes.
- `data/sources.csv` has explicit rights status for every row.
- Stage 2 counts reconcile with the official 216 atmospheric/underwater/space tests, with exclusions documented.

---

## 5. Suggested work order

1. **Repair Stage 2** — operation headers, multi-word names, aliases, 216 reconciliation; regenerate audit/splits/`historical_shots.json`.
2. **Unify schema** — add `schema.py`; update Stage 3, config, classifier, inference, web; validate prompts.
3. **Make the dataset fail-fast** — no empty-split fallback, no mixed shapes, POSIX paths, latent coverage assertion.
4. **Build `shot_segments.csv`** — OCR + flash detection + manual review; pilot 3 reels.
5. **Ingest 10–15 reels** — verify quality and rights; reach 30+ shots.
6. **Rebuild clips/latents** — phase by offset, cap per shot/phase, shot-disjoint splits.
7. **Fix Phase 1/2** — motion-only targets, matching adapter loaders, resume/mixed precision/validation.
8. **Implement Stage 6** — test evaluation, per-class recall, synthetic ground truth, real gate.
9. **Truth-up README/web** — mark the current 5 videos as a demo until a real matrix exists.

---

## 6. P1 items

- `training/01_train_spatial_lora.py` and `02_train_motion_lora.py` implement gradient accumulation, mixed precision, and gradient checkpointing, but still lack a validation loop and resumable optimizer/scheduler/RNG state.
- Phase 2’s temporal-difference loss is applied to predicted noise; noise is independent across frames, so this can encourage over-smoothing. Evaluate on predicted x0 or ablate it.
- `scripts/02b_audit_class_counts.py`: `n == 1` groups go to train only and `n == 2` groups receive either val or test (not both), so a few rare `burst_type × yield_tier` cells lack coverage in one partition. Use iterative/grouped stratification if stricter coverage is required.
- `scripts/03_cut_and_filter.py`: compilation reels are now safely rejected, but `shot_segments.csv` still has only the `Sunset` segment. Every other reel must be segmented before it can contribute clips.
- `web/app.js`: `videoMap` still only varies by burst type, so the demo playback does not change with yield/terrain/lighting/phase/film/smoke.
- `web/app.js`: `state.lighting` is never assigned and there is no lighting control, so the UI always requests `day` even though `night`/`dawn` exist in the schema.
- `scripts/05_train_classifier.py` evaluates only `burst_type`, `yield_tier`, and `phase`; terrain, lighting, and film-look control are advertised but not audited by the Stage 6 gate.
- `scripts/04_preencode_latents.py`: `--all` is `store_true` with `default=True` (a no-op); keep it for CLI compatibility or remove it.
- `requirements.txt` omits `tensorboard` (used by config), and the notebook installs unpinned `imageio`/`imageio-ffmpeg`.
- `data/sources.csv` still carries raw Internet Archive CC BY-NC-ND tags on several rows; `docs/LICENSING_RECONCILIATION.md` documents the public-domain determination, but each row should keep an explicit reviewer sign-off.

---

## 7. Quick checks before handing work back

```bash
# 1. Manifest sanity
python -c "import pandas as pd; m=pd.read_csv('data/manifest.csv'); print(len(m), m.groupby(['operation','shot_name']).ngroups, m['split'].value_counts().to_dict()); print('latents', m['latent_path'].notna().sum())"

# 2. Schema + data gate validation (fails if <30 shots, empty val/test, or <100% latents)
python scripts/validate_manifest.py

# 3. Batch smoke test: one batch per split and mode
python scripts/smoke_test_dataset.py

# 4. Split disjointness + non-empty val/test gate
python scripts/check_split_leakage.py
```

These gates are enforced by default; use `--no-gate` only when intentionally inspecting a partial dataset. The current tree is expected to **fail** steps 2 and 4 (1 shot, 0 val/test, 14/30 latents).

If these scripts do not exist yet, create them as part of the corresponding fix. Report actual numbers, not intended ones.
