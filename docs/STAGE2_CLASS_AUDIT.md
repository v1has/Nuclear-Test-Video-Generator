# Stage 2: Atmospheric Shot Class-Count & Split Audit Report

**Date Generated:** 2026-10-19 18:58:50  
**Source Dataset:** [`data/doe_nv209_atmospheric_shots.csv`](file:///D:/Projects/Nuclear Test video generator/Nuclear-Test-Video-Generator/data/doe_nv209_atmospheric_shots.csv)  
**Official Reference:** U.S. Department of Energy Report DOE/NV-209 Rev 16 (*United States Nuclear Tests: July 1945 through September 1992*)

---

## 1. Executive Summary & Gating Decision

- **Total Historical Atmospheric & Underwater Tests (1945–1962):** **206 shots**
- **Validation Gate Status:** **PASSED WITH STRATIFIED GROUPING REQUIREMENT**
- **Critical Finding (Rare Shots):** Only **5 underwater nuclear tests** were ever conducted by the United States (*Baker*, *Wigwam*, *Wahoo*, *Umbrella*, *Swordfish*).
- **Split Rule:** A naive random split fails by leaving zero underwater shots in either the validation or test sets. A **stratified grouped split** is mandatory to guarantee representation across partitions.

---

## 2. Parameter Distributions & Confounding Analysis

### 2.1 Burst Type Distribution
| Burst Type | Total Shots | % of Total | Operational Description |
|---|---|---|---|
| `surface-tower` | 116 | 56.3% | Tower, surface, barge, and crater detonations |
| `air-burst` | 85 | 41.3% | Airdrop, balloon, and rocket high-altitude tests |
| `underwater` | 5 | 2.4% | Deep water and shallow lagoon detonations |

### 2.2 Burst Type × Setting (Physical Confounding Matrix)
```text
setting        desert-day  desert-night  tropical-dawn  tropical-ocean  All
burst_type                                                                 
air-burst              24            22              6              33   85
surface-tower          27            32             37              20  116
underwater              0             0              0               5    5
All                    51            54             43              58  206
```

> [!WARNING]
> **Severe Physical Confounding Detected:**
> - **Underwater tests are 100% Pacific:** All 5 underwater detonations occurred in Pacific proving grounds (Bikini, Enewetak, offshore California). There are zero desert underwater tests. In single-parameter perturbation tests, `underwater` will invariably be entangled with `tropical-ocean` features.
> - **Megaton-class tests are 100% Pacific:** Continental U.S. safety constraints strictly limited Nevada tests to yields <100 kt (except Hood which was ~74 kt). Every `megaton-class` explosion was conducted in the Pacific.

### 2.3 Burst Type × Yield Tier
```text
yield_tier     intermediate-yield  megaton-class  sub-kiloton  tactical-yield  All
burst_type                                                                        
air-burst                      34             12            1              38   85
surface-tower                  46             16           15              39  116
underwater                      2              0            0               3    5
All                            82             28           16              80  206
```

### 2.4 Setting × Yield Tier
```text
yield_tier      intermediate-yield  megaton-class  sub-kiloton  tactical-yield  All
setting                                                                            
desert-day                      18              0            8              25   51
desert-night                    16              0            6              32   54
tropical-dawn                   22             14            0               7   43
tropical-ocean                  26             14            2              16   58
All                             82             28           16              80  206
```

---

## 3. Rare Class Audit: The 5 Historical Underwater Tests

| Test # | Operation | Shot Name | Date | Official Yield | Setting | Location | Stratified Split Assignment |
|---|---|---|---|---|---|---|---|
| 3 | Crossroads | Baker | 1946-07-24 | 21 kt | `tropical-ocean` | Bikini Atoll | **`train`** |
| 64 | Wigwam | Wigwam | 1955-05-14 | 30 kt | `tropical-ocean` | Pacific | **`val`** |
| 125 | Newsreel | Wahoo | 1958-05-16 | 9 kt | `tropical-ocean` | Enewetak Atoll | **`val`** |
| 133 | Newsreel | Umbrella | 1958-06-09 | 8 kt | `tropical-ocean` | Enewetak Atoll | **`train`** |
| 238 | Dominic | Swordfish | 1962-05-11 | Low | `tropical-ocean` | Pacific | **`test`** |

---

## 4. Splitting Methodology Comparison

### 4.1 Naive Random Split (Failure Demonstration)
With an unconstrained random 80/10/10 split (Seed 42):
```text
naive_split    test  train  val
burst_type                     
air-burst         6     67   12
surface-tower    15     93    8
underwater        1      4    0
```
*Note: Under naive random splitting, rare classes risk zero representation in validation or test partitions, causing evaluation failure.*

### 4.2 Stratified Grouped Split (Production Implementation)
Partition allocation enforcing representative sampling across all burst types and yield tiers:
```text
split          test  train  val
burst_type                     
air-burst         8     69    8
surface-tower    13     90   13
underwater        1      2    2
```

#### Split Breakdown by Yield Tier:
```text
split               test  train  val
yield_tier                          
intermediate-yield     8     65    9
megaton-class          3     22    3
sub-kiloton            2     12    2
tactical-yield         9     62    9
```

#### Split Breakdown by Setting:
```text
split           test  train  val
setting                         
desert-day         5     43    3
desert-night       7     39    8
tropical-dawn      3     36    4
tropical-ocean     7     43    8
```

---

## 5. Summary & Next Stage Recommendations

1. **Manifest Generation:** `manifest.csv` must link scraped video cuts directly to the stratified split assignments in [`data/doe_nv209_atmospheric_splits.csv`](file:///D:/Projects/Nuclear Test video generator/Nuclear-Test-Video-Generator/data/doe_nv209_atmospheric_splits.csv).
2. **Evaluation Protocol:**
   - For `underwater` generation, evaluate against the dedicated test shot (*Swordfish* or *Umbrella*) and validation shot (*Wahoo*).
   - In quantitative control verification (Stage 6), report metrics both overall and sliced by burst type to monitor rare class recall.
