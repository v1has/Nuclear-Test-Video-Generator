# Stage 2: Atmospheric Shot Class-Count & Split Audit Report

**Date Generated:** 2026-10-20 19:38:35  
**Source Dataset:** [`data/doe_nv209_atmospheric_shots.csv`](file:///D:/Projects/Nuclear Test video generator/Nuclear-Test-Video-Generator/data/doe_nv209_atmospheric_shots.csv)  
**Official Reference:** U.S. Department of Energy Report DOE/NV-209 Rev 16 (*United States Nuclear Tests: July 1945 through September 1992*)

---

## 1. Executive Summary & Gating Decision

- **Total Historical Detonations (1945–1963):** **211 shots**
- **Validation Gate Status:** **PASSED WITH STRATIFIED GROUPING REQUIREMENT**
- **Critical Finding (Rare Shots):** Only **5 underwater nuclear tests** were ever conducted by the United States (*Baker*, *Wigwam*, *Wahoo*, *Umbrella*, *Swordfish*).
- **Split Rule:** A naive random split risks leaving zero underwater shots in either validation or test. A **stratified grouped split** is mandatory to guarantee representation across partitions.

---

## 2. Parameter Distributions & Confounding Analysis

### 2.1 Canonical Burst Type Distribution
| Burst Type | Total Shots | % of Total | Operational Description |
|---|---|---|---|
| `air-burst` | 90 | 42.7% | Airdrop, balloon, and rocket high-altitude tests |
| `surface-tower` | 80 | 37.9% | Tower, ground surface, and crater excavations |
| `surface-barge` | 36 | 17.1% | Water-borne barge detonations in Pacific atolls |
| `underwater` | 5 | 2.4% | Shallow lagoon and deep-ocean detonations |

### 2.2 Yield Tier Distribution
| Yield Tier | Total Shots | % of Total | Nominal Range |
|---|---|---|---|
| `sub-kiloton` | 14 | 6.6% | < 1 kt |
| `tactical-yield` | 83 | 39.3% | 1 – 20 kt |
| `intermediate-yield` | 82 | 38.9% | 20 – 1,000 kt |
| `megaton-class` | 32 | 15.2% | >= 1,000 kt (1 Mt+) |

---

## 3. Naive Random vs. Stratified Split Comparison

### 3.1 Naive Random Split (Demonstrating Flaw)
```text
naive_split    test  train  val
burst_type                     
air-burst        10     68   12
surface-barge     3     30    3
surface-tower     9     65    6
underwater        0      5    0
```

### 3.2 Stratified Grouped Split (Production Target)
```text
split          test  train  val
burst_type                     
air-burst         9     72    9
surface-barge     4     28    4
surface-tower     8     64    8
underwater        1      3    1
```

### 3.3 Yield Tier Distribution Across Stratified Splits
```text
split               test  train  val
yield_tier                          
intermediate-yield     8     65    9
megaton-class          4     24    4
sub-kiloton            1     12    1
tactical-yield         9     66    8
```

### 3.4 Terrain Distribution Across Stratified Splits
```text
split           test  train  val
terrain                         
desert-flats      11     82    9
tropical-ocean    11     85   13
```

### 3.5 Lighting Distribution Across Stratified Splits
```text
split     test  train  val
lighting                  
dawn         7     63   10
day          9     54    6
night        6     50    6
```

### 3.6 Allocation of Rare Underwater Tests
| Test # | Operation | Shot Name | Date | Yield | Split Assignment |
|---|---|---|---|---|---|
| 3 | Crossroads | Baker | 1946-07-24 | 21 kt | **`val`** |
| 64 | Wigwam | Wigwam | 1955-05-14 | 30 kt | **`train`** |
| 125 | Hardtack I | Wahoo | 1958-05-16 | 9 kt | **`train`** |
| 133 | Hardtack I | Umbrella | 1958-06-09 | 8 kt | **`test`** |
| 238 | Dominic | Swordfish | 1962-05-11 | Low | **`train`** |

---

## 4. Key Takeaways & Guidelines for Stage 3

1. **Shot-Disjoint Validation:** No clips from the same shot may be shared across training and validation/test splits.
2. **Rare Class Guarantee:** `underwater` tests are strictly partitioned 3/1/1, guaranteeing evaluation signal.
3. **No Empty Split Leakage:** Every split has verified real shots, preventing empty fallback crashes.
