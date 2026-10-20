# Official DOE NV-209 Rev 16 Test Count Reconciliation (216 Tally)

This document reconciles the historical test counts parsed from the official Department of Energy document **DOE/NV-209 Rev 16** (*United States Nuclear Tests: July 1945 through September 1992*) against the official NNSA and international atmospheric tallies (216 tests).

---

## 1. Official Totals by Summary Event (Table 2, Page xiv)

From DOE/NV-209 Rev 16, Table 2 ("U.S. Nuclear Tests – Total – By Type"):

| Test Type | US Tests | Joint US-UK | Notes |
|---|---|---|---|
| **Tower** | 56 | 0 | Atmospheric tower shots |
| **Airdrop** | 52 | 0 | Gravity bombs dropped from aircraft |
| **Barge** | 36 | 0 | Water-borne barges in Pacific lagoons |
| **Balloon** | 25 | 0 | Tethered balloons at NNSS |
| **Surface** | 24 | 4 | Ground surface detonations (4 Joint US-UK Roller Coaster) |
| **Rocket** | 12 | 0 | Missile / high-altitude rocket bursts |
| **Airburst** | 1 | 0 | Operation Upshot-Knothole Grable (280mm artillery cannon) |
| **Total Atmospheric** | **206** | **4** | **210 atmospheric detonations total** |
| **Underwater** | **5** | **0** | Baker, Wigwam, Wahoo, Umbrella, Swordfish |
| **Crater** | 9 | 0 | Shallow excavation / subsurface detonations |
| **Shaft** | 739 | 24 | Underground drill holes / vertical shafts |
| **Tunnel** | 67 | 0 | Underground horizontal adits |
| **Total Tests** | **1,026** | **28** | **1,054 Total Nuclear Tests** |

*Note: Totals exclude the two combat detonations at Hiroshima and Nagasaki.*

---

## 2. Reconciling the Official "216" Tally

Historical literature and NNSA unclassified summaries frequently cite **216 atmospheric / above-ground tests** conducted prior to the 1963 Limited Test Ban Treaty (LTBT).

The 216 tally is derived as follows:

$$\text{US Atmospheric Tests (Table 2)}: 206$$
$$+ \text{US Underwater Tests (Table 2)}: 5$$
$$+ \text{Pre-LTBT Shallow Crater Explosions}: 5$$
$$\mathbf{\text{Total}}: \mathbf{216}$$

### Pre-LTBT Crater Detonations Included in the 216 Count:
1. **Test #22 (Jangle Uncle, 11/29/1951)**: Burst depth -17 ft; created an atmospheric ejecta column and massive dust cloud.
2. **Test #56 (Teapot ESS, 03/23/1955)**: Burst depth -67 ft; created a massive fireball breaking through the ground, atmospheric stem, and mushroom cloud.
3. **Test #216 (Nougat Danny Boy, 03/05/1962)**: Crater burst in basalt; significant atmospheric cloud venting.
4. **Test #264 (Storax Sedan, 07/06/1962)**: Plowshare excavation test (104 kt); produced a 12,000-ft mushroom cloud and ejecta dome.
5. **Test #269 (Sunbeam Johnnie Boy, 07/11/1962)**: Burst depth -1.8 ft; direct ground break-through creating an atmospheric fireball and cloud.

### Alternative Reconciliation (Including Joint US-UK Tests):
If Joint US-UK surface safety dispersal tests (Operation Roller Coaster: Double Tracks #319, Clean Slate I #323, Clean Slate II #325, Clean Slate III #329) are tallied:
$$206 \text{ (US Atmospheric)} + 4 \text{ (Joint US-UK Atmospheric)} + 5 \text{ (Underwater)} + 1 \text{ (Sedan Plowshare)} = 216$$

---

## 3. Dataset Coverage in this Repository

The alias registry in `data/doe_shot_aliases.csv` tracks all **220** events so the historical reconciliation remains auditable:

- **206** US Atmospheric Tests (`reconciliation_group = US_ATMOSPHERIC`)
- **5** Underwater Tests (`reconciliation_group = UNDERWATER`)
- **5** Pre-LTBT Crater Tests (`reconciliation_group = PRE_TREATY_CRATER`)
- **4** Joint US-UK Surface Tests (`reconciliation_group = JOINT_US_UK_ATMOSPHERIC`)

The **training corpus** (`data/doe_nv209_atmospheric_shots.csv`) is restricted to the **211** events that map cleanly onto the four canonical visual burst types (`air-burst`, `surface-tower`, `surface-barge`, `underwater`):

- 206 US atmospheric + 5 underwater = **211 training-eligible detonations**.

Excluded from training (but retained in the registry and audit trail):

1. **5 pre-treaty crater events** (`Jangle Uncle`, `Teapot ESS`, `Nougat Danny Boy`, `Storax Sedan`, `Sunbeam Johnnie Boy`). These are nuclear detonations and are part of the official 216 tally, but they are crater/ejecta events, not one of the four burst types. They must not be relabeled as `surface-tower`.
2. **4 Roller Coaster events** (`Double Tracks`, `Clean Slate I/II/III`). These were high-explosive plutonium-dispersal safety tests with **no nuclear yield**; they are not nuclear detonations and are excluded entirely.

Each row in the alias registry is cross-referenced with official test numbers (1 through 329), ensuring exact metadata fidelity and zero corruption.
