# Historical Footage Rights & Licensing Reconciliation

This document clarifies the legal basis and provenance reconciliation for archival films listed in [`data/sources.csv`](../data/sources.csv).

---

## 1. Statutory Public Domain Basis (17 U.S.C. § 105)

Under United States copyright law (17 U.S.C. § 105):
> *"Copyright protection under this title is not available for any work of the United States Government..."*

All historical nuclear test films cataloged in this repository were commissioned and produced by officers or employees of the United States Government as part of their official duties:
- **Lookout Mountain Laboratory (1352d Motion Picture Squadron, U.S. Air Force)**
- **Department of Defense (AFSWP / DASA)**
- **Atomic Energy Commission (AEC / DOE / NNSA)**
- **Joint Task Force 7 & 8 (JTF-7 / JTF-8)**

Consequently, these works entered the public domain *ab initio* upon creation and cannot be copyrighted by private individuals.

---

## 2. Reconciliation of Third-Party Internet Archive Upload Tags

In [`data/sources.csv`](../data/sources.csv), several items hosted on archive.org list metadata license tags such as `http://creativecommons.org/licenses/by-nc-nd/2.5/` (e.g., `OperationHARDTACK_UnderwaterTests1958`, `MilitaryEffectsStudiesonOperationCastle1954`).

### Legal Assessment:
1. **Invalidity of Third-Party CC Licensing on Public Domain Works:** Under Creative Commons policies and US intellectual property jurisprudence, an individual uploader who digitizes or uploads a public domain government film does not acquire copyright in the underlying work, and has no legal authority to attach restrictive Creative Commons licenses (such as BY-NC-ND) to government works.
2. **Rights Status Determination:**
   - `license`: Reflects the raw string from the archive.org metadata API for provenance tracking.
   - `rights_status`: `Public Domain`
   - `rights_basis`: `17 U.S.C. § 105 (Official work of U.S. Government)`
   - `allow_training`: `True`

---

## 3. Exclusion of Ineligible or Unverified Media

- **Bad Identifier Removed:** `gov.dod.dimoc.23521` was purged from `sources.csv` because it was discovered to resolve to an unrelated feature film trailer (*"A Few Good Men"*), not Operation Crossroads footage.
- **Strict Rights Filter:** Any archival reel without clear US Government provenance must be flagged as `Restricted / Needs Review` and excluded from model training until verified by a reviewer.
