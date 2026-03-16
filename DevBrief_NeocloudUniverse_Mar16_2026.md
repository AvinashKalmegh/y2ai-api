# Developer Brief — Neocloud Cluster Universe Expansion
## Y2AI Research | March 16, 2026 | Priority: This Week

---

## CONTEXT

On March 16, 2026, Meta announced a $27 billion AI infrastructure deal with
Nebius Group (NBIS). Nebius was not in our 592-ticker universe. The DM signal
would have been building ahead of this announcement — we missed it because the
ticker wasn't tracked.

This brief defines a systematic approach to ensure emerging infrastructure
players are added to the universe before major announcements rather than after.

---

## THE NEOCLOUD CLUSTER

Neocloud companies build GPU-native data centers specifically for AI workloads.
They are distinct from traditional data center REITs (EQIX, DLR) which are
real estate companies. Neoclouds are infrastructure companies receiving
massive hyperscaler contracts and Nvidia strategic investment.

**Add these tickers to the universe immediately:**

| Ticker | Company | Why |
|--------|---------|-----|
| NBIS | Nebius Group | $27B Meta deal today. $17.4B Microsoft deal. Nvidia $2B stake. |
| CRWV | CoreWeave | Nvidia-backed, $23B IPO March 2025, major hyperscaler contracts |
| NSCL | NScale | UK-based neocloud, Nvidia $2B investment, growing fast |

Request backfill for all three — same format as existing DM backfill sheets.
Date range: from IPO/listing date through current.

---

## THE SYSTEMATIC GAP PROBLEM

We missed Nebius because there was no process for identifying emerging universe
candidates before they become headline news. We need a repeatable scan.

**Three triggers that should automatically flag a ticker for universe addition:**

**Trigger 1 — Nvidia Strategic Investment**
Nvidia has been systematically investing $2B stakes in neocloud companies.
Each investment is a signal that the company is becoming infrastructure-critical.

Monitor: Nvidia quarterly 13F filings + press releases for new strategic stakes.
Current Nvidia-backed neoclouds: Nebius, CoreWeave, NScale, Lambda Labs.
When Nvidia announces a new strategic investment → flag for universe addition.

**Trigger 2 — Hyperscaler Contract > $1 Billion**
When a company signs a contract with Meta, Microsoft, Google, or Amazon
exceeding $1 billion, it has become institutional infrastructure.

Monitor: SEC 8-K filings for material contract announcements from these four
hyperscalers. EDGAR full-text search for "infrastructure agreement" +
company name + dollar value.

**Trigger 3 — New Nasdaq/NYSE Listing in AI Infrastructure Category**
Nebius listed in 2024. CoreWeave IPO'd in March 2025. New listings in the
AI infrastructure category should be automatically flagged.

Monitor: Nasdaq new listing announcements filtered by SIC codes:
- 7374 (Computer Processing and Data Preparation)
- 7372 (Prepackaged Software)
- 3577 (Computer Peripheral Equipment)

---

## IMPLEMENTATION TASKS

**Task 1 — Add three tickers immediately (today)**
Add NBIS, CRWV, NSCL to the DM universe pipeline.
Request backfill from Polygon for all three from listing date.
Add to the Datacenter cluster in the six-cluster scan.
Confirm DM scores appear in DM_Latest sheet by tomorrow morning.

**Task 2 — Build universe candidate watchlist (this week)**
Create a new Google Sheet tab called `Universe_Candidates`.
Columns: Ticker, Company, Category, Trigger Type, Date Flagged, DM Status,
Decision (ADD / WATCH / REJECT), Decision Date.

Populate with current candidates:
- NBIS — Neocloud — Hyperscaler contract — Add
- CRWV — Neocloud — Hyperscaler contract — Add
- NSCL — Neocloud — Nvidia investment — Add
- DELL — Hardware — Already in universe — Confirm
- ARM — Semiconductor — Review for addition

**Task 3 — Weekly EDGAR scan (next week)**
Build a lightweight Python script that runs weekly and checks:
1. Nvidia 13F filings for new strategic investments
2. 8-K filings from Meta, Microsoft, Google, Amazon for infrastructure contracts > $1B
3. New Nasdaq listings in AI infrastructure SIC codes

Output: Email alert or Google Sheet row when a new candidate is detected.
This is the systematic early warning system for universe gaps.

---

## NEOCLOUD CLUSTER DEFINITION

Once NBIS, CRWV, NSCL are added, create a formal Neocloud cluster in the
six-cluster scan alongside existing clusters (SEMIS, NUCLEAR, DATACENTER,
CYBER, CONSULTING, SAAS).

**NEOCLOUD cluster tickers:**
NBIS (Nebius), CRWV (CoreWeave), NSCL (NScale)

Add VRT (Vertiv — cooling infrastructure) and SMCI (SuperMicro — server
infrastructure) as adjacent names that benefit from neocloud buildout.

**Cluster DM interpretation:**
- NEOCLOUD cluster DM rising = hyperscaler contract pipeline building
- NEOCLOUD attractor (sustained DM > 70 for 3 weeks) = major contract
  announcement likely within 28-42 days (based on 6b formation lead times)

This is exactly the signal that would have detected Nebius before today's
Meta announcement.

---

## WHAT THIS WOULD HAVE SHOWN

Had Nebius been in the universe:
- Nvidia $2B stake announced March 10 → DM spike likely
- Pre-announcement institutional accumulation → DM rising trend
- Three-signal convergence (DM + HMS + intraday confirmation) → T1 signal
  firing before the $27B Meta announcement today

This is the PE Echo → DM causal chain working in real time. Private capital
(Nvidia strategic stake) precedes public market DM response. The $27B contract
announcement is the narrative explanation — Stage 8 of our causal chain.
The signal would have fired at Stage 4-6.

---

## PRIORITY ORDER

1. Add NBIS, CRWV, NSCL to universe — today
2. Request Polygon backfill for all three — today
3. Create Universe_Candidates watchlist tab — this week
4. Build EDGAR weekly scan script — next week

---

*Y2AI Research | Neocloud Universe Expansion Brief | March 16, 2026*
*Triggered by: Meta-Nebius $27B deal announcement, March 16, 2026*
*Gap identified: NBIS not in universe despite Nvidia $2B stake March 10, 2026*
