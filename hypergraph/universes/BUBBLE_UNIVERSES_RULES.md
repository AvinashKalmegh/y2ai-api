# Bubble Universes Rules

Curated thematic universes used by the Hypergraph Radar system.
Each universe tracks a macro investment theme through a set of tickers organized into pillars.

---

## Structure

Every bubble universe defines:

| Field | Description |
|-------|-------------|
| id | Unique identifier (e.g. `ai_infra`) |
| name | Display name |
| description | One-line theme summary |
| tickers | Flat list of all constituent stock symbols |
| pillars | Dict mapping pillar name -> list of tickers in that pillar |
| stage | Current lifecycle stage of the bubble |

---

## Lifecycle Stages

Universes progress through these stages:

```
Formation -> Early Expansion -> Mid Expansion -> Late Expansion -> Peak -> Deflation
```

---

## Active Universes

### 1. AI Infrastructure

Source: `ai_infra.py`

| Field | Value |
|-------|-------|
| ID | `ai_infra` |
| Description | AI chips, data centers, cloud, enterprise adoption |
| Total tickers | 43 |
| Current stage | Late Expansion |

#### Pillars

| Pillar | Tickers | Count |
|--------|---------|------:|
| Infrastructure & Energy | TSM, ASML, NVDA, AMD, MU, INTC, AVGO, VRT, CEG, NRG, EQIX, DLR, KLAC, LRCX, AMAT, QCOM | 16 |
| Enterprise Adoption | MSFT, AMZN, GOOGL, META, CRM, NOW, SNOW, PLTR, ADBE, ORCL, MDB, DDOG, ZS | 13 |
| Productivity & Labor | NET, CRWD, PANW | 3 |
| Demand Dynamics | TSLA, SHOP, UBER | 3 |
| Macro & Policy | NXPI, ON, SMCI, ARM | 4 |
| Financial & Market | GS, MS, JKS, FSLR | 4 |

---

### 2. Crypto & Bitcoin

Source: `crypto.py`

| Field | Value |
|-------|-------|
| ID | `crypto` |
| Description | Miners, exchanges, holders, infrastructure |
| Total tickers | 18 |
| Current stage | Mid Expansion |

#### Pillars

| Pillar | Tickers | Count |
|--------|---------|------:|
| Miners | MARA, RIOT, CLSK, CIFR, HUT, BITF | 6 |
| Exchanges | COIN, HOOD, IBKR, SCHW | 4 |
| Holders | MSTR, TSLA, PYPL, NU | 4 |
| Infrastructure | NVDA, AMD, ANET, MELI | 4 |

---

### 3. Nuclear Renaissance

Source: `nuclear.py`

| Field | Value |
|-------|-------|
| ID | `nuclear` |
| Description | Utilities, SMR developers, uranium miners |
| Total tickers | 15 |
| Current stage | Early Expansion |

#### Pillars

| Pillar | Tickers | Count |
|--------|---------|------:|
| Utilities | CEG, VST, NRG, DUK, SO, NEE | 6 |
| SMR Developers | SMR, OKLO, LEU, BWXT, FLR | 5 |
| Uranium & Fuel | CCJ, UEC, UUUU, DNN | 4 |

---

## Cross-Universe Overlap

Some tickers appear in multiple universes:

| Ticker | Universes |
|--------|-----------|
| NVDA | AI Infrastructure, Crypto |
| AMD | AI Infrastructure, Crypto |
| TSLA | AI Infrastructure, Crypto |
| CEG | AI Infrastructure, Nuclear |
| NRG | AI Infrastructure, Nuclear |

---

## Rules for Adding a New Universe

1. Create a new file in `hypergraph/universes/` (e.g. `defense.py`)
2. Define a dict with required fields: `id`, `name`, `description`, `tickers`, `pillars`, `stage`
3. Every ticker must belong to exactly one pillar within the universe
4. Set `stage` to the current lifecycle position
5. Import and register in `__init__.py` under `ALL_UNIVERSES`

### Pillar Guidelines

- Each pillar represents a distinct sub-theme or value chain segment
- Pillars should be mutually exclusive within a universe (no ticker in two pillars)
- Aim for 3-6 pillars per universe
- Minimum 3 tickers per pillar for meaningful signal

### Stage Assignment

| Stage | Description |
|-------|-------------|
| Formation | Theme emerging, early movers only, low institutional awareness |
| Early Expansion | Narrative gaining traction, capital beginning to flow in |
| Mid Expansion | Broad participation, multiple pillars activating |
| Late Expansion | Crowded positioning, high DM scores across pillars |
| Peak | Maximum euphoria, divergence between pillars beginning |
| Deflation | Capital rotating out, pillar-by-pillar breakdown |

---

## Registry

All universes are loaded via `hypergraph/universes/__init__.py`:

```python
ALL_UNIVERSES = {
    'ai_infra': AI_INFRA_UNIVERSE,
    'crypto': CRYPTO_UNIVERSE,
    'nuclear': NUCLEAR_UNIVERSE,
}
```

Access any universe with `ALL_UNIVERSES['ai_infra']`.
