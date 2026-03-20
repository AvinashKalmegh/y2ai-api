"""
Neocloud Universe — GPU-native AI data center infrastructure
============================================================
Companies building GPU-native data centers specifically for AI workloads.
Distinct from traditional data center REITs (EQIX, DLR) — these are
infrastructure companies receiving massive hyperscaler contracts and
Nvidia strategic investment.

Cluster DM interpretation:
  - NEOCLOUD cluster DM rising = hyperscaler contract pipeline building
  - NEOCLOUD attractor (sustained DM > 70 for 3 weeks) = major contract
    announcement likely within 28-42 days (based on 6b formation lead times)

Triggered by: Meta-Nebius $27B deal, March 16 2026
"""

NEOCLOUD_UNIVERSE = {
    'id': 'neocloud',
    'name': 'Neocloud Infrastructure',
    'description': 'GPU-native AI data centers, hyperscaler contract recipients, Nvidia-backed infrastructure',

    'tickers': [
        # Core Neoclouds (2) — GPU-native data center operators
        'NBIS', 'CRWV',
        # Adjacent Infrastructure (2) — benefit from neocloud buildout
        'VRT', 'SMCI',
    ],

    'pillars': {
        'Core Neoclouds': ['NBIS', 'CRWV'],
        'Adjacent Infrastructure': ['VRT', 'SMCI'],
    },

    'stage': 'Early Expansion',
}
