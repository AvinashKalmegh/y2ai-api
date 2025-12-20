"""
Nuclear Renaissance Universe — 15 stocks, 3 pillars
"""

NUCLEAR_UNIVERSE = {
    'id': 'nuclear',
    'name': 'Nuclear Renaissance',
    'description': 'Utilities, SMR developers, uranium miners',
    
    'tickers': [
        # Utilities with Nuclear (6)
        'CEG', 'VST', 'NRG', 'DUK', 'SO', 'NEE',
        # SMR & Developers (5)
        'SMR', 'OKLO', 'LEU', 'BWXT', 'FLR',
        # Uranium & Fuel (4)
        'CCJ', 'UEC', 'UUUU', 'DNN',
    ],
    
    'pillars': {
        'Utilities': ['CEG', 'VST', 'NRG', 'DUK', 'SO', 'NEE'],
        'SMR Developers': ['SMR', 'OKLO', 'LEU', 'BWXT', 'FLR'],
        'Uranium & Fuel': ['CCJ', 'UEC', 'UUUU', 'DNN'],
    },
    
    'stage': 'Early Expansion',
}