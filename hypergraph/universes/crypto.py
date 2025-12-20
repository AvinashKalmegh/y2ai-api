"""
Crypto/Bitcoin Universe — 18 stocks, 4 pillars
"""

CRYPTO_UNIVERSE = {
    'id': 'crypto',
    'name': 'Crypto & Bitcoin',
    'description': 'Miners, exchanges, holders, infrastructure',
    
    'tickers': [
        # Miners (6)
        'MARA', 'RIOT', 'CLSK', 'CIFR', 'HUT', 'BITF',
        # Exchanges & Brokers (4)
        'COIN', 'HOOD', 'IBKR', 'SCHW',
        # Corporate Holders (4)
        'MSTR', 'TSLA', 'PYPL', 'NU',
        # Infrastructure & Services (4)
        'NVDA', 'AMD', 'ANET', 'MELI',
    ],
    
    'pillars': {
        'Miners': ['MARA', 'RIOT', 'CLSK', 'CIFR', 'HUT', 'BITF'],
        'Exchanges': ['COIN', 'HOOD', 'IBKR', 'SCHW'],
        'Holders': ['MSTR', 'TSLA', 'PYPL', 'NU'],
        'Infrastructure': ['NVDA', 'AMD', 'ANET', 'MELI'],
    },
    
    'stage': 'Mid Expansion',
}
