"""
AI Infrastructure Universe — 43 stocks, 6 pillars
"""

AI_INFRA_UNIVERSE = {
    'id': 'ai_infra',
    'name': 'AI Infrastructure',
    'description': 'AI chips, data centers, cloud, enterprise adoption',
    
    'tickers': [
        # Infrastructure & Energy (16)
        'TSM', 'ASML', 'NVDA', 'AMD', 'MU', 'INTC', 'AVGO', 'VRT',
        'CEG', 'NRG', 'EQIX', 'DLR', 'KLAC', 'LRCX', 'AMAT', 'QCOM',
        # Enterprise Adoption (13)
        'MSFT', 'AMZN', 'GOOGL', 'META', 'CRM', 'NOW', 'SNOW', 'PLTR',
        'ADBE', 'ORCL', 'MDB', 'DDOG', 'ZS',
        # Productivity & Labor (3)
        'NET', 'CRWD', 'PANW',
        # Demand Dynamics (3)
        'TSLA', 'SHOP', 'UBER',
        # Macro & Policy (4)
        'NXPI', 'ON', 'SMCI', 'ARM',
        # Financial & Market (4)
        'GS', 'MS', 'JKS', 'FSLR',
    ],
    
    'pillars': {
        'Infrastructure & Energy': ['TSM', 'ASML', 'NVDA', 'AMD', 'MU', 'INTC', 'AVGO', 'VRT', 'CEG', 'NRG', 'EQIX', 'DLR', 'KLAC', 'LRCX', 'AMAT', 'QCOM'],
        'Enterprise Adoption': ['MSFT', 'AMZN', 'GOOGL', 'META', 'CRM', 'NOW', 'SNOW', 'PLTR', 'ADBE', 'ORCL', 'MDB', 'DDOG', 'ZS'],
        'Productivity & Labor': ['NET', 'CRWD', 'PANW'],
        'Demand Dynamics': ['TSLA', 'SHOP', 'UBER'],
        'Macro & Policy': ['NXPI', 'ON', 'SMCI', 'ARM'],
        'Financial & Market': ['GS', 'MS', 'JKS', 'FSLR'],
    },
    
    'stage': 'Late Expansion',  # Formation → Early Expansion → Mid Expansion → Late Expansion → Peak → Deflation
}