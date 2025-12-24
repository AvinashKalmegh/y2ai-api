-- ============================================================================
-- ARGUS-1 Master Signals Table
-- Stores unified regime assessment from analytical layer
-- ============================================================================

CREATE TABLE IF NOT EXISTS argus_master_signals (
    id SERIAL PRIMARY KEY,
    date DATE UNIQUE NOT NULL,
    calculated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Core regime assessment
    regime VARCHAR(20) NOT NULL,  -- NORMAL, ELEVATED, TENSION, FRAGILE, BREAK
    authority VARCHAR(20) NOT NULL,  -- MARKET, STRUCTURAL, NARRATIVE, BREAK
    confidence VARCHAR(10) NOT NULL,  -- HIGH, MEDIUM, LOW
    
    -- AMRI composite and decomposition
    amri_composite FLOAT NOT NULL,
    amri_s FLOAT,  -- Structural capacity
    amri_b FLOAT,  -- Behavioral pressure
    amri_c FLOAT,  -- Catalyst risk
    
    -- TTI (Time-to-Instability)
    tti_display VARCHAR(100),
    tti_rate FLOAT,
    tti_days_to_fragile INTEGER,
    tti_days_to_break INTEGER,
    
    -- SAC (Shock Absorption Capacity)
    sac_composite FLOAT,
    sac_weakest VARCHAR(50),
    
    -- NST (News Sentiment)
    veto_active BOOLEAN DEFAULT FALSE,
    veto_count INTEGER DEFAULT 0,
    thesis_balance FLOAT,
    
    -- Contagion
    contagion_score FLOAT,
    contagion_regime VARCHAR(20),
    
    -- Fingerprint
    fingerprint_episode VARCHAR(50),
    fingerprint_match FLOAT,
    fingerprint_pattern VARCHAR(20),
    
    -- Rotation
    rotation_leader VARCHAR(50),
    rotation_laggard VARCHAR(50),
    rotation_spread FLOAT,
    
    -- Events
    next_event VARCHAR(100),
    days_to_event INTEGER,
    
    -- Recovery
    recovery_active BOOLEAN DEFAULT FALSE,
    recovery_strength FLOAT,
    
    -- Full JSON result for flexibility
    full_result JSONB
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_argus_master_date ON argus_master_signals(date DESC);
CREATE INDEX IF NOT EXISTS idx_argus_master_regime ON argus_master_signals(regime);
CREATE INDEX IF NOT EXISTS idx_argus_master_veto ON argus_master_signals(veto_active);

-- View for latest signal
CREATE OR REPLACE VIEW v_latest_argus_signal AS
SELECT * FROM argus_master_signals 
ORDER BY date DESC 
LIMIT 1;

-- View for recent regime history
CREATE OR REPLACE VIEW v_regime_history AS
SELECT 
    date,
    regime,
    authority,
    amri_composite,
    veto_active,
    contagion_score,
    tti_display
FROM argus_master_signals 
ORDER BY date DESC 
LIMIT 30;
