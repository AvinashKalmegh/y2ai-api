-- NST Integration Schema for FlowOS
-- Run this in your Supabase SQL editor to create the NST tables

-- =============================================================================
-- nst_mentions - Individual classified mentions from Google Alerts
-- =============================================================================

CREATE TABLE IF NOT EXISTS nst_mentions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- Source info
  source TEXT NOT NULL,
  headline TEXT NOT NULL,
  snippet TEXT,
  url TEXT,
  published_at TIMESTAMPTZ,
  
  -- NST classification fields
  bubble_type TEXT NOT NULL,
  attractor_mentioned BOOLEAN DEFAULT FALSE,
  attractor_ticker TEXT,
  alert_category TEXT NOT NULL DEFAULT 'theme',
  keywords_matched TEXT[],
  
  -- Claude classification
  sentiment_score FLOAT,
  sentiment_label TEXT,
  catalyst_type TEXT,
  
  -- Constraints
  CONSTRAINT valid_bubble_type CHECK (bubble_type IN (
    'AI/Compute', 'Energy/Grid', 'Crypto', 'Clean Energy'
  )),
  CONSTRAINT valid_alert_category CHECK (alert_category IN (
    'attractor', 'theme', 'stress', 'catalyst'
  )),
  CONSTRAINT valid_sentiment_label CHECK (sentiment_label IS NULL OR sentiment_label IN (
    'bullish', 'bearish', 'neutral'
  )),
  CONSTRAINT valid_catalyst_type CHECK (catalyst_type IS NULL OR catalyst_type IN (
    'POSITIVE_PENDING', 'NEGATIVE'
  ))
);

-- Indexes for nst_mentions
CREATE INDEX IF NOT EXISTS idx_nst_mentions_bubble_type ON nst_mentions(bubble_type);
CREATE INDEX IF NOT EXISTS idx_nst_mentions_published_at ON nst_mentions(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_nst_mentions_alert_category ON nst_mentions(alert_category);
CREATE INDEX IF NOT EXISTS idx_nst_mentions_attractor ON nst_mentions(attractor_mentioned) WHERE attractor_mentioned = TRUE;


-- =============================================================================
-- nst_daily_summary - Pre-aggregated daily stats per bubble type
-- =============================================================================

CREATE TABLE IF NOT EXISTS nst_daily_summary (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- Key fields
  date DATE NOT NULL,
  bubble_type TEXT NOT NULL,
  
  -- Aggregated counts
  total_mentions INTEGER DEFAULT 0,
  attractor_mentions INTEGER DEFAULT 0,
  stress_mentions INTEGER DEFAULT 0,
  catalyst_mentions INTEGER DEFAULT 0,
  
  -- Calculated metrics
  narrative_density FLOAT,
  density_level TEXT,
  
  -- Unique constraint for upsert
  UNIQUE(date, bubble_type),
  
  -- Constraints
  CONSTRAINT valid_summary_bubble_type CHECK (bubble_type IN (
    'AI/Compute', 'Energy/Grid', 'Crypto', 'Clean Energy'
  )),
  CONSTRAINT valid_density_level CHECK (density_level IS NULL OR density_level IN (
    'LOW', 'MODERATE', 'HIGH', 'EXTREME'
  ))
);

-- Indexes for nst_daily_summary
CREATE INDEX IF NOT EXISTS idx_nst_summary_bubble_type ON nst_daily_summary(bubble_type);
CREATE INDEX IF NOT EXISTS idx_nst_summary_date ON nst_daily_summary(date DESC);
CREATE INDEX IF NOT EXISTS idx_nst_summary_bubble_date ON nst_daily_summary(bubble_type, date DESC);


-- =============================================================================
-- ENABLE ROW LEVEL SECURITY (optional but recommended)
-- =============================================================================

-- Uncomment these if you want RLS enabled
-- ALTER TABLE nst_mentions ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE nst_daily_summary ENABLE ROW LEVEL SECURITY;

-- Allow all authenticated users to read
-- CREATE POLICY "Allow read access" ON nst_mentions FOR SELECT USING (true);
-- CREATE POLICY "Allow read access" ON nst_daily_summary FOR SELECT USING (true);

-- Allow service role to insert/update
-- CREATE POLICY "Allow service insert" ON nst_mentions FOR INSERT WITH CHECK (true);
-- CREATE POLICY "Allow service insert" ON nst_daily_summary FOR INSERT WITH CHECK (true);
-- CREATE POLICY "Allow service update" ON nst_daily_summary FOR UPDATE USING (true);


-- =============================================================================
-- HELPER FUNCTION: Calculate density for a date range
-- =============================================================================

CREATE OR REPLACE FUNCTION calculate_narrative_density(
  p_bubble_type TEXT,
  p_start_date DATE,
  p_end_date DATE
)
RETURNS TABLE (
  narrative_density FLOAT,
  total_mentions BIGINT,
  attractor_mentions BIGINT
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT 
    CASE 
      WHEN SUM(s.total_mentions) > 0 
      THEN ROUND(CAST(SUM(s.attractor_mentions) AS NUMERIC) / SUM(s.total_mentions), 3)
      ELSE 0.0
    END AS narrative_density,
    COALESCE(SUM(s.total_mentions), 0) AS total_mentions,
    COALESCE(SUM(s.attractor_mentions), 0) AS attractor_mentions
  FROM nst_daily_summary s
  WHERE s.bubble_type = p_bubble_type
    AND s.date BETWEEN p_start_date AND p_end_date;
END;
$$;


-- =============================================================================
-- VERIFICATION QUERIES (run after creating tables)
-- =============================================================================

-- Check tables exist
-- SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name LIKE 'nst%';

-- Check columns
-- SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'nst_mentions';
-- SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'nst_daily_summary';
