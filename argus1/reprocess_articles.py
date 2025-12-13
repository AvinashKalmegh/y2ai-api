"""
Quick reprocess script for existing articles
Fixes the RawArticle article_hash issue
"""

import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def reprocess_articles(start_date: str, end_date: str, limit: int = 10):
    """Reprocess articles with enhanced signal detection"""
    
    from supabase import create_client
    from argus1.processor import ClaudeProcessor
    from argus1.aggregator import RawArticle
    
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_KEY')
    
    if not supabase_url or not supabase_key:
        print("ERROR: Supabase not configured")
        return
    
    supabase = create_client(supabase_url, supabase_key)
    processor = ClaudeProcessor()
    
    # Fetch articles
    response = (
        supabase.table('processed_articles')
        .select('*')
        .gte('published_at', start_date)
        .lte('published_at', end_date)
        .order('published_at', desc=True)
        .limit(limit)
        .execute()
    )
    
    articles = response.data
    print(f"Found {len(articles)} articles to reprocess")
    
    stats = {"processed": 0, "failed": 0}
    
    for i, article_data in enumerate(articles):
        print(f"\nProcessing {i+1}/{len(articles)}: {article_data.get('title', '')[:60]}...")
        
        # Create RawArticle WITHOUT article_hash (it's a computed property)
        raw = RawArticle(
            source_type=article_data.get('source_type', 'unknown'),
            source_name=article_data.get('source_name', ''),
            title=article_data.get('title', ''),
            url=article_data.get('url', ''),
            content=article_data.get('content', '') or '',  # Handle None
            published_at=article_data.get('published_at', ''),
            author=article_data.get('author'),
            ticker=article_data.get('ticker'),
            keywords_used=article_data.get('keywords_used', [])
        )
        
        # Process with Claude
        result = processor.categorize_and_extract(raw)
        
        if result:
            # Build update payload
            update_data = {
                "y2ai_category": result.y2ai_category,
                "impact_score": result.impact_score,
                "sentiment": result.sentiment,
                "extracted_facts": result.extracted_facts,
                "companies_mentioned": result.companies_mentioned,
                "dollar_amounts": result.dollar_amounts,
                "key_quotes": result.key_quotes,
                
                # Capex signals
                "capex_detected": result.capex_detected,
                "capex_direction": result.capex_direction,
                "capex_magnitude": result.capex_magnitude,
                "capex_company": result.capex_company,
                "capex_amount": result.capex_amount,
                "capex_context": result.capex_context,
                
                # Energy signals
                "energy_detected": result.energy_detected,
                "energy_event_type": result.energy_event_type,
                "energy_direction": result.energy_direction,
                "energy_region": result.energy_region,
                "energy_context": result.energy_context,
                
                # Compute signals
                "compute_detected": result.compute_detected,
                "compute_event_type": result.compute_event_type,
                "compute_direction": result.compute_direction,
                "compute_companies_affected": result.compute_companies_affected,
                "compute_context": result.compute_context,
                
                # Depreciation signals
                "depreciation_detected": result.depreciation_detected,
                "depreciation_event_type": result.depreciation_event_type,
                "depreciation_amount": result.depreciation_amount,
                "depreciation_company": result.depreciation_company,
                "depreciation_context": result.depreciation_context,
                
                # Veto signals
                "veto_detected": result.veto_detected,
                "veto_trigger_type": result.veto_trigger_type,
                "veto_severity": result.veto_severity,
                "veto_context": result.veto_context,
                
                # Newsletter hints
                "include_in_weekly": result.include_in_weekly,
                "suggested_pillar": result.suggested_pillar,
                "one_line_summary": result.one_line_summary,
                
                # Thesis fields (THE NEW ONES!)
                "thesis_infrastructure_support": getattr(result, 'thesis_infrastructure_support', False),
                "thesis_bubble_warning": getattr(result, 'thesis_bubble_warning', False),
                "thesis_constraint_evidence": getattr(result, 'thesis_constraint_evidence', False),
                "thesis_demand_validation": getattr(result, 'thesis_demand_validation', False),
                "thesis_explanation": getattr(result, 'thesis_explanation', None),
                
                "reprocessed_at": datetime.utcnow().isoformat()
            }
            
            try:
                supabase.table('processed_articles').update(update_data).eq(
                    'id', article_data['id']
                ).execute()
                
                stats['processed'] += 1
                
                # Show thesis results
                print(f"  ✅ thesis_support={result.thesis_infrastructure_support}, bubble_warning={result.thesis_bubble_warning}")
                if result.thesis_explanation:
                    print(f"     {result.thesis_explanation[:80]}...")
                    
            except Exception as e:
                print(f"  ❌ Update error: {e}")
                stats['failed'] += 1
        else:
            print(f"  ❌ Claude processing failed")
            stats['failed'] += 1
    
    print(f"\n{'='*60}")
    print(f"COMPLETE: Processed {stats['processed']}, Failed {stats['failed']}")
    print(f"{'='*60}")
    
    return stats


if __name__ == "__main__":
    import sys
    
    # Default: reprocess 5 articles from today
    start_date = sys.argv[1] if len(sys.argv) > 1 else "2025-12-11"
    end_date = sys.argv[2] if len(sys.argv) > 2 else start_date
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    
    print(f"Reprocessing articles from {start_date} to {end_date} (limit: {limit})")
    reprocess_articles(start_date, end_date, limit)