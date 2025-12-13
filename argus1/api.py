"""
ARGUS-1 API LAYER
FastAPI endpoints for collection, querying, and newsletter generation
"""

import os
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ARGUS-1 News Intelligence API",
    description="Y2AI news aggregation and processing system",
    version="1.0.0"
)


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class CollectionRequest(BaseModel):
    hours_back: int = 24
    sources: Optional[List[str]] = None  # None = all sources

class ArticleQuery(BaseModel):
    category: Optional[str] = None
    min_impact: Optional[float] = None
    sentiment: Optional[str] = None
    hours_back: int = 168
    limit: int = 100

class NewsletterRequest(BaseModel):
    days_back: int = 7
    edition_title: Optional[str] = None

class CollectionResult(BaseModel):
    run_id: int
    raw_count: int
    processed_count: int
    duration_seconds: float
    errors: List[str]

class ArticleResponse(BaseModel):
    article_hash: str
    source_type: str
    source_name: str
    title: str
    url: str
    published_at: str
    y2ai_category: str
    impact_score: float
    sentiment: str
    extracted_facts: List[str]
    companies_mentioned: List[str]
    dollar_amounts: List[str]

class StatsResponse(BaseModel):
    period_days: int
    total_raw: int
    total_processed: int
    by_source: dict
    by_category: dict


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    return {
        "service": "ARGUS-1 News Intelligence",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "collect": "POST /collect - Run collection from all sources",
            "articles": "GET /articles - Query processed articles",
            "categories": "GET /categories - Get category definitions",
            "stats": "GET /stats - Collection statistics",
            "newsletter": "POST /newsletter/generate - Generate newsletter draft"
        }
    }


@app.post("/collect", response_model=CollectionResult)
async def run_collection(request: CollectionRequest, background_tasks: BackgroundTasks):
    """
    Run a collection from all news sources.
    
    This collects from:
    - NewsAPI (if key configured)
    - Alpha Vantage (if key configured)
    - SEC EDGAR (always available)
    - RSS Feeds (always available)
    
    Then processes through Claude for categorization.
    """
    from .aggregator import NewsAggregator
    from .processor import ClaudeProcessor
    from .storage import get_storage
    
    start_time = datetime.utcnow()
    errors = []
    
    try:
        storage = get_storage()
        
        # Start collection run
        run_id = storage.start_collection_run(request.sources or ["all"])
        
        # Collect raw articles
        aggregator = NewsAggregator()
        raw_articles = aggregator.collect_all(hours_back=request.hours_back)
        
        # Store raw articles
        raw_dicts = [a.to_dict() for a in raw_articles]
        storage.insert_raw_articles(raw_dicts, collection_run_id=run_id)
        
        # Process through Claude
        processor = ClaudeProcessor()
        filtered = processor.quick_relevance_filter(raw_articles)
        processed = processor.process_batch(filtered, max_batch=50)
        
        # Store processed articles
        processed_dicts = [p.to_dict() for p in processed]
        storage.insert_processed_articles(processed_dicts)
        
        # Complete run
        duration = (datetime.utcnow() - start_time).total_seconds()
        storage.complete_collection_run(
            run_id, 
            raw_count=len(raw_articles),
            processed_count=len(processed),
            errors=errors
        )
        
        return CollectionResult(
            run_id=run_id,
            raw_count=len(raw_articles),
            processed_count=len(processed),
            duration_seconds=duration,
            errors=errors
        )
        
    except Exception as e:
        logger.error(f"Collection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/articles", response_model=List[ArticleResponse])
async def get_articles(
    category: Optional[str] = Query(None, description="Filter by Y2AI category"),
    min_impact: Optional[float] = Query(None, ge=0, le=1, description="Minimum impact score"),
    sentiment: Optional[str] = Query(None, description="Filter by sentiment"),
    hours_back: int = Query(168, description="Hours to look back"),
    limit: int = Query(100, le=500, description="Maximum results")
):
    """
    Query processed articles with filters.
    
    Categories: spending, constraints, data, policy, skepticism, smartmoney, china, energy, adoption
    Sentiments: bullish, bearish, neutral
    """
    from .storage import get_storage
    
    storage = get_storage()
    articles = storage.get_processed_articles(
        category=category,
        min_impact=min_impact,
        sentiment=sentiment,
        hours_back=hours_back,
        limit=limit
    )
    
    return [ArticleResponse(
        article_hash=a.get("article_hash", ""),
        source_type=a.get("source_type", ""),
        source_name=a.get("source_name", ""),
        title=a.get("title", ""),
        url=a.get("url", ""),
        published_at=a.get("published_at", ""),
        y2ai_category=a.get("y2ai_category", ""),
        impact_score=a.get("impact_score", 0),
        sentiment=a.get("sentiment", "neutral"),
        extracted_facts=a.get("extracted_facts", []),
        companies_mentioned=a.get("companies_mentioned", []),
        dollar_amounts=a.get("dollar_amounts", [])
    ) for a in articles]


@app.get("/categories")
async def get_categories():
    """Get Y2AI category definitions"""
    from .processor import Y2AI_CATEGORIES
    return Y2AI_CATEGORIES


@app.get("/stats", response_model=StatsResponse)
async def get_stats(days_back: int = Query(30, description="Days to analyze")):
    """Get collection and processing statistics"""
    from .storage import get_storage
    
    storage = get_storage()
    stats = storage.get_collection_stats(days_back=days_back)
    
    return StatsResponse(
        period_days=stats.get("period_days", days_back),
        total_raw=stats.get("total_raw", 0),
        total_processed=stats.get("total_processed", 0),
        by_source=stats.get("by_source", {}),
        by_category=stats.get("by_category", {})
    )


@app.post("/newsletter/generate")
async def generate_newsletter(request: NewsletterRequest):
    """
    Generate a newsletter draft from recent articles.
    
    Returns organized content ready for Y2AI Weekly publication.
    """
    from .storage import get_storage
    from .processor import NewsletterProcessor
    
    storage = get_storage()
    
    # Get articles organized by category
    articles_by_category = storage.get_newsletter_ready_articles(days_back=request.days_back)
    
    # Get next edition number
    next_edition = storage.get_latest_edition_number() + 1
    
    # Prepare newsletter structure
    newsletter_data = {
        "edition_number": next_edition,
        "title": request.edition_title or f"Y2AI Weekly #{next_edition}",
        "generated_at": datetime.utcnow().isoformat(),
        "period_days": request.days_back,
        "article_count": sum(len(v) for v in articles_by_category.values()),
        "categories": {}
    }
    
    # Add top articles per category
    for category, articles in articles_by_category.items():
        newsletter_data["categories"][category] = {
            "count": len(articles),
            "top_articles": articles[:5],  # Top 5 by impact
            "key_facts": [],
            "dollar_amounts": []
        }
        
        # Extract key facts and amounts
        for article in articles[:5]:
            newsletter_data["categories"][category]["key_facts"].extend(
                article.get("extracted_facts", [])[:2]
            )
            newsletter_data["categories"][category]["dollar_amounts"].extend(
                article.get("dollar_amounts", [])
            )
    
    # Generate summary statistics
    all_sentiments = []
    all_companies = set()
    for articles in articles_by_category.values():
        for a in articles:
            all_sentiments.append(a.get("sentiment", "neutral"))
            all_companies.update(a.get("companies_mentioned", []))
    
    newsletter_data["summary"] = {
        "sentiment_distribution": {
            "bullish": all_sentiments.count("bullish"),
            "bearish": all_sentiments.count("bearish"),
            "neutral": all_sentiments.count("neutral")
        },
        "top_companies": list(all_companies)[:15],
        "category_ranking": sorted(
            newsletter_data["categories"].keys(),
            key=lambda c: newsletter_data["categories"][c]["count"],
            reverse=True
        )
    }
    
    return newsletter_data


@app.get("/newsletter/latest")
async def get_latest_newsletter():
    """Get the most recently published newsletter"""
    from .storage import get_storage
    
    storage = get_storage()
    
    if hasattr(storage, 'client') and storage.is_connected():
        result = storage.client.table("newsletter_editions")\
            .select("*")\
            .eq("status", "published")\
            .order("published_at", desc=True)\
            .limit(1)\
            .execute()
        
        if result.data:
            return result.data[0]
    
    return {"message": "No published newsletters found"}


# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.get("/health")
async def health_check():
    """Check system health"""
    from .storage import get_storage
    
    status = {
        "api": "healthy",
        "storage": "unknown",
        "newsapi": "not configured",
        "alphavantage": "not configured",
        "anthropic": "not configured"
    }
    
    # Check storage
    try:
        storage = get_storage()
        if hasattr(storage, 'is_connected') and storage.is_connected():
            status["storage"] = "connected (supabase)"
        else:
            status["storage"] = "connected (sqlite)"
    except:
        status["storage"] = "error"
    
    # Check API keys
    if os.getenv('NEWSAPI_KEY'):
        status["newsapi"] = "configured"
    if os.getenv('ALPHAVANTAGE_KEY'):
        status["alphavantage"] = "configured"
    if os.getenv('ANTHROPIC_API_KEY'):
        status["anthropic"] = "configured"
    
    return status


# =============================================================================
# RUN SERVER
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
