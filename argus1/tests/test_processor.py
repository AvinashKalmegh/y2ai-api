"""
Unit Tests for Processor JSON Parsing and Data Validation

Tests cover:
- JSON extraction from various Claude response formats
- Validation and fixing of malformed results
- Category and sentiment normalization
- Edge cases in article processing
"""

import pytest
import json
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from argus1.processor_enhanced import (
    extract_json_from_text,
    validate_and_fix_result,
    Y2AI_CATEGORIES,
    VALID_CATEGORIES,
    VALID_SENTIMENTS,
)


# =============================================================================
# JSON EXTRACTION TESTS
# =============================================================================

class TestJSONExtraction:
    """Test JSON extraction from various response formats"""
    
    def test_clean_json(self):
        """Extract clean JSON"""
        text = '{"category": "spending", "impact_score": 0.8}'
        
        result = extract_json_from_text(text)
        
        assert result is not None
        assert result["category"] == "spending"
        assert result["impact_score"] == 0.8
    
    def test_json_with_whitespace(self):
        """Extract JSON with leading/trailing whitespace"""
        text = '''
        
        {"category": "data", "sentiment": "bullish"}
        
        '''
        
        result = extract_json_from_text(text)
        
        assert result is not None
        assert result["category"] == "data"
    
    def test_json_in_markdown_code_block(self):
        """Extract JSON from markdown code block"""
        text = '''Here's the analysis:

```json
{
    "category": "constraints",
    "impact_score": 0.7,
    "sentiment": "bearish"
}
```

That's my assessment.'''
        
        result = extract_json_from_text(text)
        
        assert result is not None
        assert result["category"] == "constraints"
        assert result["sentiment"] == "bearish"
    
    def test_json_in_plain_code_block(self):
        """Extract JSON from plain code block (no json label)"""
        text = '''Analysis:

```
{"category": "policy", "impact_score": 0.5}
```
'''
        
        result = extract_json_from_text(text)
        
        assert result is not None
        assert result["category"] == "policy"
    
    def test_json_with_surrounding_text(self):
        """Extract JSON from text with preamble and postamble"""
        text = '''Based on my analysis of this article, here is the structured extraction:

{"category": "energy", "impact_score": 0.9, "sentiment": "bullish"}

I hope this helps with your research.'''
        
        result = extract_json_from_text(text)
        
        assert result is not None
        assert result["category"] == "energy"
    
    def test_json_with_trailing_comma(self):
        """Handle JSON with trailing commas (common LLM error)"""
        text = '''```json
{
    "category": "adoption",
    "extracted_facts": ["fact1", "fact2",],
    "companies_mentioned": ["MSFT", "GOOGL",],
}
```'''
        
        result = extract_json_from_text(text)
        
        assert result is not None
        assert result["category"] == "adoption"
        assert "MSFT" in result["companies_mentioned"]
    
    def test_nested_json_objects(self):
        """Extract JSON with nested objects"""
        text = '''{"category": "data", "metadata": {"source": "reuters", "confidence": 0.9}}'''
        
        result = extract_json_from_text(text)
        
        assert result is not None
        assert result["category"] == "data"
        assert result["metadata"]["source"] == "reuters"
    
    def test_json_with_arrays(self):
        """Extract JSON with arrays"""
        text = '''{
            "category": "spending",
            "extracted_facts": ["MSFT spending $80B", "Google capex up 30%"],
            "dollar_amounts": ["$80 billion", "$30 billion"]
        }'''
        
        result = extract_json_from_text(text)
        
        assert result is not None
        assert len(result["extracted_facts"]) == 2
        assert "$80 billion" in result["dollar_amounts"]
    
    def test_empty_text(self):
        """Handle empty text"""
        result = extract_json_from_text("")
        
        assert result is None
    
    def test_none_text(self):
        """Handle None text"""
        result = extract_json_from_text(None)
        
        assert result is None
    
    def test_no_json_in_text(self):
        """Handle text with no JSON"""
        text = "This is just plain text with no JSON content at all."
        
        result = extract_json_from_text(text)
        
        assert result is None
    
    def test_malformed_json(self):
        """Handle malformed JSON gracefully"""
        text = '''{"category": "spending", "impact_score": }'''  # Missing value
        
        result = extract_json_from_text(text)
        
        # Should return None for truly malformed JSON
        assert result is None
    
    def test_multiple_json_objects(self):
        """Extract first JSON object when multiple present"""
        text = '''First: {"category": "spending"}
Second: {"category": "data"}'''
        
        result = extract_json_from_text(text)
        
        assert result is not None
        assert result["category"] == "spending"  # First one


# =============================================================================
# VALIDATION AND FIXING TESTS
# =============================================================================

class TestValidationAndFixing:
    """Test validation and auto-fixing of extracted results"""
    
    def test_valid_result_unchanged(self):
        """Valid result passes through unchanged"""
        result = {
            "category": "spending",
            "impact_score": 0.8,
            "sentiment": "bullish",
            "extracted_facts": ["fact1"],
            "companies_mentioned": ["NVDA"],
            "dollar_amounts": ["$80B"],
            "key_quotes": ["quote1"]
        }
        
        fixed = validate_and_fix_result(result)
        
        assert fixed["category"] == "spending"
        assert fixed["impact_score"] == 0.8
        assert fixed["sentiment"] == "bullish"
    
    def test_fixes_uppercase_category(self):
        """Fixes uppercase category"""
        result = {"category": "SPENDING", "sentiment": "bullish"}
        
        fixed = validate_and_fix_result(result)
        
        assert fixed["category"] == "spending"
    
    def test_fixes_category_with_extra_text(self):
        """Fixes category with extra descriptive text"""
        result = {"category": "spending (capital expenditure)"}
        
        fixed = validate_and_fix_result(result)
        
        assert fixed["category"] == "spending"
    
    def test_invalid_category_defaults_to_data(self):
        """Invalid category defaults to 'data'"""
        result = {"category": "unknown_category"}
        
        fixed = validate_and_fix_result(result)
        
        assert fixed["category"] == "data"
    
    def test_missing_category_defaults_to_data(self):
        """Missing category defaults to 'data'"""
        result = {}
        
        fixed = validate_and_fix_result(result)
        
        assert fixed["category"] == "data"
    
    def test_fixes_uppercase_sentiment(self):
        """Fixes uppercase sentiment"""
        result = {"category": "data", "sentiment": "BULLISH"}
        
        fixed = validate_and_fix_result(result)
        
        assert fixed["sentiment"] == "bullish"
    
    def test_invalid_sentiment_defaults_to_neutral(self):
        """Invalid sentiment defaults to 'neutral'"""
        result = {"category": "data", "sentiment": "very_positive"}
        
        fixed = validate_and_fix_result(result)
        
        assert fixed["sentiment"] == "neutral"
    
    def test_clamps_impact_score_high(self):
        """Clamps impact score > 1.0"""
        result = {"category": "data", "impact_score": 1.5}
        
        fixed = validate_and_fix_result(result)
        
        assert fixed["impact_score"] == 1.0
    
    def test_clamps_impact_score_low(self):
        """Clamps impact score < 0.0"""
        result = {"category": "data", "impact_score": -0.5}
        
        fixed = validate_and_fix_result(result)
        
        assert fixed["impact_score"] == 0.0
    
    def test_fixes_string_impact_score(self):
        """Converts string impact score to float"""
        result = {"category": "data", "impact_score": "0.75"}
        
        fixed = validate_and_fix_result(result)
        
        assert fixed["impact_score"] == 0.75
    
    def test_invalid_impact_score_defaults(self):
        """Invalid impact score defaults to 0.5"""
        result = {"category": "data", "impact_score": "high"}
        
        fixed = validate_and_fix_result(result)
        
        assert fixed["impact_score"] == 0.5
    
    def test_converts_string_to_list(self):
        """Converts string values to single-item lists"""
        result = {
            "category": "data",
            "extracted_facts": "Single fact as string",
            "companies_mentioned": "NVDA"
        }
        
        fixed = validate_and_fix_result(result)
        
        assert fixed["extracted_facts"] == ["Single fact as string"]
        assert fixed["companies_mentioned"] == ["NVDA"]
    
    def test_converts_none_to_empty_list(self):
        """Converts None to empty list"""
        result = {
            "category": "data",
            "extracted_facts": None,
            "companies_mentioned": None
        }
        
        fixed = validate_and_fix_result(result)
        
        assert fixed["extracted_facts"] == []
        assert fixed["companies_mentioned"] == []
    
    def test_ensures_all_list_fields(self):
        """Ensures all list fields are present"""
        result = {"category": "data"}
        
        fixed = validate_and_fix_result(result)
        
        assert fixed["extracted_facts"] == []
        assert fixed["companies_mentioned"] == []
        assert fixed["dollar_amounts"] == []
        assert fixed["key_quotes"] == []


# =============================================================================
# CATEGORY MATCHING TESTS
# =============================================================================

class TestCategoryMatching:
    """Test partial category name matching"""
    
    def test_all_valid_categories(self):
        """All defined categories are valid"""
        for category in Y2AI_CATEGORIES.keys():
            result = {"category": category}
            fixed = validate_and_fix_result(result)
            assert fixed["category"] == category
    
    def test_partial_match_spending(self):
        """Partial match for 'spending'"""
        result = {"category": "cap spending"}
        fixed = validate_and_fix_result(result)
        assert fixed["category"] == "spending"
    
    def test_partial_match_constraints(self):
        """Partial match for 'constraints'"""
        result = {"category": "supply constraints"}
        fixed = validate_and_fix_result(result)
        assert fixed["category"] == "constraints"
    
    def test_partial_match_china(self):
        """Partial match for 'china'"""
        result = {"category": "china_related"}
        fixed = validate_and_fix_result(result)
        assert fixed["category"] == "china"


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases in processing"""
    
    def test_unicode_content(self):
        """Handle Unicode in JSON"""
        text = '{"category": "spending", "extracted_facts": ["Microsoft announced €50 billion investment"]}'
        
        result = extract_json_from_text(text)
        
        assert result is not None
        assert "€50 billion" in result["extracted_facts"][0]
    
    def test_escaped_quotes(self):
        """Handle escaped quotes in JSON"""
        text = '{"category": "data", "key_quotes": ["CEO said \\"AI is transformative\\""]}'
        
        result = extract_json_from_text(text)
        
        assert result is not None
        assert "transformative" in result["key_quotes"][0]
    
    def test_newlines_in_values(self):
        """Handle newlines in JSON values"""
        text = '''{
            "category": "spending",
            "extracted_facts": ["First line\\nSecond line"]
        }'''
        
        result = extract_json_from_text(text)
        
        assert result is not None
    
    def test_very_long_content(self):
        """Handle very long JSON content"""
        long_fact = "A" * 10000
        text = f'{{"category": "data", "extracted_facts": ["{long_fact}"]}}'
        
        result = extract_json_from_text(text)
        
        assert result is not None
        assert len(result["extracted_facts"][0]) == 10000
    
    def test_empty_arrays(self):
        """Handle empty arrays"""
        text = '{"category": "data", "extracted_facts": [], "companies_mentioned": []}'
        
        result = extract_json_from_text(text)
        
        assert result is not None
        assert result["extracted_facts"] == []
    
    def test_numeric_string_in_array(self):
        """Handle numeric strings in arrays"""
        text = '{"category": "data", "dollar_amounts": ["$80B", "100 million", "$50.5 billion"]}'
        
        result = extract_json_from_text(text)
        
        assert result is not None
        assert len(result["dollar_amounts"]) == 3
    
    def test_boolean_values(self):
        """Handle boolean values (should be ignored/converted)"""
        text = '{"category": "data", "is_important": true, "impact_score": 0.8}'
        
        result = extract_json_from_text(text)
        
        assert result is not None
        assert result["is_important"] is True  # Preserved but unused
    
    def test_null_values(self):
        """Handle null values"""
        text = '{"category": "spending", "impact_score": null}'
        
        result = extract_json_from_text(text)
        fixed = validate_and_fix_result(result)
        
        assert fixed is not None
        assert fixed["impact_score"] == 0.5  # Default


# =============================================================================
# INTEGRATION-STYLE TESTS
# =============================================================================

class TestProcessorIntegration:
    """Integration-style tests simulating real Claude responses"""
    
    def test_typical_claude_response(self):
        """Parse typical Claude response format"""
        text = '''Based on my analysis of this article about Microsoft's AI infrastructure investment, here is the structured extraction:

```json
{
    "category": "spending",
    "extracted_facts": [
        "Microsoft announced $80 billion AI infrastructure investment",
        "Investment spans data centers across 15 countries",
        "Project timeline extends through 2027"
    ],
    "impact_score": 0.9,
    "sentiment": "bullish",
    "companies_mentioned": ["Microsoft", "NVIDIA", "AMD"],
    "dollar_amounts": ["$80 billion"],
    "key_quotes": ["This represents our largest infrastructure commitment in company history"]
}
```

This article is highly relevant to the Y2AI infrastructure thesis as it demonstrates continued hyperscaler commitment to AI capacity expansion.'''
        
        result = extract_json_from_text(text)
        fixed = validate_and_fix_result(result)
        
        assert fixed["category"] == "spending"
        assert fixed["impact_score"] == 0.9
        assert len(fixed["extracted_facts"]) == 3
        assert "Microsoft" in fixed["companies_mentioned"]
    
    def test_minimal_claude_response(self):
        """Parse minimal Claude response"""
        text = '{"category":"data","impact_score":0.5,"sentiment":"neutral","extracted_facts":[],"companies_mentioned":[],"dollar_amounts":[],"key_quotes":[]}'
        
        result = extract_json_from_text(text)
        fixed = validate_and_fix_result(result)
        
        assert fixed["category"] == "data"
        assert fixed["sentiment"] == "neutral"
    
    def test_verbose_claude_response(self):
        """Parse verbose Claude response with extra fields"""
        text = '''```json
{
    "category": "skepticism",
    "extracted_facts": ["Analysts warn of AI bubble", "Valuations exceed fundamentals"],
    "impact_score": 0.85,
    "sentiment": "bearish",
    "companies_mentioned": ["NVDA", "MSFT", "GOOGL"],
    "dollar_amounts": [],
    "key_quotes": ["The emperor has no clothes"],
    "reasoning": "This article presents a contrarian view on AI valuations",
    "confidence": "high",
    "additional_context": "Published in major financial publication"
}
```'''
        
        result = extract_json_from_text(text)
        fixed = validate_and_fix_result(result)
        
        assert fixed["category"] == "skepticism"
        assert fixed["sentiment"] == "bearish"
        # Extra fields are preserved
        assert result["reasoning"] == "This article presents a contrarian view on AI valuations"


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
