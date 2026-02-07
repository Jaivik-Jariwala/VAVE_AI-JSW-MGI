"""
Tools module for VAVE Agent - Executable functions for SQL, Vector Search, 
External APIs, and Calculations.

Note: This module uses SQLite. If your app uses PostgreSQL, you can modify
execute_sql_query to use psycopg2 instead of sqlite3.
"""
import sqlite3
import pandas as pd
import json
import logging
import re
import math
from typing import Callable, Optional, Dict, List, Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)


def execute_sql_query(query: str, db_path: str) -> List[Dict[str, Any]]:
    """
    Executes SQL and returns raw list of dicts.
    
    Args:
        query: SQL query string (should be SELECT only for safety)
        db_path: Path to the SQLite database file (or PostgreSQL connection function)
        
    Returns:
        Raw list of dictionaries representing the rows.
        If error occurs, returns list with single dict containing error.
    """
    # Basic safety check - only allow SELECT statements
    query_upper = query.strip().upper()
    if not query_upper.startswith('SELECT'):
        return [{"error": "Only SELECT queries allowed."}]
    
    # Additional safety: block dangerous SQL keywords
    dangerous_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE', 'TRUNCATE', 'EXEC', 'EXECUTE']
    if any(keyword in query_upper for keyword in dangerous_keywords):
        return [{"error": "Query contains prohibited SQL keywords. Only SELECT queries are allowed."}]
    
    conn = None
    try:
        # Check if db_path is a callable (PostgreSQL connection function)
        if callable(db_path):
            import psycopg2
            from psycopg2.extras import DictCursor
            conn = db_path()
            cursor = conn.cursor(cursor_factory=DictCursor)
            cursor.execute(query)
            rows = cursor.fetchall()
            results = [dict(row) for row in rows]
            cursor.close()
            conn.close()
        else:
            # SQLite path
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            results = [dict(row) for row in rows]
            conn.close()
        
        return results if results else []
        
    except sqlite3.Error as e:
        error_msg = f"SQL Error: {str(e)}"
        logger.error(f"{error_msg} - Query: {query}")
        # Detect dirty data / type issues and wrap in a friendly error descriptor
        if any(err_kw in str(e).lower() for err_kw in ["invalid input syntax for type", "could not convert", "datatype mismatch"]):
            return [{
                "error": "DATA_FORMAT_ERROR",
                "friendly_message": "Data format error detected in database. Please check logs."
            }]
        return [{"error": error_msg}]
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"{error_msg} - Query: {query}")
        if any(err_kw in str(e).lower() for err_kw in ["invalid input syntax for type", "could not convert", "datatype mismatch"]):
            return [{
                "error": "DATA_FORMAT_ERROR",
                "friendly_message": "Data format error detected in database. Please check logs."
            }]
        return [{"error": error_msg}]
    finally:
        if conn and not callable(db_path):
            conn.close()


def search_knowledge_base(query: str, vector_db_func: Callable) -> str:
    """
    Wrapper function for vector database search.
    
    Args:
        query: Natural language query string
        vector_db_func: Function that takes (query, top_k) and returns (table_data, context_str)
        
    Returns:
        JSON string containing search results and context
    """
    try:
        if vector_db_func is None:
            return json.dumps({
                "error": "Vector database function not available",
                "query": query
            })
        
        # Call the vector search function (from app.py's retrieve_context)
        table_data, context_str = vector_db_func(query, top_k=10)
        
        if not table_data:
            return json.dumps({
                "result": "No relevant ideas found in knowledge base",
                "row_count": 0,
                "query": query
            })
        
        # Format results
        results = []
        for item in table_data:
            result_item = {
                "idea_id": item.get("Idea Id", "N/A"),
                "cost_reduction_idea": item.get("Cost Reduction Idea", "N/A"),
                "saving_value_inr": item.get("Saving Value (INR)", "N/A"),
                "weight_saving_kg": item.get("Weight Saving (Kg)", "N/A"),
                "status": item.get("Status", "N/A"),
                "dept": item.get("Dept", "N/A"),
                "way_forward": item.get("Way Forward", "N/A")
            }
            results.append(result_item)
        
        return json.dumps({
            "result": results,
            "context": context_str,
            "format": "semantic_search",
            "row_count": len(results),
            "query": query
        })
        
    except Exception as e:
        error_msg = f"Vector search error: {str(e)}"
        logger.error(f"{error_msg} - Query: {query}")
        return json.dumps({
            "error": error_msg,
            "query": query
        })


def execute_calculation(expression: str, context_data: Optional[Dict] = None) -> str:
    """
    Execute mathematical calculations safely.
    
    Args:
        expression: Mathematical expression (e.g., "SUM(100, 200, 300)")
        context_data: Optional context data for variable substitution
        
    Returns:
        JSON string with calculation result
    """
    try:
        import re
        import math
        
        # Safety: Only allow safe mathematical operations
        allowed_chars = set('0123456789+-*/.()[], SUM AVG MAX MIN COUNT ')
        if not all(c in allowed_chars or c.isalpha() or c in ['_', ','] for c in expression):
            return json.dumps({
                "error": "Expression contains unsafe characters",
                "expression": expression
            })
        
        # Handle common aggregation functions
        if expression.upper().startswith('SUM'):
            # Extract numbers from SUM(100, 200, 300)
            numbers = re.findall(r'[\d.]+', expression)
            result = sum(float(n) for n in numbers)
            return json.dumps({
                "result": result,
                "expression": expression,
                "operation": "SUM"
            })
        elif expression.upper().startswith('AVG') or expression.upper().startswith('AVERAGE'):
            numbers = re.findall(r'[\d.]+', expression)
            if numbers:
                result = sum(float(n) for n in numbers) / len(numbers)
                return json.dumps({
                    "result": result,
                    "expression": expression,
                    "operation": "AVG"
                })
        elif expression.upper().startswith('MAX'):
            numbers = re.findall(r'[\d.]+', expression)
            if numbers:
                result = max(float(n) for n in numbers)
                return json.dumps({
                    "result": result,
                    "expression": expression,
                    "operation": "MAX"
                })
        elif expression.upper().startswith('MIN'):
            numbers = re.findall(r'[\d.]+', expression)
            if numbers:
                result = min(float(n) for n in numbers)
                return json.dumps({
                    "result": result,
                    "expression": expression,
                    "operation": "MIN"
                })
        
        # For now, do NOT support arbitrary expression evaluation to avoid security risks.
        # Encourage callers to use explicit aggregate functions instead.
        return json.dumps({
            "error": "Only simple aggregate expressions (SUM, AVG, MAX, MIN) are supported. "
                     "Arbitrary calculations are disabled for security reasons.",
            "expression": expression
        })
        
    except Exception as e:
        return json.dumps({
            "error": f"Calculation error: {str(e)}",
            "expression": expression
        })


def search_external_apis(query: str, api_type: str = "general") -> str:
    """
    Placeholder for external API calls (web search, part specs, cost data).
    
    Args:
        query: Search query
        api_type: Type of API ('web_search', 'part_specs', 'cost_data')
        
    Returns:
        JSON string with search results
    """
    try:
        # TODO: Integrate actual APIs like:
        # - Tavily API for web search
        # - SerpApi for Google Search
        # - Custom enterprise APIs for part specs
        
        logger.info(f"External API search requested: {api_type} - {query}")
        
        # Placeholder response
        return json.dumps({
            "result": f"External API search for '{query}' (type: {api_type}) - Not implemented yet",
            "api_type": api_type,
            "query": query,
            "note": "This is a placeholder. Integrate actual APIs as needed.",
            "suggestions": [
                "For web search: Use Tavily API or SerpApi",
                "For part specs: Integrate with your parts database API",
                "For cost data: Connect to pricing APIs or ERP systems"
            ]
        })
        
    except Exception as e:
        return json.dumps({
            "error": f"External API error: {str(e)}",
            "query": query,
            "api_type": api_type
        })


def validate_constraints(data: Dict, constraint_type: str, rules: Optional[Dict] = None) -> str:
    """
    Validate data against constraints (physics, regulations, business rules).
    
    Args:
        data: Data to validate
        constraint_type: Type of constraint ('physics', 'cost', 'regulations', 'business')
        rules: Optional custom rules dictionary
        
    Returns:
        JSON string with validation results
    """
    try:
        validation_results = {
            "passed": True,
            "constraint_type": constraint_type,
            "violations": [],
            "warnings": []
        }
        
        if constraint_type == "cost":
            # Example: Check if cost savings are realistic
            if "saving_value_inr" in data:
                raw_val = str(data.get("saving_value_inr", 0))
                clean_val = re.sub(r'[^\d.]', '', raw_val)
                saving = float(clean_val) if clean_val else 0.0
                if saving > 1000000:  # Flag unusually high savings
                    validation_results["warnings"].append(
                        f"Very high saving value: INR {saving:,.2f}. Please verify."
                    )
                if saving < 0:
                    validation_results["violations"].append(
                        "Negative saving value detected - this is invalid."
                    )
                    validation_results["passed"] = False
        
        elif constraint_type == "physics":
            # Example: Check if weight savings are physically possible
            if "weight_saving" in data:
                weight = float(data.get("weight_saving", 0))
                if weight > 1000:  # More than 1000kg seems unrealistic for a part
                    validation_results["warnings"].append(
                        f"Unusually high weight saving: {weight}kg. Please verify."
                    )
        
        elif constraint_type == "business":
            # Example: Check business rules
            if "status" in data and "saving_value_inr" in data:
                status = data.get("status", "")
                saving = float(data.get("saving_value_inr", 0))
                if status == "OK" and saving < 10:
                    validation_results["warnings"].append(
                        "Idea marked as OK but has very low savings. Consider review."
                    )
        
        if validation_results["violations"]:
            validation_results["passed"] = False
        
        return json.dumps(validation_results)
        
    except Exception as e:
        return json.dumps({
            "error": f"Validation error: {str(e)}",
            "constraint_type": constraint_type
        })


def rerank_results(results: List[Dict], query: str, top_k: int = 5) -> str:
    """
    Re-rank search results based on relevance to query.
    
    Args:
        results: List of result dictionaries
        query: Original query for relevance scoring
        top_k: Number of top results to return
        
    Returns:
        JSON string with re-ranked results
    """
    try:
        if not results:
            return json.dumps({
                "result": [],
                "row_count": 0,
                "query": query
            })
        
        import re
        
        # Simple keyword-based scoring (can be enhanced with embeddings)
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scored_results = []
        for item in results:
            score = 0
            text_fields = []
            
            # Collect all text fields
            for key, value in item.items():
                if isinstance(value, str):
                    text_fields.append(value.lower())
            
            text = " ".join(text_fields)
            
            # Score based on keyword matches
            for word in query_words:
                if word in text:
                    score += 1
            
            # Boost score for exact phrase matches
            if query_lower in text:
                score += 5
            
            scored_results.append({
                "item": item,
                "relevance_score": score
            })
        
        # Sort by score (descending)
        scored_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Return top_k results
        top_results = [r["item"] for r in scored_results[:top_k]]
        
        return json.dumps({
            "result": top_results,
            "row_count": len(top_results),
            "query": query,
            "reranked": True,
            "scores": [r["relevance_score"] for r in scored_results[:top_k]]
        })
        
    except Exception as e:
        return json.dumps({
            "error": f"Reranking error: {str(e)}",
            "query": query
        })

def perform_web_search(query: str, max_results: int = 5) -> str:
    """
    Robust Web Search designed for Free Tier limitations.
    Uses strict rate-limiting and User-Agent rotation to avoid 429 errors.
    """
    import time
    import random
    from duckduckgo_search import DDGS

    # 1. Sanitize Query
    if not query or not query.strip(): return ""
    clean_query = query.strip()[:400]

    # 2. Config for Free Tier (Slower but Safer)
    MAX_RETRIES = 5
    # Random wait between 3 to 8 seconds to mimic human behavior
    BASE_WAIT = random.uniform(3, 8) 
    
    logger.info(f"Web Search: '{clean_query[:40]}...' (Wait: {BASE_WAIT:.1f}s)")
    time.sleep(BASE_WAIT) 

    # 3. Execution Loop
    for attempt in range(MAX_RETRIES):
        try:
            # Re-initialize DDGS every time to get a fresh session
            with DDGS() as ddgs:
                # Use 'html' backend which is often more lenient than 'api'
                results_gen = ddgs.text(
                    clean_query, 
                    max_results=max_results, 
                    backend="html" 
                )
                results = list(results_gen) if results_gen else []

            if results:
                # Success! Format the output
                context_blob = ""
                for i, r in enumerate(results):
                    title = r.get('title', 'No Title')
                    body = r.get('body', 'No Description')
                    href = r.get('href', '#')
                    context_blob += f"[{i+1}] {title}: {body} (Source: {href})\n\n"
                
                return context_blob

        except Exception as e:
            wait_time = (2 ** attempt) * 5  # Exponential Backoff: 5s, 10s, 20s...
            logger.warning(f"Search Attempt {attempt+1} failed: {e}. Sleeping {wait_time}s...")
            time.sleep(wait_time)

    logger.error(f"All search attempts failed for: {clean_query}")
    return ""