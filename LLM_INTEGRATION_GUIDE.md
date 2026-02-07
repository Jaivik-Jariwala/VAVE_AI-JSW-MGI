# LLM Integration Guide for VAVE Presentation Engine

This guide shows you exactly how to replace the mock LLM functions with real API calls.

## Quick Start: Replace Mock Functions

The `LLMEnrichmentEngine` class has three main methods that need LLM integration:

1. `enrich_idea()` - Expands raw ideas into detailed engineering content
2. `generate_feasibility_matrix()` - Creates feasibility analysis
3. `generate_executive_summary()` - Generates professional narrative
4. `analyze_supply_chain_impact()` - Analyzes supply chain implications

## Method 1: OpenAI GPT-4 Integration

### Step 1: Install OpenAI SDK

```bash
pip install openai
```

### Step 2: Update `enrich_idea()` Method

```python
import openai
from typing import Dict, Any

class LLMEnrichmentEngine:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        openai.api_key = self.api_key
    
    def enrich_idea(self, raw_idea: Dict[str, Any]) -> Dict[str, Any]:
        """Replace mock with real OpenAI API call."""
        
        prompt = f"""
        Act as a Senior VAVE Engineer at a top automotive OEM. 
        I have a raw cost-reduction idea: {raw_idea.get('raw_description', '')}
        
        Please expand this into a detailed engineering proposal for a feasibility meeting. 
        Output the following sections in JSON format:
        
        {{
            "engineering_logic": "Explain technically why this works (mention physics, material properties, or manufacturing processes)",
            "critical_risks": ["Risk 1", "Risk 2", "Risk 3"],
            "validation_plan": ["Test 1", "Test 2", "Test 3"],
            "strategic_pitch": "One-sentence pitch to CEO about why this is smart business move"
        }}
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a Senior VAVE Engineer with 20+ years of automotive industry experience."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            # Parse JSON response
            import json
            content = response.choices[0].message.content
            # Extract JSON from markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            enriched = json.loads(content)
            return enriched
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            # Fallback to mock data
            return self._mock_enrich_idea(raw_idea)
```

### Step 3: Update `generate_feasibility_matrix()` Method

```python
def generate_feasibility_matrix(self, ideas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Replace mock with real OpenAI API call."""
    
    ideas_list = "\n".join([
        f"{i+1}. {idea.get('title', '')} - {idea.get('raw_description', '')}"
        for i, idea in enumerate(ideas)
    ])
    
    prompt = f"""
    Act as an external manufacturing consultant. Review these {len(ideas)} ideas:
    
    {ideas_list}
    
    Create a feasibility analysis table in JSON format with this structure:
    [
        {{
            "idea_title": "Idea Title",
            "ease_of_implementation": 8,
            "capex_impact": "Low/Medium/High",
            "time_to_market": "Immediate/6 Months/2 Years",
            "verdict": "Quick Win/Strategic Bet/Kill",
            "saving_amount": 100.0
        }}
    ]
    
    Be harsh and realistic. Rate ease of implementation 1-10 (10 = No tooling change, 1 = New factory required).
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a manufacturing consultant with expertise in automotive production."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=2000
        )
        
        content = response.choices[0].message.content
        # Extract JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        matrix = json.loads(content)
        
        # Merge with original idea saving amounts
        for item in matrix:
            matching_idea = next(
                (idea for idea in ideas if idea.get('title') == item.get('idea_title')),
                None
            )
            if matching_idea:
                item['saving_amount'] = matching_idea.get('saving_amount', 0)
        
        return matrix
        
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return self._mock_feasibility_matrix(ideas)
```

## Method 2: Google Gemini Integration

### Step 1: Install Gemini SDK

```bash
pip install google-generativeai
```

### Step 2: Update Methods

```python
import google.generativeai as genai
import os

class LLMEnrichmentEngine:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    def enrich_idea(self, raw_idea: Dict[str, Any]) -> Dict[str, Any]:
        """Replace mock with real Gemini API call."""
        
        prompt = f"""
        Act as a Senior VAVE Engineer at a top automotive OEM. 
        I have a raw cost-reduction idea: {raw_idea.get('raw_description', '')}
        
        Please expand this into a detailed engineering proposal. Output JSON:
        {{
            "engineering_logic": "...",
            "critical_risks": ["...", "...", "..."],
            "validation_plan": ["...", "...", "..."],
            "strategic_pitch": "..."
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            content = response.text
            
            # Extract JSON
            import json
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                enriched = json.loads(json_match.group())
                return enriched
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return self._mock_enrich_idea(raw_idea)
```

## Method 3: Using Your Existing LLM Setup

If you're already using an LLM in your `app.py`, you can reuse that:

```python
from app import generate_response  # Your existing LLM function

class LLMEnrichmentEngine:
    def enrich_idea(self, raw_idea: Dict[str, Any]) -> Dict[str, Any]:
        """Use your existing LLM infrastructure."""
        
        query = f"""
        Expand this VAVE idea into engineering proposal:
        {raw_idea.get('raw_description', '')}
        
        Provide: engineering_logic, critical_risks, validation_plan, strategic_pitch
        """
        
        # Use your existing LLM function
        response = generate_response(
            username="system",
            user_query=query
        )
        
        # Parse response and structure it
        # (Adjust based on your LLM response format)
        return self._parse_response(response)
```

## Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=sk-your-key-here
# OR
GEMINI_API_KEY=your-key-here
```

Load in your code:

```python
from dotenv import load_dotenv
load_dotenv()

# Then use:
api_key = os.getenv('OPENAI_API_KEY')
```

## Error Handling Best Practices

Always include fallback to mock data:

```python
def enrich_idea(self, raw_idea: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # Real LLM call
        return self._call_llm_api(raw_idea)
    except Exception as e:
        logger.warning(f"LLM API failed, using mock data: {e}")
        return self._mock_enrich_idea(raw_idea)
```

## Cost Optimization Tips

1. **Cache Results**: Store enriched ideas in a database to avoid re-processing
2. **Batch Processing**: Process multiple ideas in one API call when possible
3. **Use Smaller Models**: GPT-3.5-turbo is cheaper than GPT-4 for some tasks
4. **Rate Limiting**: Implement rate limiting to avoid API quota issues

## Testing

Test your LLM integration:

```python
from vave_presentation_engine import LLMEnrichmentEngine

engine = LLMEnrichmentEngine()
test_idea = {
    "title": "Test Idea",
    "raw_description": "Replace steel with aluminum",
    "saving_amount": 100.0
}

enriched = engine.enrich_idea(test_idea)
print(enriched)
```

## Next Steps

1. Replace mock functions one at a time
2. Test each function individually
3. Monitor API costs and response times
4. Add caching for frequently used ideas
5. Implement retry logic for API failures

