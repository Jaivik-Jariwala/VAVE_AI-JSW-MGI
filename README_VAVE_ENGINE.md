# VAVE Presentation Generator Engine

A professional PowerPoint presentation generator for cost-reduction engineering ideas with LLM enrichment capabilities.

## Features

- **LLM Enrichment Layer**: Expands raw engineering ideas into detailed feasibility studies
- **Boardroom-Ready Output**: Professional PowerPoint presentations with proper formatting
- **Modular Architecture**: Object-oriented design with clear separation of concerns
- **Comprehensive Analysis**: Includes engineering logic, risk assessment, validation plans, and strategic pitches

## Installation

```bash
pip install python-pptx
```

## Usage

### Basic Usage

```bash
python vave_presentation_engine.py --input data.json --output presentation.pptx
```

### Programmatic Usage

```python
from vave_presentation_engine import VAVEPresentation
import json

# Load data
with open('data.json', 'r') as f:
    data = json.load(f)

# Generate presentation
generator = VAVEPresentation(
    project_metadata=data['project_metadata'],
    ideas=data['ideas']
)
presentation = generator.generate()
generator.save('output.pptx')
```

## Integrating Real LLM APIs

### OpenAI Integration

Replace the mock methods in `LLMEnrichmentEngine` with:

```python
import openai

def enrich_idea(self, raw_idea: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""
    Act as a Senior VAVE Engineer at a top automotive OEM. 
    I have a raw cost-reduction idea: {raw_idea['raw_description']}
    
    Please expand this into a detailed engineering proposal...
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    
    # Parse response and return structured data
    return self._parse_llm_response(response.choices[0].message.content)
```

### Google Gemini Integration

```python
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel('gemini-pro')

def enrich_idea(self, raw_idea: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""
    Act as a Senior VAVE Engineer...
    {raw_idea['raw_description']}
    """
    
    response = model.generate_content(prompt)
    
    # Parse and return structured data
    return self._parse_llm_response(response.text)
```

## Presentation Structure

1. **Title Slide**: Project name, presenter, date
2. **Executive Summary**: Total savings, achievement percentage, narrative
3. **Methodology**: AI-driven VAVE pipeline explanation
4. **Deep Dive Slides**: One per idea with:
   - Engineering rationale
   - Validation roadmap
   - Risk mitigation
   - Strategic CEO pitch
5. **Feasibility Matrix**: Kill/Keep analysis table
6. **Supply Chain Analysis**: Vendor and sourcing implications
7. **Roadmap**: Implementation phases and next steps

## Customization

### Colors

Modify the color scheme in `VAVEPresentation.__init__()`:

```python
self.primary_color = RGBColor(0, 51, 102)  # Deep blue
self.accent_color = RGBColor(0, 102, 204)  # Bright blue
self.success_color = RGBColor(0, 153, 76)   # Green
```

### Layouts

Adjust slide layouts by changing the layout index:

```python
slide = self.prs.slides.add_slide(self.prs.slide_layouts[1])  # Title and content
```

Available layouts:
- `0`: Title slide
- `1`: Title and content
- `6`: Blank (for custom layouts)

## Data Format

The input JSON should follow this structure:

```json
{
  "project_metadata": {
    "project_name": "Project Name",
    "car_model": "Vehicle Model",
    "target_savings": 50000,
    "presenter": "Presenter Name"
  },
  "ideas": [
    {
      "title": "Idea Title",
      "raw_description": "Raw description of the idea",
      "saving_amount": 100.0,
      "origin": "Existing DB or AI Generated",
      "idea_id": "ID-001"
    }
  ]
}
```

## License

MIT License - Feel free to use and modify for your projects.

