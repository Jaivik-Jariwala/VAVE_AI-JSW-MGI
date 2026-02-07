"""
Test script for VAVE Presentation Generator Engine
Run this to verify the engine works correctly.
"""

from vave_presentation_engine import VAVEPresentation
import json
from pathlib import Path

def test_presentation_generation():
    """Test the presentation generation with sample data."""
    
    # Load sample data
    data_file = Path('data.json')
    if not data_file.exists():
        print("❌ Error: data.json not found. Please create it first.")
        return False
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    project_metadata = data.get('project_metadata', {})
    ideas = data.get('ideas', [])
    
    if not ideas:
        print("❌ Error: No ideas found in data.json")
        return False
    
    print(f"📊 Generating presentation for {len(ideas)} ideas...")
    print(f"   Project: {project_metadata.get('car_model', 'Unknown')}")
    print(f"   Target Savings: {project_metadata.get('target_savings', 0):,.0f}")
    
    # Generate presentation
    try:
        generator = VAVEPresentation(project_metadata, ideas)
        presentation = generator.generate()
        
        # Save to file
        output_file = 'test_output.pptx'
        generator.save(output_file)
        
        print(f"\n✅ Success! Presentation generated:")
        print(f"   File: {output_file}")
        print(f"   Slides: {len(presentation.slides)}")
        print(f"   Ideas analyzed: {len(ideas)}")
        
        # Calculate total savings
        total_savings = sum(idea.get('saving_amount', 0) for idea in ideas)
        print(f"   Total Savings: {total_savings:,.0f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error generating presentation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("=" * 60)
    print("VAVE Presentation Generator - Test Script")
    print("=" * 60)
    print()
    
    success = test_presentation_generation()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ Test completed successfully!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Test failed. Please check the errors above.")
        print("=" * 60)

