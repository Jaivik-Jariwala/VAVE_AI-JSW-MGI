# VLM Engine Implementation Guide

## Overview

The VLM (Visual Language Model) Engine has been successfully implemented to automate image handling for the VAVE application. The engine handles images based on the "Idea Origin" and provides appropriate images for both current and proposal scenarios.

## Architecture

### Components

1. **`vlm_engine.py`**: Core VLM Engine class that handles all image logic
2. **`agent.py`**: Updated to integrate VLM Engine (already had partial integration)
3. **`app.py`**: Already configured to pass required dependencies to VAVEAgent
4. **`add_image_columns.sql`**: SQL migration script to add new image columns

## Image Handling Logic

### 1. Existing Database Ideas
- **Current Scenario Image**: Retrieved directly from database (`current_scenario_image` column)
- **Proposal Scenario Image**: Retrieved directly from database (`proposal_scenario_image` column)

### 2. AI Innovation Ideas
- **Current Scenario Image**: Found using semantic similarity search (FAISS + sentence model) to find the most similar existing idea's `current_scenario_image`
- **Proposal Scenario Image**: Generated using Hugging Face Inference API (Stable Diffusion XL) and saved to `static/generated/`

### 3. Web Source Ideas
- **Current Scenario Image**: Found using semantic similarity search to find the most similar existing idea's `current_scenario_image`
- **Proposal Scenario Image**: Scraped from the web source URL (og:image or first `<img>` tag) and saved to `static/generated/`

## Setup Instructions

### 1. Database Migration

Run the SQL migration script to add the new columns:

```bash
psql -U your_username -d your_database -f add_image_columns.sql
```

Or manually execute:
```sql
ALTER TABLE ideas ADD COLUMN IF NOT EXISTS current_scenario_image TEXT;
ALTER TABLE ideas ADD COLUMN IF NOT EXISTS proposal_scenario_image TEXT;
```

### 2. Environment Variables

Add the following to your `.env` file:

```env
# Hugging Face API Token for AI image generation (free tier available)
HUGGINGFACE_API_TOKEN=your_huggingface_token_here
```

**Getting a Hugging Face Token:**
1. Sign up at https://huggingface.co (free)
2. Go to Settings → Access Tokens
3. Create a new token with "Read" permissions
4. Copy the token to your `.env` file

**Note**: The free tier has rate limits but is sufficient for development and limited production use.

### 3. Directory Structure

The VLM engine automatically creates the following directory:
- `static/generated/` - Stores AI-generated and web-scraped images

Ensure this directory is writable by your application.

### 4. Dependencies

The following Python packages are required (should already be installed):
- `requests` - For API calls and web scraping
- `beautifulsoup4` (bs4) - For HTML parsing
- `Pillow` (PIL) - For image processing
- `sentence-transformers` - Already used for embeddings
- `faiss` - Already used for vector search
- `numpy` - Already used

Install if missing:
```bash
pip install requests beautifulsoup4 Pillow
```

## How It Works

### Integration Flow

1. **VAVEAgent Initialization** (in `app.py`):
   - `build_vector_db()` creates `faiss_index` and `embedding_model`
   - These are passed to `VAVEAgent` along with `get_db_connection` function
   - `VAVEAgent.__init__()` initializes `VLMEngine` if all dependencies are available

2. **Idea Generation** (in `agent.py`):
   - When ideas are generated via `run()`, they are enriched with images
   - `_enrich_ideas_with_images()` is called for each origin type
   - VLM Engine's `get_images_for_idea()` is called with idea text, origin, and context

3. **Image Retrieval/Generation** (in `vlm_engine.py`):
   - Based on origin, appropriate method is called
   - Images are returned as relative paths (e.g., `static/generated/image.png`)
   - Errors are handled gracefully (returns `None` if generation/scraping fails)

## API Usage

### Hugging Face Inference API

The engine uses the **free tier** of Hugging Face Inference API:
- Model: `stabilityai/stable-diffusion-xl-base-1.0`
- Endpoint: `https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0`
- Rate Limits: Free tier has limits but is usable for development

**Alternative Free Options** (if needed):
- Replicate API (has free tier)
- Stability AI API (has free credits)
- Local Stable Diffusion (if GPU available)

## Error Handling

The VLM engine is designed to fail gracefully:
- If image generation fails → Returns `None` (no image)
- If web scraping fails → Returns `None` (no image)
- If database query fails → Returns `None` (no image)
- Application continues to work even if VLM engine is unavailable

## Testing

To test the VLM engine:

1. **Test Existing Database Ideas**:
   - Ensure some ideas in database have `current_scenario_image` and `proposal_scenario_image` populated
   - Query existing ideas and verify images are returned

2. **Test AI Innovation Ideas**:
   - Generate new AI ideas
   - Check that `current_scenario_image` points to a similar existing image
   - Check that `proposal_scenario_image` points to a generated image in `static/generated/`

3. **Test Web Source Ideas**:
   - Generate ideas from web sources (with URLs in `way_forward`)
   - Check that `current_scenario_image` points to a similar existing image
   - Check that `proposal_scenario_image` points to a scraped image in `static/generated/`

## Troubleshooting

### VLM Engine Not Initializing
- Check that `db_conn`, `faiss_index`, and `sentence_model` are all provided to `VAVEAgent`
- Check application logs for initialization errors

### AI Image Generation Failing
- Verify `HUGGINGFACE_API_TOKEN` is set in `.env`
- Check Hugging Face API status
- Check rate limits (free tier has limits)
- Review logs for specific error messages

### Web Scraping Failing
- Some websites block automated scraping
- Check if URL is accessible
- Review logs for specific error messages
- The engine will gracefully return `None` if scraping fails

### Images Not Displaying
- Verify `static/generated/` directory exists and is writable
- Check file paths are correct (relative to application root)
- Verify web server can serve files from `static/` directory

## Future Enhancements

Potential improvements:
1. **Caching**: Cache generated/scraped images to avoid regeneration
2. **Image Optimization**: Compress/resize images before saving
3. **Multiple Image Sources**: Try multiple image sources if one fails
4. **Local Image Generation**: Use local Stable Diffusion if GPU available
5. **Image Validation**: Verify downloaded images are valid before saving

## Files Modified/Created

- ✅ `vlm_engine.py` - New file (VLM Engine implementation)
- ✅ `agent.py` - Already had VLM integration (verified working)
- ✅ `app.py` - Already configured correctly (no changes needed)
- ✅ `add_image_columns.sql` - New file (database migration)

## Summary

The VLM Engine is now fully integrated and ready to use. It automatically handles images for all three idea origin types:
- **Existing Database**: Uses images from database
- **AI Innovation**: Finds similar current image, generates proposal image
- **Web Source**: Finds similar current image, scrapes proposal image

The implementation is modular, error-tolerant, and uses free APIs where possible.

