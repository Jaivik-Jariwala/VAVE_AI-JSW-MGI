# VAVE AI System - Comprehensive System Design Report

## Executive Summary

The VAVE (Value Analysis Value Engineering) AI System is an enterprise-grade automotive cost reduction platform that combines **Retrieval-Augmented Generation (RAG)**, **Multi-Agent AI Orchestration**, **Physics-Informed Engineering Constraints**, and **Visual Language Model (VLM) Integration** to generate, validate, and present cost reduction ideas for automotive components.

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [AI Agent Methodology](#2-ai-agent-methodology)
3. [Component Architecture](#3-component-architecture)
4. [Data Flow: User Prompt to Output](#4-data-flow-user-prompt-to-output)
5. [Technology Stack](#5-technology-stack)
6. [System Design Patterns](#6-system-design-patterns)
7. [Database Architecture](#7-database-architecture)
8. [API Endpoints & Routes](#8-api-endpoints--routes)
9. [Security & Authentication](#9-security--authentication)
10. [Performance & Scalability](#10-performance--scalability)

---

## 1. System Architecture Overview

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND LAYER                           │
│  (Flask Templates + JavaScript + HTML/CSS)                      │
└────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FLASK APPLICATION LAYER                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Routes     │  │  Middleware  │  │  Auth/Login  │         │
│  │  /chat       │  │  CORS        │  │  Flask-Login │         │
│  │  /generate_* │  │  Logging     │  │  Role-Based  │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      AI AGENT ORCHESTRATION LAYER               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              VAVEAgent (Core Orchestrator)              │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │   │
│  │  │ Innovation   │  │ Web Mining   │  │ DB Retrieval  │  │   │
│  │  │ Engine       │  │ Engine       │  │ (RAG)         │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ LLM Layer    │    │ VLM Engine   │    │ Vector DB    │
│ (Gemini API) │    │ (Image Gen)  │    │ (FAISS)      │
└──────────────┘    └──────────────┘    └──────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DATA PERSISTENCE LAYER                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ PostgreSQL   │  │ SQLite       │  │ DuckDB       │         │
│  │ (Ideas DB)   │  │ (Users DB)   │  │ (Data Lake)  │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Core Design Principles

1. **Multi-Stream Processing**: Three parallel streams (DB RAG, AI Innovation, Web Mining) ensure comprehensive idea generation
2. **Physics-Informed Constraints**: Dynamic Vehicle×Component matrix prevents physically impossible suggestions
3. **Autonomous Validation**: Multi-criteria scoring (feasibility, cost, weight, homologation) with automatic filtering
4. **Semantic Visual Matching**: Cosine similarity-based image retrieval for context-aware visualizations
5. **Modular Agent Architecture**: Each agent component is independently testable and replaceable

---

## 2. AI Agent Methodology

### 2.1 VAVEAgent: The Core Orchestrator

**Location**: `agent.py` - `VAVEAgent` class

**Purpose**: Coordinates three parallel idea generation streams and applies physics-informed validation.

#### 2.1.1 Agent Initialization

```python
VAVEAgent(
    db_path: str,                    # Database path (legacy SQLite reference)
    vector_db_func: Callable,        # RAG retrieval function
    llm_client=None,                 # Optional LLM client (uses Gemini by default)
    pg_conn_func=None,               # PostgreSQL connection factory
    db_conn=None,                    # DB connection for VLM
    faiss_index=None,                # FAISS vector index
    sentence_model=None              # SentenceTransformer for embeddings
)
```

**Initialization Flow**:
1. Loads image index from `static/image_captions.json`
2. Builds semantic embeddings for all image captions (normalized cosine similarity)
3. Initializes VLM Engine if dependencies are available
4. Sets up vehicle/component knowledge matrices

#### 2.1.2 The "Matrix Logic" - Physics-Informed Context Resolution

**Key Innovation**: Instead of hard-coding vehicle constraints, the system uses a **Vehicle×Component Matrix**:

**Vehicle DB (Axis X - Physics)**:
- `hector`: 1650kg, C-SUV, ICE, 36k/year
- `astor`: 1350kg, B-SUV, ICE, 40k/year
- `zsev`: 1620kg, B-SUV, EV, 15k/year
- `comet`: 850kg, GSEV, EV, 12k/year
- `nexon`: 1400kg, B-SUV, Benchmark

**Component DB (Axis Y - Function)**:
- `brake`: Thermal mass, friction, FMVSS 105
- `hvac`: CFM, power consumption, defrost standards
- `body`: Crash safety, torsional stiffness, pedestrian norms
- `general`: Packaging, DVPR, manufacturability

**Context Resolution Method**: `_resolve_engineering_context(query: str)`

1. **Vehicle Detection**: Pattern matching against aliases (e.g., "MG Hector" → `hector`)
2. **Component Detection**: Keyword matching (e.g., "brake disc" → `brake`)
3. **Special Rules**: EV×HVAC triggers range impact constraints
4. **Output**: Formatted engineering context string injected into LLM system prompts

**Example**:
```
Query: "Reduce weight of MG Hector Brake Disc"
→ Vehicle: hector (1650kg, C-SUV)
→ Component: brake (thermal mass critical)
→ Context: "Heavy-vehicle brake thermal risk. Avoid rotor thinning unless CFD validated."
```

### 2.2 Three-Stream Idea Generation Pipeline

#### Stream A: Existing Database (RAG)

**Method**: `run()` → `vector_db_func(user_query, top_k=5)`

**Process**:
1. Semantic search via FAISS using query embeddings
2. Retrieves top 5 existing ideas from PostgreSQL
3. Enriches with images via semantic matching
4. Returns as "Existing DB" origin

**Technology**: 
- **FAISS** (Facebook AI Similarity Search) for vector similarity
- **SentenceTransformer** (`all-MiniLM-L6-v2` or similar) for embeddings
- **PostgreSQL** for structured idea storage

#### Stream B: AI Innovation Engine

**Method**: `run_innovation_engine(query, context_ideas, used_images)`

**Process**:
1. **Target Component Extraction**: `_smart_extract_target()` removes stopwords, extracts meaningful component names
2. **Dynamic LLM Generation**: `_generate_with_llm()` 
   - Injects Vehicle×Component context into system prompt
   - Requests 12 ideas (high volume for filtering)
   - Uses Gemini API with temperature=0.7
3. **Autonomous Validation**: `_validate_and_filter_ideas()`
   - Normalizes scores (0-100)
   - Applies physics heuristics (heavy vehicle brake risk, EV HVAC range impact)
   - Filters by minimum score threshold (25/100)
   - Marks as "Auto-Approved" or "Needs Human Review"
4. **Image Assignment**: Semantic visual matching with uniqueness enforcement
5. **Returns**: Up to 8 validated ideas

**LLM Prompt Structure**:
```
System: [Vehicle×Component Context] + Engineering Instructions
User: "Construct 12 NEW proposals for {target} on {vehicle_name}"
Output: JSON array with 12 ideas, each with:
  - cost_reduction_idea
  - way_forward
  - homologation_theory
  - visual_prompt
  - feasibility_score, cost_saving_score, weight_reduction_score, homologation_feasibility_score
```

#### Stream C: Web Mining Engine

**Method**: `run_web_mining_engine(query, target_component)`

**Process**:
1. **Academic Query Generation**: 5 specialized search queries
   - PDF-focused: `automotive {component} cost reduction filetype:pdf`
   - SAE.org: `site:sae.org {component} lightweighting {vehicle}`
   - Industry trends: `automotive {component} material substitution 2024 2025`
2. **Parallel Web Search**: 3 workers, 3 results per query (DuckDuckGo via `duckduckgo_search`)
3. **Physics-Adapted Synthesis**: 
   - Injects Vehicle×Component context
   - Instructs LLM to **adapt** generic research to specific vehicle constraints
   - Example: "Plastic pedal" → Rejected (ROI fail for 36k volume)
   - Example: "Composite reinforcement" → Accepted (material tech transfer)
4. **Returns**: 4 adapted research proposals

**Key Innovation**: Instead of rejecting generic research, the system **adapts** it to vehicle-specific constraints.

### 2.3 Validation & Filtering Engine

**Method**: `_validate_and_filter_ideas(ideas, target_component, query, min_score=25)`

**Multi-Layer Validation**:

1. **Score Normalization**:
   - Coerces all scores to integers (0-100)
   - Handles missing/invalid values gracefully

2. **Scope Validation**:
   - Tokenizes target component
   - Ensures idea text contains at least one target token
   - Rejects out-of-scope ideas

3. **Physical Impossibility Detection**:
   - Regex patterns: `100% weight reduction`, `zero cost manufacturing`, `no testing required`
   - Hard rejection for impossible claims

4. **Physics-Informed Heuristics**:
   - **Heavy Vehicle Brake Risk**: If `vehicle_weight >= 1600kg` AND `component == brake` AND idea mentions "reduce weight/thin rotor/disc" → Flag as "Needs Human Review" with thermal mass warning
   - **EV×HVAC Range Impact**: If `vehicle_type == EV` AND `component == hvac` AND idea mentions "increase blower power/watts" → Flag with range impact warning

5. **Scoring Gate**:
   - Minimum score across all 4 categories must be >= 25
   - Ideas below threshold go to "Needs Human Review" (not discarded)
   - Ideas above threshold + no physics flags → "Auto-Approved"

6. **Output**: Merged list (Auto-Approved first, then Needs Review), capped at 8 ideas

### 2.4 Semantic Visual Matching

**Method**: `_evaluate_visual_match(query_text, folder_name, used_images, threshold=0.35)`

**Process**:
1. Encodes query text using SentenceTransformer
2. Computes cosine similarity against all pre-computed image caption embeddings
3. Sorts by similarity score (descending)
4. Returns first match above threshold (0.35) that hasn't been used
5. Falls back to keyword matching if embeddings unavailable

**Image Index Structure** (`static/image_captions.json`):
```json
{
  "85.jpg": "Close up of black car rear bumper with parking sensors",
  "120.jpg": "Silver alloy wheel rim showing brake caliper",
  ...
}
```

**Uniqueness Enforcement**: Tracks `used_images` set across all ideas to prevent duplicate image assignments.

---

## 3. Component Architecture

### 3.1 Flask Application (`app.py`)

**Size**: ~5053 lines

**Key Responsibilities**:
- HTTP request handling
- User authentication & authorization
- Database connection management
- Model initialization (GPT-2, BLIP, SentenceTransformer)
- Vector database building (FAISS)
- Route definitions

**Critical Functions**:

1. **`build_vector_db()`** (Line ~1155):
   - Loads ideas from PostgreSQL
   - Generates embeddings using SentenceTransformer
   - Builds FAISS index (L2 distance)
   - Saves index to disk for persistence

2. **`retrieve_context(query, top_k=10)`**:
   - Encodes query
   - Searches FAISS index
   - Retrieves matching rows from PostgreSQL
   - Returns (table_data, context_string)

3. **`agentic_rag_chat(username, user_query, image_path)`** (Line ~4813):
   - Initializes VAVEAgent (lazy loading)
   - Calls `agent.run(user_query)`
   - Formats response for frontend
   - Logs events to PostgreSQL `events` table

4. **`generate_response(username, user_query)`** (Legacy fallback):
   - Uses GPT-2 for text generation
   - Vector search for context
   - Returns formatted response

### 3.2 VLM Engine (`vlm_engine.py`)

**Purpose**: Generates and retrieves images for ideas

**Key Methods**:

1. **`get_images_for_idea(idea_text, origin, extra_context)`**:
   - Resolves current scenario image (from agent or generates)
   - Resolves competitor image (from agent or generates)
   - Generates proposal image:
     - **AI Innovation**: Uses `_construct_overlay_prompt()` → CAD-style overlay via Pollinations API
     - **DB/Web**: Uses `_create_engineering_annotation()` → PIL-based overlay on base image

2. **`_construct_overlay_prompt(idea_text, vehicle_name, component_hint)`**:
   - Builds prompt: "Technical engineering CAD overlay of {component} for {vehicle}, highlighting {change}. Schematic wireframe style, neon blue highlights, white background"
   - Used for AI-generated ideas to simulate engineering modifications

3. **`_create_engineering_annotation(base_image_path, idea_text)`**:
   - Opens base image (PIL)
   - Extracts numbers from idea text (regex)
   - Draws "Before → After" annotations
   - Adds measurement lines, reduction badges
   - Saves to `static/generated/`

4. **`_generate_cloud_image_pollinations(prompt_text)`**:
   - Calls Pollinations.ai API
   - Caches generated images
   - Returns relative path

### 3.3 Tools Module (`tools.py`)

**Purpose**: Executable functions for agent tool-calling

**Functions**:
- `execute_sql_query()`: Safe SQL execution (SELECT only)
- `search_knowledge_base()`: Vector search wrapper
- `execute_calculation()`: Math operations (SUM, AVG, MAX, MIN)
- `validate_constraints()`: Physics/cost/regulatory validation
- `perform_web_search()`: DuckDuckGo search with rate limiting

### 3.4 Data Processor (`data_processor.py`)

**Purpose**: Knowledge base upload processing

**Key Function**: `process_knowledge_base_upload(excel_path, zip_path, db_config)`

**Process**:
1. Extracts image ZIP to temp directory
2. Locates `proposal/`, `mg/`, `competitor/` folders recursively
3. Reads Excel file (`AIML Dummy ideas` sheet)
4. Maps Excel columns to database schema
5. Inserts ideas into PostgreSQL with image blobs (BYTEA)
6. Cleans up temp files

### 3.5 Excel Generator Engine (`excel_generator_engine.py`)

**Purpose**: Generates formatted Excel files with embedded images

**Key Function**: `generate_excel_from_table(table_data, output_folder)`

**Process**:
1. Creates OpenPyXL workbook
2. Writes headers with styling (blue fill, white text)
3. For each row:
   - Writes text data
   - Embeds images (Current Scenario, Competitor, Proposal) using `openpyxl.drawing.image`
   - Resizes images to 100x100 pixels
4. Sets column widths (20 for images, 40 for long text)
5. Saves to timestamped file

### 3.6 Presentation Engine (`vave_presentation_engine.py`)

**Purpose**: Generates boardroom-ready PowerPoint presentations

**Key Classes**:

1. **`LLMEnrichmentEngine`**:
   - Uses Gemini API to enrich each idea
   - Generates: engineering_logic, critical_risks, validation_plan, supply_chain_analysis
   - Falls back to mock data if API unavailable

2. **`VAVEPresentation`**:
   - Creates PowerPoint using `python-pptx`
   - Widescreen 16:9 format
   - Adds title slide, idea slides, summary slide
   - Embeds images, adds charts

**Key Function**: `generate_deep_dive_ppt(ideas, output_path)`

### 3.7 Data Lake (`data_lake.py`)

**Purpose**: Big-data style storage for analytics

**Technology**: DuckDB (columnar database)

**Structure**:
- `data_lake/raw/`: Timestamped CSV uploads
- `data_lake/processed/`: Parquet files partitioned by `load_date`
- `data_lake/lake.duckdb`: Metadata table (`lake_loads`)

**Functions**:
- `ingest_uploaded_csv()`: Moves CSV to raw zone
- `run_etl_on_raw_csv()`: Converts CSV → Parquet
- `analytics_aggregate_ideas()`: Aggregates metrics
- `lake_status()`: Returns load statistics

---

## 4. Data Flow: User Prompt to Output

### 4.1 Complete Request Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: User submits query via /chat POST                      │
│ Input: {"message": "Reduce weight of MG Hector Brake Disc"}    │
└────────────────────────────┬────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Flask route /chat calls agentic_rag_chat()             │
│ - Extracts user_query                                           │
│ - Logs event to PostgreSQL events table                         │
└────────────────────────────┬────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: VAVEAgent.run(user_query)                               │
│                                                                  │
│ 3.1: Extract Target Component                                   │
│      _smart_extract_target("Reduce weight of MG Hector...")     │
│      → "brake disc"                                             │
│                                                                  │
│ 3.2: Resolve Engineering Context                                │
│      _resolve_engineering_context(query)                         │
│      → Vehicle: hector (1650kg, C-SUV)                           │
│      → Component: brake (thermal mass critical)                 │
│      → Context string: "VEHICLE: MG Hector | WEIGHT: 1650kg..."  │
└────────────────────────────┬────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: Three Parallel Streams                                 │
│                                                                  │
│ ┌──────────────────────────────────────────────────────────┐   │
│ │ STREAM A: Existing DB (RAG)                              │   │
│ │ retrieve_context(query, top_k=5)                         │   │
│ │ → FAISS search → PostgreSQL lookup                      │   │
│ │ → 5 existing ideas                                       │   │
│ └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│ ┌──────────────────────────────────────────────────────────┐   │
│ │ STREAM B: AI Innovation Engine                           │   │
│ │ run_innovation_engine(query, context_ideas)              │   │
│ │   → _generate_with_llm() [12 ideas via Gemini]          │   │
│ │   → _validate_and_filter_ideas() [Physics checks]       │   │
│ │   → Image assignment [Semantic matching]                 │   │
│ │ → 6-8 validated ideas                                   │   │
│ └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│ ┌──────────────────────────────────────────────────────────┐   │
│ │ STREAM C: Web Mining Engine                             │   │
│ │ run_web_mining_engine(query, target_component)          │   │
│ │   → 5 parallel web searches [DuckDuckGo]                │   │
│ │   → LLM synthesis [Physics-adapted]                     │   │
│ │ → 4 research proposals                                  │   │
│ └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: Image Enrichment (VLM Engine)                          │
│                                                                  │
│ For each idea:                                                  │
│   - get_images_for_idea(idea_text, origin, extra_context)      │
│   - Current: Semantic match or generate                        │
│   - Competitor: Semantic match or generate                      │
│   - Proposal: CAD overlay (AI) or annotation (DB/Web)           │
└────────────────────────────┬────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 6: Data Normalization                                      │
│ _normalize_data(existing_ideas, "Existing DB")                  │
│ _normalize_data(new_ideas, "AI Innovation")                     │
│ _normalize_data(web_ideas, "World Wide Web")                    │
│                                                                  │
│ Output: Unified list with consistent keys                       │
└────────────────────────────┬────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 7: Response Formatting                                     │
│ _format_final_response(all_data, user_query)                    │
│                                                                  │
│ Returns:                                                        │
│ {                                                               │
│   "response_text": "Analyzed query: '...'",                    │
│   "table_data": [                                              │
│     {                                                           │
│       "Idea Id": "AI-GEN-01",                                  │
│       "Cost Reduction Idea": "...",                            │
│       "Current Scenario Image": "/static/images/mg/85.jpg",   │
│       "Proposal Scenario Image": "/static/generated/...",      │
│       "Feasibility Score": 78,                                 │
│       "Validation Status": "Auto-Approved",                    │
│       ...                                                       │
│     }                                                           │
│   ]                                                             │
│ }                                                               │
└────────────────────────────┬────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 8: Frontend Rendering                                      │
│ - Displays response_text                                        │
│ - Renders table_data in Analysis Results table                  │
│ - Shows images in modal on click                                │
│ - Provides Excel/PPT export buttons                            │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Detailed Sub-Flows

#### 4.2.1 AI Innovation Engine Flow

```
_generate_with_llm(query, context, target)
  │
  ├─> db_examples = Extract 4 examples from context
  ├─> engineering_context = _resolve_engineering_context(query)
  │     ├─> Detect vehicle (hector/astor/zsev/comet/nexon)
  │     ├─> Detect component (brake/hvac/body/general)
  │     └─> Build context string with constraints
  │
  ├─> system_prompt = engineering_context + CTO instructions
  ├─> prompt = "Construct 12 NEW proposals for {target} on {vehicle_name}"
  │
  ├─> call_llm(prompt, system_prompt)
  │     ├─> _call_gemini() [Primary]
  │     │     ├─> Try gemini-2.5-flash-lite
  │     │     ├─> Try gemini-2.5-flash (fallback)
  │     │     └─> Try gemini-3-flash (fallback)
  │     └─> _call_ollama() [Alternative]
  │
  ├─> _parse_llm_json(raw_response)
  │     ├─> Remove ```json markers
  │     ├─> Extract JSON array via regex
  │     └─> json.loads()
  │
  └─> Return 12 ideas (or fewer if parsing fails)
```

#### 4.2.2 Validation Flow

```
_validate_and_filter_ideas(ideas, target_component, query, min_score=25)
  │
  ├─> For each idea:
  │     │
  │     ├─> Normalize scores (0-100, integers)
  │     │
  │     ├─> Scope check: target tokens in idea text?
  │     │     └─> No → Reject (Out of Scope)
  │     │
  │     ├─> Impossible pattern check (100% reduction, zero cost, etc.)
  │     │     └─> Yes → Reject (Physically Impossible)
  │     │
  │     ├─> Resolve Matrix Context (for physics heuristics)
  │     │     └─> _resolve_engineering_context(query)
  │     │
  │     ├─> Physics Heuristics:
  │     │     ├─> Heavy Vehicle Brake Risk?
  │     │     │     └─> Yes → Flag for Review
  │     │     └─> EV×HVAC Range Impact?
  │     │           └─> Yes → Flag for Review
  │     │
  │     ├─> Scoring Gate:
  │     │     ├─> min_score >= 25 AND no physics flags?
  │     │     │     └─> Auto-Approved
  │     │     └─> Else → Needs Human Review
  │     │
  │     └─> Add to validated[] or needs_review[]
  │
  └─> Return validated + needs_review (max 8)
```

#### 4.2.3 Image Enrichment Flow

```
_enrich_ideas_with_images(ideas, origin, used_images)
  │
  ├─> For each idea:
  │     │
  │     ├─> Extract idea_text, visual_prompt
  │     │
  │     ├─> Resolve base_image (mg_vehicle_image)
  │     │     └─> If missing → _get_smart_image()
  │     │
  │     ├─> Resolve competitor_image
  │     │     └─> If missing → _get_smart_image()
  │     │
  │     ├─> Call VLM Engine:
  │     │     vlm.get_images_for_idea(idea_text, origin, extra_context)
  │     │       │
  │     │       ├─> Current Scenario:
  │     │       │     └─> Use base_image or generate
  │     │       │
  │     │       ├─> Competitor:
  │     │       │     └─> Use competitor_image or generate
  │     │       │
  │     │       └─> Proposal:
  │     │             ├─> If origin == "AI Innovation":
  │     │             │     └─> _construct_overlay_prompt() → Pollinations API
  │     │             └─> Else:
  │     │                   └─> _create_engineering_annotation() → PIL overlay
  │     │
  │     └─> Assign images to idea dict
  │
  └─> Return enriched ideas
```

---

## 5. Technology Stack

### 5.1 Core Framework
- **Flask 2.x**: Web framework
- **Flask-Login**: User authentication
- **Flask-CORS**: Cross-origin resource sharing

### 5.2 AI/ML Libraries
- **Google Generative AI (Gemini)**: Primary LLM (gemini-2.5-flash-lite, gemini-2.5-flash, gemini-3-flash)
- **SentenceTransformer**: Text embeddings (`all-MiniLM-L6-v2` or similar)
- **FAISS**: Vector similarity search (Facebook AI Similarity Search)
- **Transformers (HuggingFace)**: GPT-2 tokenizer, BLIP model
- **PyTorch**: Deep learning backend

### 5.3 Database
- **PostgreSQL**: Primary database (ideas, events tables)
- **SQLite**: User authentication database
- **DuckDB**: Data lake (analytics, Parquet storage)

### 5.4 Image Processing
- **PIL/Pillow**: Image manipulation, annotation overlays
- **Pollinations.ai API**: AI image generation
- **OpenPyXL**: Excel generation with embedded images

### 5.5 Web & Search
- **DuckDuckGo Search**: Web search (via `duckduckgo_search` library)
- **BeautifulSoup**: HTML parsing (if needed)

### 5.6 Document Generation
- **python-pptx**: PowerPoint generation
- **openpyxl**: Excel file generation

### 5.7 Utilities
- **python-dotenv**: Environment variable management
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **requests**: HTTP client

---

## 6. System Design Patterns

### 6.1 Agent Pattern
**VAVEAgent** acts as an autonomous agent that:
- Perceives: User query, database context, web research
- Thinks: Resolves engineering context, validates physics
- Acts: Generates ideas, assigns images, formats output

### 6.2 Strategy Pattern
**LLM Provider Selection**:
- Primary: Gemini API (multiple model fallbacks)
- Alternative: Ollama (local LLM)
- Selection via `LLM_PROVIDER` environment variable

### 6.3 Factory Pattern
**Database Connection Factory**:
- `get_db_connection()`: PostgreSQL factory
- `get_user_db_conn()`: SQLite factory
- `_connect()`: DuckDB factory (data lake)

### 6.4 Template Method Pattern
**Idea Generation Pipeline**:
- Abstract: `run()` defines skeleton
- Concrete: `run_innovation_engine()`, `run_web_mining_engine()` implement steps

### 6.5 Observer Pattern
**Event Logging**:
- `log_event()` writes to PostgreSQL `events` table
- Enables analytics and audit trails

### 6.6 Repository Pattern
**Data Access**:
- `retrieve_context()`: Abstracts FAISS + PostgreSQL
- `vector_db_func`: Injected dependency for testability

### 6.7 Decorator Pattern
**Authentication & Authorization**:
- `@login_required`: Flask-Login decorator
- `@role_required(allowed_roles)`: Custom role-based access

---

## 7. Database Architecture

### 7.1 PostgreSQL Schema

#### `ideas` Table
**Purpose**: Stores cost reduction ideas with full metadata

**Key Columns**:
- `id`: SERIAL PRIMARY KEY
- `idea_id`: TEXT UNIQUE
- `cost_reduction_idea`: TEXT
- `saving_value_inr`: REAL
- `weight_saving`: REAL
- `status`: TEXT
- `way_forward`: TEXT
- `feasibility_score`, `cost_saving_score`, `weight_reduction_score`, `homologation_feasibility_score`: REAL
- `proposal_image_data`, `mg_vehicle_data`, `competitor_vehicle_data`: BYTEA (binary image storage)
- `created_at`: TIMESTAMP

**Indexes**: 
- `idea_id` (UNIQUE)
- Likely indexes on `status`, `dept` for filtering

#### `events` Table
**Purpose**: Event stream for analytics

**Columns**:
- `id`: SERIAL PRIMARY KEY
- `event_type`: TEXT (e.g., "chat_request", "idea_generated")
- `username`: TEXT
- `payload`: JSONB (flexible event data)
- `created_at`: TIMESTAMP (default CURRENT_TIMESTAMP)

### 7.2 SQLite Schema

#### `users` Table
**Purpose**: User authentication

**Columns**:
- `id`: INTEGER PRIMARY KEY
- `username`: TEXT UNIQUE
- `password_hash`: TEXT (bcrypt)
- `role`: TEXT (User/Admin/SuperAdmin)
- `created_at`: TIMESTAMP

### 7.3 DuckDB Schema (Data Lake)

#### `lake_loads` Table
**Purpose**: Metadata for data lake loads

**Columns**:
- `id`: BIGINT
- `logical_dataset`: TEXT
- `raw_path`: TEXT
- `parquet_path`: TEXT
- `load_date`: TEXT
- `load_ts`: TEXT
- `row_count`: BIGINT

**Parquet Storage**:
- Partitioned by `load_date`
- Stored in `data_lake/processed/ideas/load_date=YYYY-MM-DD/`

### 7.4 FAISS Index

**Structure**:
- **Type**: `IndexFlatL2` (Euclidean distance)
- **Dimension**: 384 (SentenceTransformer embedding size)
- **Storage**: `model/faiss_index.bin` (disk persistence)
- **Usage**: In-memory during runtime, rebuilt on startup if missing

---

## 8. API Endpoints & Routes

### 8.1 Authentication Routes

- **`GET /login`**: Login page
- **`POST /login`**: Authenticate user
- **`GET /logout`**: Logout user
- **`GET /register`**: Registration page (if enabled)
- **`POST /register`**: Create new user

### 8.2 Main Application Routes

- **`GET /`**: Redirects to `/chat_app`
- **`GET /chat_app`**: Main chat interface (requires login)
- **`POST /chat`**: Process user query, return ideas
  - **Input**: `{"message": "query text"}`, optional `image` file
  - **Output**: `{"success": true, "response_text": "...", "table_data": [...]}`

### 8.3 Export Routes

- **`POST /generate_excel`**: Generate Excel file from table data
  - **Input**: `{"table_data": [...]}`
  - **Output**: `{"excel_url": "/download/excel/filename.xlsx"}`
- **`POST /generate_ppt`**: Generate PowerPoint presentation
  - **Input**: `{"table_data": [...], "response_text": "..."}`
  - **Output**: `{"ppt_url": "/download/ppt/filename.pptx"}`

### 8.4 Admin Routes (Role-Protected)

- **`GET /admin`**: Admin dashboard (SuperAdmin/Admin only)
- **`GET /admin/users`**: User management (SuperAdmin only)
- **`POST /admin/users`**: Create/update users
- **`GET /admin/upload`**: Knowledge base upload page
- **`POST /admin/upload`**: Process Excel + ZIP upload

### 8.5 Static & Download Routes

- **`GET /static/images/<folder>/<filename>`**: Serve images
- **`GET /static/generated/<filename>`**: Serve generated images
- **`GET /download/excel/<filename>`**: Download Excel file
- **`GET /download/ppt/<filename>`**: Download PowerPoint file

---

## 9. Security & Authentication

### 9.1 Authentication Flow

1. User submits credentials via `/login` POST
2. Flask-Login validates against SQLite `users` table
3. Password verified via `check_password_hash()` (bcrypt)
4. Session created, user object stored in `flask.session`
5. `@login_required` decorator checks authentication on protected routes

### 9.2 Authorization (Role-Based Access Control)

**Roles**:
- **User**: Default role, can use chat, generate exports
- **Admin**: Can upload knowledge base, manage some users
- **SuperAdmin**: Full access, user management, system configuration

**Implementation**:
```python
@role_required(['SuperAdmin', 'Admin'])
def admin_route():
    ...
```

### 9.3 Security Measures

1. **SQL Injection Prevention**:
   - Parameterized queries (PostgreSQL `%s`, SQLite `?`)
   - `execute_sql_query()` blocks non-SELECT statements

2. **File Upload Security**:
   - `secure_filename()` sanitizes filenames
   - `allowed_file()` checks extensions
   - `MAX_UPLOAD_SIZE` limit (50MB)

3. **CORS Configuration**:
   - `Flask-CORS` enabled (configurable origins)

4. **Secret Key**:
   - Stored in `.env` file (not committed)
   - Used for session signing

---

## 10. Performance & Scalability

### 10.1 Optimization Strategies

1. **Lazy Loading**:
   - VAVEAgent initialized on first use (not at startup)
   - Models loaded only when needed

2. **Caching**:
   - FAISS index persisted to disk
   - Generated images cached in `static/generated/`
   - Image embeddings pre-computed on startup

3. **Parallel Processing**:
   - Web search: 3 workers via `ThreadPoolExecutor`
   - Image processing: Sequential (can be parallelized)

4. **Database Connection Pooling**:
   - New connection per request (can be optimized with connection pool)

### 10.2 Scalability Considerations

**Current Limitations**:
- Single-threaded Flask (development server)
- No horizontal scaling (single instance)
- FAISS index in-memory (limited by RAM)

**Potential Improvements**:
- **Production WSGI Server**: Gunicorn/uWSGI with multiple workers
- **Redis Cache**: Cache frequent queries, image embeddings
- **Distributed FAISS**: Shard index across multiple nodes
- **Async Processing**: Celery for long-running tasks (PPT generation)
- **CDN**: Serve static images from CDN
- **Database Read Replicas**: Separate read/write databases

### 10.3 Performance Metrics

**Typical Response Times** (estimated):
- Vector search: ~50-100ms
- LLM generation (12 ideas): ~5-15 seconds
- Web search (5 queries): ~10-20 seconds
- Image generation: ~2-5 seconds per image
- **Total end-to-end**: ~20-40 seconds

**Bottlenecks**:
1. LLM API calls (network latency)
2. Web search (rate limiting, network)
3. Image generation (Pollinations API)

---

## 11. System Initialization Flow

### 11.1 Startup Sequence (`app.py` main block)

```
1. init_user_db_schema()
   └─> Ensure users.db exists, create default SuperAdmin

2. init_db()
   └─> Ensure PostgreSQL tables exist (ideas, events)

3. setup_model()
   └─> Load GPT-2 tokenizer, BLIP model, SentenceTransformer
   └─> Move to GPU if available

4. build_vector_db()
   └─> Load ideas from PostgreSQL
   └─> Generate embeddings
   └─> Build/save FAISS index
   └─> Initialize VAVEAgent

5. setup_vlm()
   └─> Load BLIP model for image captioning
   └─> Build image index from database

6. cleanup_temp_files()
   └─> Remove old temp files

7. app.run(host='0.0.0.0', port=5000)
```

### 11.2 Initialization Dependencies

```
init_user_db_schema (no dependencies)
    │
    ▼
init_db (requires PostgreSQL connection)
    │
    ▼
setup_model (requires MODEL_DIR, GPU optional)
    │
    ▼
build_vector_db (requires init_db, setup_model)
    │
    ├─> Loads ideas from PostgreSQL
    ├─> Generates embeddings (SentenceTransformer)
    ├─> Builds FAISS index
    └─> Initializes VAVEAgent
        │
        └─> VAVEAgent.__init__()
            ├─> Loads image index (static/image_captions.json)
            ├─> Builds image embeddings
            └─> Initializes VLM Engine (if dependencies available)
    │
    ▼
setup_vlm (requires build_vector_db for image data)
    │
    └─> Loads BLIP model
        └─> Processes images from database
```

---

## 12. Error Handling & Resilience

### 12.1 Error Handling Strategy

1. **Graceful Degradation**:
   - LLM failure → Falls back to legacy `generate_response()`
   - VLM failure → Uses placeholder images
   - Web search failure → Uses fallback trend text

2. **Logging**:
   - All errors logged to `logger.log` and console
   - Structured logging with timestamps, levels, context

3. **User-Friendly Messages**:
   - API errors return JSON with `"error"` key
   - Frontend displays user-friendly messages

### 12.2 Retry Logic

**Web Search** (`tools.py`):
- 5 retries with exponential backoff
- Random wait (3-8s) to mimic human behavior
- Rate limiting via DuckDuckGo backend selection

**LLM Calls** (`agent.py`):
- Model fallback chain (3 models)
- Rate limit handling (429 errors → sleep 5s)

---

## 13. Testing & Validation

### 13.1 Test Files

- **`test_matrix.py`**: Validates Vehicle×Component matrix logic
  - Case A: Hector brake disc (should flag thermal risk)
  - Case B: Comet brake disc (should auto-approve)
  - Case C: ZS EV blower power (should flag range impact)

### 13.2 Validation Mechanisms

1. **Autonomous Validation**: `_validate_and_filter_ideas()`
2. **Physics Heuristics**: Heavy vehicle brake risk, EV HVAC range impact
3. **Score Thresholds**: Minimum 25/100 across all categories
4. **Scope Validation**: Target component must appear in idea text

---

## 14. Future Enhancements & Recommendations

### 14.1 Short-Term

1. **Connection Pooling**: Use `psycopg2.pool` for PostgreSQL
2. **Async Image Processing**: Parallelize VLM image generation
3. **Caching Layer**: Redis for frequent queries
4. **Better Error Messages**: More specific validation feedback

### 14.2 Long-Term

1. **Multi-Tenancy**: Support multiple organizations
2. **Real-Time Collaboration**: WebSocket for live updates
3. **Advanced Analytics**: Dashboard with charts, trends
4. **Mobile App**: React Native or Flutter
5. **API Gateway**: Separate API service from web UI
6. **Microservices**: Split into independent services (Agent, VLM, DB, etc.)

---

## 15. Conclusion

The VAVE AI System represents a sophisticated integration of **RAG**, **Multi-Agent AI**, **Physics-Informed Constraints**, and **Visual Language Models** to deliver enterprise-grade cost reduction ideation for automotive engineering.

**Key Innovations**:
1. **Dynamic Vehicle×Component Matrix**: Prevents physically impossible suggestions
2. **Three-Stream Generation**: Ensures comprehensive idea coverage
3. **Autonomous Validation**: Reduces manual review burden
4. **Semantic Visual Matching**: Context-aware image selection
5. **Physics-Informed Heuristics**: Flags risky proposals automatically

**System Strengths**:
- Modular, testable architecture
- Graceful degradation on failures
- Comprehensive logging and analytics
- Role-based security
- Production-ready export capabilities

**Areas for Improvement**:
- Performance optimization (caching, async)
- Horizontal scalability
- Enhanced error recovery
- More sophisticated validation rules

---

**Document Version**: 1.0  
**Last Updated**: 2025-01-20  
**Author**: System Analysis  
**Status**: Comprehensive System Design Report
