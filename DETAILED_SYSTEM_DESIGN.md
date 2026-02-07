# VAVE AI System - Detailed High-Level & Low-Level System Design

## Table of Contents

1. [Detailed High-Level Architecture](#1-detailed-high-level-architecture)
2. [Core Low-Level System Design](#2-core-low-level-system-design)
3. [Complete User Input to Output Flow](#3-complete-user-input-to-output-flow)
4. [AI Agent Architecture - Comprehensive](#4-ai-agent-architecture---comprehensive)
5. [Internal State Management](#5-internal-state-management)
6. [Data Structures & Algorithms](#6-data-structures--algorithms)
7. [Concurrency & Threading Model](#7-concurrency--threading-model)
8. [Memory Management](#8-memory-management)
9. [API Call Sequences](#9-api-call-sequences)

---

## 1. Detailed High-Level Architecture

### 1.1 Layered Architecture with Component Interactions

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PRESENTATION LAYER                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Frontend (Flask Templates + JavaScript)                            │   │
│  │  - chat_app.html: Main UI with query input, results table           │   │
│  │  - app.js: AJAX calls to /chat, image modal handling               │   │
│  │  - style.css: Responsive styling                                    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────┬──────────────────────────────────────────┘
                                     │ HTTP POST/GET
                                     │ JSON Request/Response
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         APPLICATION SERVICE LAYER                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Flask Application (app.py)                                          │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │   │
│  │  │ Route Handler│  │ Auth Manager │  │ Event Logger  │             │   │
│  │  │ /chat        │  │ Flask-Login  │  │ log_event()  │             │   │
│  │  │ /generate_*  │  │ Role-Based   │  │ PostgreSQL   │             │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘             │   │
│  │                                                                      │   │
│  │  ┌──────────────────────────────────────────────────────────────┐  │   │
│  │  │  Global State Management                                     │  │   │
│  │  │  - embedding_model: SentenceTransformer                      │  │   │
│  │  │  - faiss_index: FAISS IndexFlatL2                            │  │   │
│  │  │  - vave_agent: VAVEAgent (lazy initialized)                 │  │   │
│  │  │  - idea_texts: List[str] (in-memory embeddings)              │  │   │
│  │  │  - idea_rows: List[Dict] (cached DB rows)                    │  │   │
│  │  └──────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────┬──────────────────────────────────────────┘
                                     │ Function Call
                                     │ agentic_rag_chat()
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      AI AGENT ORCHESTRATION LAYER                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  VAVEAgent (agent.py)                                                │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │  Core Orchestrator: run(user_query: str)                     │   │   │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │   │   │
│  │  │  │ Component   │  │ Context      │  │ Image        │      │   │   │
│  │  │  │ Extractor   │  │ Resolver     │  │ Index        │      │   │   │
│  │  │  │ _smart_     │  │ _resolve_    │  │ _load_       │      │   │   │
│  │  │  │ extract_    │  │ engineering_ │  │ image_       │      │   │   │
│  │  │  │ target()    │  │ context()    │  │ index()      │      │   │   │
│  │  │  └──────────────┘  └──────────────┘  └──────────────┘      │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  │                                                                      │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │  Three Parallel Streams (Executed Sequentially)             │   │   │
│  │  │                                                              │   │   │
│  │  │  STREAM A: Existing DB (RAG)                                 │   │   │
│  │  │  ┌────────────────────────────────────────────────────┐     │   │   │
│  │  │  │ vector_db_func(query, top_k=5)                    │     │   │   │
│  │  │  │  → retrieve_context() in app.py                   │     │   │   │
│  │  │  │     → embedding_model.encode(query)               │     │   │   │
│  │  │  │     → faiss_index.search(query_embedding, k=5)    │     │   │   │
│  │  │  │     → PostgreSQL SELECT WHERE id IN (...)          │     │   │   │
│  │  │  │  → Returns: (existing_ideas: List[Dict], context) │     │   │   │
│  │  │  └────────────────────────────────────────────────────┘     │   │   │
│  │  │                                                              │   │   │
│  │  │  STREAM B: AI Innovation Engine                            │   │   │
│  │  │  ┌────────────────────────────────────────────────────┐     │   │   │
│  │  │  │ run_innovation_engine(query, context_ideas)        │     │   │   │
│  │  │  │  → _generate_with_llm()                           │     │   │   │
│  │  │  │     → _resolve_engineering_context()              │     │   │   │
│  │  │  │     → call_llm() [Gemini API]                     │     │   │   │
│  │  │  │     → _parse_llm_json()                           │     │   │   │
│  │  │  │  → _validate_and_filter_ideas()                   │     │   │   │
│  │  │  │     → Physics heuristics                          │     │   │   │
│  │  │  │     → Score normalization                         │     │   │   │
│  │  │  │  → Image assignment (semantic matching)          │     │   │   │
│  │  │  │  → Returns: validated_ideas: List[Dict]            │     │   │   │
│  │  │  └────────────────────────────────────────────────────┘     │   │   │
│  │  │                                                              │   │   │
│  │  │  STREAM C: Web Mining Engine                               │   │   │
│  │  │  ┌────────────────────────────────────────────────────┐     │   │   │
│  │  │  │ run_web_mining_engine(query, target_component)    │     │   │   │
│  │  │  │  → Generate 5 search queries                      │     │   │   │
│  │  │  │  → ThreadPoolExecutor (3 workers)                 │     │   │   │
│  │  │  │     → perform_web_search() [DuckDuckGo]           │     │   │   │
│  │  │  │  → Aggregate web_context: str                     │     │   │   │
│  │  │  │  → call_llm() [Physics-adapted synthesis]         │     │   │   │
│  │  │  │  → _parse_llm_json()                              │     │   │   │
│  │  │  │  → Returns: web_ideas: List[Dict]                 │     │   │   │
│  │  │  └────────────────────────────────────────────────────┘     │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  │                                                                      │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │  Image Enrichment (VLM Engine)                              │   │   │
│  │  │  → _enrich_ideas_with_images() for each stream              │   │   │
│  │  │     → vlm.get_images_for_idea()                             │   │   │
│  │  │        → Semantic matching or generation                     │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  │                                                                      │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │  Data Normalization & Formatting                             │   │   │
│  │  │  → _normalize_data() for each stream                         │   │   │
│  │  │  → _format_final_response()                                  │   │   │
│  │  │  → Stores in self._last_result_data                          │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────┬──────────────────────────────────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        ▼                            ▼                            ▼
┌──────────────┐            ┌──────────────┐            ┌──────────────┐
│ LLM Service  │            │ VLM Engine   │            │ Vector DB    │
│ (Gemini API) │            │ (vlm_engine) │            │ (FAISS)      │
│              │            │              │            │              │
│ - HTTP POST  │            │ - Image      │            │ - IndexFlatL2│
│ - JSON       │            │   matching   │            │ - 384-dim    │
│ - Streaming  │            │ - PIL        │            │ - L2 distance│
│   (optional) │            │   overlays   │            │ - In-memory  │
│              │            │ - Pollinations│           │ - Disk cache │
└──────────────┘            └──────────────┘            └──────────────┘
        │                            │                            │
        └────────────────────────────┼────────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA PERSISTENCE LAYER                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                    │
│  │ PostgreSQL   │  │ SQLite       │  │ DuckDB       │                    │
│  │              │  │              │  │              │                    │
│  │ - ideas      │  │ - users      │  │ - lake_loads │                    │
│  │ - events     │  │ - auth only  │  │ - Parquet    │                    │
│  │ - BYTEA      │  │ - file-based │  │ - Analytics  │                    │
│  │   images     │  │              │  │              │                    │
│  └──────────────┘  └──────────────┘  └──────────────┘                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Component Dependency Graph

```
app.py (Flask Application)
  │
  ├─> VAVEAgent (agent.py)
  │     │
  │     ├─> retrieve_context() [app.py function]
  │     │     ├─> embedding_model (SentenceTransformer)
  │     │     ├─> faiss_index (FAISS)
  │     │     └─> PostgreSQL (get_db_connection())
  │     │
  │     ├─> VLMEngine (vlm_engine.py)
  │     │     ├─> db_conn (PostgreSQL connection)
  │     │     ├─> faiss_index (for image search)
  │     │     └─> sentence_model (for embeddings)
  │     │
  │     ├─> call_llm() → Gemini API
  │     │     └─> google.generativeai
  │     │
  │     └─> perform_web_search() [tools.py]
  │           └─> duckduckgo_search
  │
  ├─> build_vector_db() [app.py]
  │     ├─> PostgreSQL
  │     ├─> embedding_model
  │     └─> FAISS
  │
  ├─> generate_excel_from_table_in_memory() [excel_generator_engine.py]
  │     └─> openpyxl
  │
  └─> generate_deep_dive_ppt() [vave_presentation_engine.py]
        ├─> python-pptx
        └─> Gemini API (LLMEnrichmentEngine)
```

---

## 2. Core Low-Level System Design

### 2.1 VAVEAgent Class: Internal State & Data Structures

**File**: `agent.py`, Lines 92-994

#### 2.1.1 Instance Variables

```python
class VAVEAgent:
    def __init__(self, ...):
        # Core Dependencies
        self.db_path: str                    # Legacy SQLite path (reference only)
        self.vector_db_func: Callable        # Function: (query, top_k) -> (List[Dict], str)
        self.llm_client: Optional            # Optional custom LLM client
        self.pg_conn_func: Optional[Callable] # PostgreSQL connection factory
        self.sentence_model: Optional        # SentenceTransformer model
        
        # Image Index (Semantic Search)
        self.image_index: Dict[str, str]     # {filename: caption}
        self.image_filenames: List[str]      # Ordered list for indexing
        self.image_embeddings: np.ndarray     # Shape: (N, 384) normalized embeddings
        
        # VLM Engine
        self.vlm: Optional[VLMEngine]        # Initialized if dependencies available
        
        # Runtime State (set by _resolve_engineering_context)
        self._last_vehicle_key: str           # "hector", "astor", etc.
        self._last_component_key: str         # "brake", "hvac", etc.
        self._last_vehicle_name: str         # "MG Hector"
        self._last_vehicle_type: str         # "ICE", "EV"
        self._last_vehicle_weight: Optional[int] # 1650, 1350, etc.
        
        # Result Cache
        self._last_result_data: List[Dict]   # Cached normalized results
```

#### 2.1.2 Key Data Structures

**1. Idea Dictionary Structure**:
```python
idea: Dict[str, Any] = {
    # Core Fields
    "idea_id": str,                          # "AI-GEN-01", "WEB-RESEARCH-01"
    "cost_reduction_idea": str,              # Full technical description
    "way_forward": str,                      # Engineering narrative
    "homologation_theory": str,              # Regulatory impact
    
    # Scoring (0-100 integers)
    "feasibility_score": int,
    "cost_saving_score": int,
    "weight_reduction_score": int,
    "homologation_feasibility_score": int,
    
    # Visual
    "visual_prompt": str,                    # <= 15 words description
    "mg_vehicle_image": str,                 # Path: "/static/images/mg/85.jpg"
    "current_scenario_image": str,           # Same as mg_vehicle_image
    "proposal_scenario_image": str,          # "/static/generated/overlay_xxx.jpg"
    "competitor_image": str,                 # "/static/images/competitor/1.jpg"
    
    # Metadata
    "vehicle_name": str,                     # "MG Hector" (from matrix)
    "vehicle_key": str,                      # "hector"
    "component_key": str,                    # "brake"
    "status": str,                           # "AI Proposal", "Research Paper"
    "dept": str,                             # "VAVE"
    "saving_value_inr": float,               # 0.0 (default)
    "weight_saving": float,                  # 0.0 (default)
    
    # Validation
    "validation_status": str,                # "Auto-Approved", "Needs Human Review"
    "validation_notes": str,                 # Explanation of validation decision
}
```

**2. Image Index Structure**:
```python
# Loaded from static/image_captions.json
image_index: Dict[str, str] = {
    "85.jpg": "Close up of black car rear bumper with parking sensors",
    "120.jpg": "Silver alloy wheel rim showing brake caliper",
    ...
}

# Computed on initialization
image_embeddings: np.ndarray = np.array([
    [0.123, -0.456, ..., 0.789],  # Embedding for "85.jpg"
    [0.234, -0.567, ..., 0.890],  # Embedding for "120.jpg"
    ...
])  # Shape: (N_images, 384), L2-normalized
```

**3. Vehicle×Component Matrix**:
```python
# Static dictionaries at module level (agent.py lines 20-88)
VEHICLE_DB: Dict[str, Dict[str, Any]] = {
    "hector": {
        "name": "MG Hector",
        "type": "ICE",
        "segment": "C-SUV",
        "weight_kg": 1650,
        "volume": "36,000/year",
        "constraints": "High Kinetic Energy (1.7T)..."
    },
    ...
}

COMPONENT_DB: Dict[str, Dict[str, str]] = {
    "brake": {
        "system": "Braking",
        "physics": "Thermal Mass, Friction Coeff...",
        "regulations": "FMVSS 105, ECE R13.",
        "focus": "Rotor thickness, Caliper material..."
    },
    ...
}
```

### 2.2 Low-Level Algorithm: Semantic Image Matching

**File**: `agent.py`, Lines 113-156

```python
def _evaluate_visual_match(
    query_text: str,
    folder_name: str,
    used_images: set,
    threshold: float = 0.35
) -> tuple[str, str]:
    """
    Algorithm: Cosine Similarity Search
    
    Time Complexity: O(N) where N = number of images
    Space Complexity: O(1) (no additional structures)
    """
    # Step 1: Encode query (1 embedding vector)
    query_vec = self.sentence_model.encode([query_text])  # Shape: (1, 384)
    query_norm = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
    
    # Step 2: Matrix multiplication (vectorized)
    # query_norm: (1, 384)
    # image_embeddings: (N, 384)
    # Result: (1, N) similarity scores
    scores = np.dot(query_norm, self.image_embeddings.T).flatten()  # Shape: (N,)
    
    # Step 3: Sort by similarity (descending)
    top_indices = np.argsort(scores)[::-1]  # O(N log N)
    
    # Step 4: Find first unused match above threshold
    for idx in top_indices:
        score = scores[idx]
        if score < threshold:
            break  # Early termination (scores are sorted)
        
        filename = self.image_filenames[idx]
        if filename not in used_images:
            matched_label = self.image_index[filename]
            return f"/static/images/{folder_name}/{filename}", matched_label
    
    return "NaN", "No Matching Image Found"
```

**Performance Characteristics**:
- **Pre-computation**: Image embeddings computed once at startup (O(N * 384 * embedding_time))
- **Query time**: O(N log N) for sorting, O(N) worst-case for linear search
- **Memory**: O(N * 384 * 4 bytes) ≈ 1.5MB for 1000 images

### 2.3 Low-Level Algorithm: Context Resolution

**File**: `agent.py`, Lines 320-392

```python
def _resolve_engineering_context(self, query: str) -> str:
    """
    Algorithm: Pattern Matching + Dictionary Lookup
    
    Time Complexity: O(M * K) where M = query length, K = max aliases per vehicle/component
    Space Complexity: O(1)
    """
    q = (query or "").lower()
    
    # Step 1: Vehicle Detection (O(M * K))
    vehicle_key = "hector"  # Default
    vehicle_aliases = {
        "hector": ["mg hector", "hector"],
        "astor": ["mg astor", "astor"],
        "zsev": ["zs ev", "mg zs ev", "zsev", "zs-ev", "zs electric"],
        "comet": ["mg comet", "comet"],
        "nexon": ["tata nexon", "nexon"],
    }
    
    for key, aliases in vehicle_aliases.items():
        if any(a in q for a in aliases):  # O(M * len(aliases))
            vehicle_key = key
            break  # First match wins
    
    # Step 2: Component Detection (O(M * K))
    component_key = "general"  # Default
    component_aliases = {
        "brake": ["brake", "brakes", "disc", "rotor", "caliper", ...],
        "hvac": ["hvac", "blower", "blowing", "ac", "a/c", ...],
        "body": ["biw", "body", "door", "bumper", ...],
    }
    
    for key, aliases in component_aliases.items():
        if any(a in q for a in aliases):
            component_key = key
            break
    
    # Step 3: Dictionary Lookup (O(1))
    vehicle = VEHICLE_DB.get(vehicle_key, VEHICLE_DB["hector"])
    component = COMPONENT_DB.get(component_key, COMPONENT_DB["general"])
    
    # Step 4: Special Rule Application (O(1))
    extra_rules = ""
    if vehicle.get("type", "").upper() == "EV" and component_key == "hvac":
        extra_rules = "EV×HVAC RULE: Range impact is critical..."
    
    # Step 5: String Formatting (O(1))
    context = f"VEHICLE: {vehicle['name']} | WEIGHT: {vehicle['weight_kg']}..."
    
    # Step 6: State Update (O(1))
    self._last_vehicle_key = vehicle_key
    self._last_component_key = component_key
    self._last_vehicle_name = vehicle.get("name")
    # ... (other state variables)
    
    return context
```

### 2.4 Low-Level Algorithm: Validation & Filtering

**File**: `agent.py`, Lines 670-761

```python
def _validate_and_filter_ideas(
    ideas: List[Dict],
    target_component: str,
    query: str = "",
    min_score: int = 25
) -> List[Dict]:
    """
    Algorithm: Multi-Pass Filtering with Heuristics
    
    Time Complexity: O(N * M) where N = ideas, M = average idea text length
    Space Complexity: O(N) for validated/needs_review lists
    """
    validated: List[Dict] = []
    needs_review: List[Dict] = []
    
    # Pre-compute target tokens (O(M))
    target_tokens = [
        w for w in re.findall(r'\b[a-zA-Z0-9]+\b', target_component.lower())
        if len(w) > 3
    ]
    
    # Resolve context once (O(M * K))
    self._resolve_engineering_context(query or target_component)
    vehicle_weight = getattr(self, "_last_vehicle_weight", None)
    vehicle_type = str(getattr(self, "_last_vehicle_type", "")).upper()
    component_key = getattr(self, "_last_component_key", "general")
    
    # Pass 1: Normalize & Validate (O(N))
    for idea in ideas:
        # 1.1: Score Normalization (O(1))
        for key in ["feasibility_score", "cost_saving_score", ...]:
            raw = idea.get(key, 0)
            try:
                val = int(float(raw))
                idea[key] = max(0, min(100, val))  # Clamp to [0, 100]
            except:
                idea[key] = 0
        
        # 1.2: Text Blob Creation (O(M))
        text_blob = f"{idea.get('cost_reduction_idea', '')} {idea.get('way_forward', '')}".lower()
        
        # 1.3: Scope Check (O(M * T) where T = target_tokens)
        scope_ok = True
        if target_tokens:
            scope_ok = any(tok in text_blob for tok in target_tokens)
        
        if not scope_ok:
            idea["validation_status"] = "Rejected - Out of Scope"
            continue  # Skip to next idea
        
        # 1.4: Impossible Pattern Check (O(M * P) where P = patterns)
        impossible_patterns = [
            r"100%\s*weight\s*reduction",
            r"zero\s*cost\s*manufacturing",
            ...
        ]
        impossible_flag = any(re.search(pat, text_blob) for pat in impossible_patterns)
        
        if impossible_flag:
            idea["validation_status"] = "Rejected - Physically Impossible"
            continue
        
        # 1.5: Physics Heuristics (O(M))
        heavy_brake_risk = False
        if component_key == "brake" and isinstance(vehicle_weight, (int, float)) and vehicle_weight >= 1600:
            if any(k in text_blob for k in ["reduce weight", "lightweight", "thin", ...]):
                if any(k in text_blob for k in ["disc", "rotor", ...]):
                    heavy_brake_risk = True
        
        ev_hvac_range_risk = False
        if vehicle_type == "EV" and component_key == "hvac":
            if any(k in text_blob for k in ["increase blower", "increase power", ...]):
                ev_hvac_range_risk = True
        
        # 1.6: Scoring Gate (O(1))
        min_idea_score = min(idea.get(k, 0) for k in score_keys)
        scores_ok = min_idea_score >= min_score
        
        # 1.7: Classification
        if scores_ok and not heavy_brake_risk and not ev_hvac_range_risk:
            idea["validation_status"] = "Auto-Approved"
            validated.append(idea)
        else:
            idea["validation_status"] = "Needs Human Review"
            if heavy_brake_risk:
                idea["validation_notes"] = f"{vehicle_name}: Heavy-vehicle brake thermal risk..."
            elif ev_hvac_range_risk:
                idea["validation_notes"] = f"{vehicle_name}: EV×HVAC range impact risk..."
            else:
                idea["validation_notes"] = f"Score {min_idea_score} below threshold..."
            needs_review.append(idea)
    
    # Pass 2: Merge & Limit (O(1))
    final_list = validated + needs_review
    return final_list[:8]  # Return top 8
```

### 2.5 Low-Level Algorithm: LLM Call with Fallback

**File**: `agent.py`, Lines 214-257

```python
def _call_gemini(self, prompt: str, system_prompt: str) -> str:
    """
    Algorithm: Model Fallback Chain
    
    Time Complexity: O(1) per model attempt, O(M) total where M = models
    Space Complexity: O(1)
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "ERROR_MISSING_KEY"
    
    genai.configure(api_key=api_key)
    
    # Model priority list (fastest to slowest)
    candidate_models = [
        "gemini-2.5-flash-lite",  # Fastest, lower cost
        "gemini-2.5-flash",       # Balanced
        "gemini-3-flash"           # Latest, may be slower
    ]
    
    last_error = ""
    
    # Try each model in order
    for model_name in candidate_models:
        try:
            # Step 1: Initialize model (O(1))
            model = genai.GenerativeModel(model_name)
            
            # Step 2: Construct full prompt (O(P) where P = prompt length)
            full_prompt = f"System Instruction: {system_prompt}\n\nUser Query: {prompt}"
            
            # Step 3: Configure generation (O(1))
            generation_config = genai.types.GenerationConfig(temperature=0.7)
            
            # Step 4: API Call (Network I/O, ~2-15 seconds)
            response = model.generate_content(full_prompt, generation_config=generation_config)
            
            # Step 5: Extract text (O(R) where R = response length)
            if response.text:
                return response.text
            else:
                raise ValueError("Empty response text")
        
        except Exception as e:
            error_str = str(e)
            last_error = error_str
            
            # Handle rate limiting (429)
            if "429" in error_str:
                logger.warning(f"Rate Limit Hit ({model_name}). Sleeping 5s...")
                time.sleep(5)  # Wait before retry
                continue
            
            # Handle model not found (404)
            elif "404" in error_str or "not found" in error_str.lower():
                continue  # Try next model
            
            # Other errors
            else:
                logger.error(f"Gemini API Error ({model_name}): {e}")
                continue  # Try next model
    
    # All models failed
    return f"ERROR_NO_MODELS_WORKING. Last error: {last_error}"
```

---

## 3. Complete User Input to Output Flow

### 3.1 Detailed Execution Sequence

**Scenario**: User submits query "Reduce weight of MG Hector Brake Disc"

#### Phase 1: HTTP Request Reception (app.py)

```
Time: T0
Location: app.py, Line 3659
Event: POST /chat received

┌─────────────────────────────────────────────────────────────┐
│ Step 1.1: Request Parsing                                   │
│ - Extract user_query from request.form.get('message')       │
│ - Extract image_path from request.files.get('image')        │
│ - Validate: if not user_query, return empty response       │
└─────────────────────────────────────────────────────────────┘

Time: T0 + 5ms
Location: app.py, Line 3686
Event: Event Logging

┌─────────────────────────────────────────────────────────────┐
│ Step 1.2: Log Event                                          │
│ log_event("chat_request", username, payload)                │
│   → PostgreSQL INSERT INTO events (...)                     │
│   → Returns immediately (async logging)                    │
└─────────────────────────────────────────────────────────────┘

Time: T0 + 10ms
Location: app.py, Line 3695
Event: Call Agent Function

┌─────────────────────────────────────────────────────────────┐
│ Step 1.3: Invoke Agent                                      │
│ result = agentic_rag_chat(username, user_query, image_path) │
└─────────────────────────────────────────────────────────────┘
```

#### Phase 2: Agent Initialization (app.py → agent.py)

```
Time: T0 + 15ms
Location: app.py, Line 4819
Event: Lazy Agent Initialization Check

┌─────────────────────────────────────────────────────────────┐
│ Step 2.1: Check if vave_agent is None                      │
│ if vave_agent is None:                                      │
│   → Initialize VAVEAgent                                    │
└─────────────────────────────────────────────────────────────┘

Time: T0 + 20ms
Location: app.py, Line 4823
Event: VAVEAgent.__init__()

┌─────────────────────────────────────────────────────────────┐
│ Step 2.2: Agent Constructor                                  │
│ VAVEAgent(                                                  │
│   db_path=str(DB_PATH),                                     │
│   vector_db_func=retrieve_context,  # Function reference    │
│   pg_conn_func=get_db_connection,   # Function reference    │
│   db_conn=get_db_connection,        # Function reference    │
│   faiss_index=faiss_index,          # Global FAISS index   │
│   sentence_model=embedding_model    # Global SentenceTransformer
│ )                                                            │
└─────────────────────────────────────────────────────────────┘

Time: T0 + 25ms
Location: agent.py, Line 105
Event: Load Image Index

┌─────────────────────────────────────────────────────────────┐
│ Step 2.3: Image Index Loading                                │
│ _load_image_index()                                          │
│   → Read static/image_captions.json                          │
│   → Parse JSON: {filename: caption}                          │
│   → Store in self.image_index: Dict[str, str]               │
└─────────────────────────────────────────────────────────────┘

Time: T0 + 30ms
Location: agent.py, Line 97
Event: Build Image Embeddings

┌─────────────────────────────────────────────────────────────┐
│ Step 2.4: Semantic Embedding Computation                     │
│ if self.sentence_model:                                      │
│   self.image_filenames = list(self.image_index.keys())      │
│   captions = list(self.image_index.values())                │
│   embeddings = self.sentence_model.encode(captions)         │
│     → Shape: (N_images, 384)                                 │
│   norms = np.linalg.norm(embeddings, axis=1, keepdims=True) │
│   self.image_embeddings = embeddings / norms                │
│     → L2-normalized for cosine similarity                    │
│   → Time: ~1-2 seconds for 1000 images                      │
└─────────────────────────────────────────────────────────────┘

Time: T0 + 2000ms
Location: agent.py, Line 114
Event: VLM Engine Initialization

┌─────────────────────────────────────────────────────────────┐
│ Step 2.5: VLM Engine Setup                                   │
│ if db_conn and faiss_index and sentence_model:              │
│   from vlm_engine import VLMEngine                          │
│   self.vlm = VLMEngine(db_conn, faiss_index, sentence_model)│
│   → VLMEngine.__init__()                                     │
│     → Set base_dir, static_gen_dir                           │
│     → Create directories if needed                           │
└─────────────────────────────────────────────────────────────┘

Time: T0 + 2010ms
Location: app.py, Line 4840
Event: Agent Ready, Call run()

┌─────────────────────────────────────────────────────────────┐
│ Step 2.6: Execute Agent Pipeline                            │
│ response_text = vave_agent.run(user_query)                  │
└─────────────────────────────────────────────────────────────┘
```

#### Phase 3: Target Component Extraction (agent.py)

```
Time: T0 + 2015ms
Location: agent.py, Line 482
Event: Extract Target Component

┌─────────────────────────────────────────────────────────────┐
│ Step 3.1: Smart Target Extraction                           │
│ target_component = self._smart_extract_target(user_query)   │
│                                                              │
│ Algorithm:                                                   │
│   1. Tokenize: re.findall(r'\b[a-zA-Z0-9]+\b', query)      │
│      → ["reduce", "weight", "of", "mg", "hector", ...]     │
│   2. Filter stopwords: {"reduce", "weight", "of", ...}     │
│      → ["mg", "hector", "brake", "disc"]                    │
│   3. Return last 2 meaningful words                         │
│      → "brake disc"                                          │
│                                                              │
│ Result: target_component = "brake disc"                     │
└─────────────────────────────────────────────────────────────┘
```

#### Phase 4: Engineering Context Resolution (agent.py)

```
Time: T0 + 2020ms
Location: agent.py, Line 320 (called from multiple places)
Event: Resolve Vehicle×Component Context

┌─────────────────────────────────────────────────────────────┐
│ Step 4.1: Vehicle Detection                                 │
│ _resolve_engineering_context(user_query)                     │
│                                                              │
│ Query: "Reduce weight of MG Hector Brake Disc"              │
│ q = query.lower() = "reduce weight of mg hector brake disc" │
│                                                              │
│ Vehicle Aliases Check:                                      │
│   - "mg hector" in q? → True                                │
│   → vehicle_key = "hector"                                  │
│                                                              │
│ Component Aliases Check:                                    │
│   - "brake" in q? → True                                    │
│   - "disc" in q? → True                                     │
│   → component_key = "brake"                                 │
└─────────────────────────────────────────────────────────────┘

Time: T0 + 2025ms
Location: agent.py, Line 373
Event: Build Context String

┌─────────────────────────────────────────────────────────────┐
│ Step 4.2: Context String Construction                        │
│ vehicle = VEHICLE_DB["hector"]                              │
│   → {"name": "MG Hector", "weight_kg": 1650, ...}          │
│                                                              │
│ component = COMPONENT_DB["brake"]                           │
│   → {"system": "Braking", "physics": "Thermal Mass...", ...}│
│                                                              │
│ context = f"""                                               │
│   VEHICLE: MG Hector | TYPE: ICE | WEIGHT_KG: 1650         │
│   VEHICLE CONSTRAINTS: High Kinetic Energy (1.7T)...        │
│   COMPONENT SYSTEM: Braking                                 │
│   COMPONENT PHYSICS: Thermal Mass, Friction Coeff...        │
│   ...                                                       │
│ """                                                          │
│                                                              │
│ State Update:                                                │
│   self._last_vehicle_key = "hector"                        │
│   self._last_vehicle_name = "MG Hector"                    │
│   self._last_vehicle_weight = 1650                         │
│   self._last_component_key = "brake"                       │
└─────────────────────────────────────────────────────────────┘
```

#### Phase 5: Stream A - Existing DB Retrieval (RAG)

```
Time: T0 + 2030ms
Location: agent.py, Line 486
Event: Vector Database Search

┌─────────────────────────────────────────────────────────────┐
│ Step 5.1: Call Vector DB Function                           │
│ existing_ideas, context = self.vector_db_func(              │
│   user_query, top_k=5                                       │
│ )                                                            │
│                                                              │
│ This calls: app.py → retrieve_context()                     │
└─────────────────────────────────────────────────────────────┘

Time: T0 + 2035ms
Location: app.py, Line ~1155 (retrieve_context function)
Event: Query Embedding

┌─────────────────────────────────────────────────────────────┐
│ Step 5.2: Encode Query                                       │
│ query_embedding = embedding_model.encode([user_query])      │
│   → Shape: (1, 384)                                          │
│   → Time: ~50-100ms                                          │
└─────────────────────────────────────────────────────────────┘

Time: T0 + 2135ms
Location: app.py, retrieve_context()
Event: FAISS Search

┌─────────────────────────────────────────────────────────────┐
│ Step 5.3: Vector Similarity Search                           │
│ distances, indices = faiss_index.search(                    │
│   query_embedding.astype('float32'),                        │
│   k=5                                                        │
│ )                                                            │
│                                                              │
│ Algorithm (FAISS IndexFlatL2):                               │
│   - Computes L2 distance to all vectors                     │
│   - Returns top 5 closest                                    │
│   - Time: ~1-5ms for 10,000 vectors                         │
│                                                              │
│ Result:                                                      │
│   indices = [1234, 5678, 9012, 3456, 7890]                 │
│   distances = [0.45, 0.52, 0.58, 0.61, 0.65]               │
└─────────────────────────────────────────────────────────────┘

Time: T0 + 2140ms
Location: app.py, retrieve_context()
Event: PostgreSQL Query

┌─────────────────────────────────────────────────────────────┐
│ Step 5.4: Database Lookup                                    │
│ conn = get_db_connection()                                  │
│ cursor = conn.cursor()                                      │
│ query = "SELECT * FROM ideas WHERE id IN (1234, 5678, ...)" │
│ cursor.execute(query)                                       │
│ rows = cursor.fetchall()                                    │
│                                                              │
│ Result:                                                      │
│   existing_ideas = [                                         │
│     {id: 1234, cost_reduction_idea: "...", ...},           │
│     {id: 5678, cost_reduction_idea: "...", ...},           │
│     ... (5 ideas)                                            │
│   ]                                                          │
└─────────────────────────────────────────────────────────────┘

Time: T0 + 2150ms
Location: agent.py, Line 492
Event: Image Assignment for Existing Ideas

┌─────────────────────────────────────────────────────────────┐
│ Step 5.5: Image Fallback for DB Ideas                       │
│ for idea in existing_ideas:                                 │
│   if not idea.get('mg_vehicle_image'):                     │
│     matched_img = self._get_fallback_mg_image(...)          │
│     idea['mg_vehicle_image'] = matched_img                  │
└─────────────────────────────────────────────────────────────┘
```

#### Phase 6: Stream B - AI Innovation Engine

```
Time: T0 + 2160ms
Location: agent.py, Line 506
Event: Start Innovation Engine

┌─────────────────────────────────────────────────────────────┐
│ Step 6.1: Call Innovation Engine                            │
│ new_ideas = self.run_innovation_engine(                     │
│   user_query, existing_ideas, global_used_images           │
│ )                                                            │
└─────────────────────────────────────────────────────────────┘

Time: T0 + 2165ms
Location: agent.py, Line 443
Event: Generate Ideas via LLM

┌─────────────────────────────────────────────────────────────┐
│ Step 6.2: LLM Generation                                     │
│ generated_ideas = self._generate_with_llm(                   │
│   user_query, existing_ideas, target_component              │
│ )                                                            │
└─────────────────────────────────────────────────────────────┘

Time: T0 + 2170ms
Location: agent.py, Line 513
Event: Build System Prompt

┌─────────────────────────────────────────────────────────────┐
│ Step 6.3: System Prompt Construction                        │
│ db_examples = "\n".join([                                    │
│   f"- Idea: {c.get('Cost Reduction Idea')}..."              │
│   for c in existing_ideas[:4]                               │
│ ])                                                           │
│                                                              │
│ engineering_context = self._resolve_engineering_context(...)│
│   → Already computed, uses cached state                     │
│                                                              │
│ system_prompt = f"""                                        │
│   {engineering_context}                                     │
│   You are a Chief Technical Officer...                      │
│   INSTRUCTIONS:                                             │
│   1. ANALYZE THE DATASET...                                 │
│   2. ENGINEERING COMPONENT FOCUS...                         │
│   3. DERIVE NEW IDEAS: Create 12 NEW ideas...              │
│   ...                                                       │
│ """                                                          │
└─────────────────────────────────────────────────────────────┘

Time: T0 + 2180ms
Location: agent.py, Line 540
Event: Build User Prompt

┌─────────────────────────────────────────────────────────────┐
│ Step 6.4: User Prompt Construction                          │
│ prompt = f"""                                               │
│   TARGET COMPONENT: "brake disc"                            │
│   USER QUERY: "Reduce weight of MG Hector Brake Disc"       │
│                                                              │
│   REFERENCE CONTEXT:                                        │
│   - Idea: Reduce rotor thickness...                         │
│   - Idea: Switch to composite material...                   │
│   ...                                                       │
│                                                              │
│   TASK:                                                    │
│   Construct 12 NEW, ENGINEER-LEVEL proposals for            │
│   "brake disc" on MG Hector.                               │
│   ...                                                       │
│ """                                                          │
└─────────────────────────────────────────────────────────────┘

Time: T0 + 2190ms
Location: agent.py, Line 571
Event: Call LLM API

┌─────────────────────────────────────────────────────────────┐
│ Step 6.5: Gemini API Call                                    │
│ raw_response = self.call_llm(prompt, system_prompt)         │
│   → _call_gemini(prompt, system_prompt)                     │
│     → genai.GenerativeModel("gemini-2.5-flash-lite")        │
│     → model.generate_content(full_prompt, config)           │
│                                                              │
│ Network I/O:                                                │
│   - Request sent to Google API                              │
│   - Processing time: ~5-15 seconds                           │
│   - Response received: JSON string with 12 ideas            │
│                                                              │
│ Response Format:                                            │
│   ```json                                                   │
│   [                                                         │
│     {                                                       │
│       "idea_id": "AI-GEN-01",                              │
│       "cost_reduction_idea": "Reduce brake disc thickness...",│
│       "way_forward": "Detailed engineering...",            │
│       "feasibility_score": 78,                             │
│       ...                                                   │
│     },                                                      │
│     ... (12 ideas)                                          │
│   ]                                                         │
│   ```                                                       │
└─────────────────────────────────────────────────────────────┘

Time: T0 + 12000ms (12 seconds later)
Location: agent.py, Line 576
Event: Parse JSON Response

┌─────────────────────────────────────────────────────────────┐
│ Step 6.6: JSON Parsing                                      │
│ clean_json = raw_response.replace("```json", "").replace("```", "")│
│ match = re.search(r'\[\s*\{.*\}\s*\]', clean_json, re.DOTALL)│
│ ideas = json.loads(match.group(0))                          │
│                                                              │
│ Result:                                                      │
│   generated_ideas = [                                       │
│     {idea_id: "AI-GEN-01", ...},                            │
│     {idea_id: "AI-GEN-02", ...},                            │
│     ... (12 ideas)                                          │
│   ]                                                          │
└─────────────────────────────────────────────────────────────┘

Time: T0 + 12010ms
Location: agent.py, Line 450
Event: Validate Ideas

┌─────────────────────────────────────────────────────────────┐
│ Step 6.7: Autonomous Validation                             │
│ validated_ideas = self._validate_and_filter_ideas(          │
│   generated_ideas,                                          │
│   target_component="brake disc",                            │
│   query=user_query,                                         │
│   min_score=25                                              │
│ )                                                            │
│                                                              │
│ For each of 12 ideas:                                       │
│   1. Normalize scores (0-100)                                │
│   2. Check scope (target tokens in text?)                   │
│   3. Check impossible patterns                              │
│   4. Apply physics heuristics:                              │
│      - Heavy vehicle brake risk?                            │
│        → vehicle_weight=1650 >= 1600 ✓                      │
│        → component="brake" ✓                                │
│        → text contains "reduce weight" ✓                    │
│        → text contains "disc" ✓                             │
│        → FLAG: heavy_brake_risk = True                      │
│   5. Scoring gate: min_score >= 25?                         │
│   6. Classify:                                              │
│      - If scores_ok AND no flags → "Auto-Approved"         │
│      - Else → "Needs Human Review"                          │
│                                                              │
│ Result:                                                      │
│   - 3 ideas: Auto-Approved                                  │
│   - 7 ideas: Needs Human Review (including brake risk flags)│
│   - 2 ideas: Rejected (scope/impossible)                    │
│                                                              │
│   validated_ideas = [                                       │
│     {validation_status: "Auto-Approved", ...},              │
│     {validation_status: "Needs Human Review",               │
│      validation_notes: "MG Hector: Heavy-vehicle brake...", │
│      ...},                                                   │
│     ... (8 ideas, top-scoring)                              │
│   ]                                                          │
└─────────────────────────────────────────────────────────────┘

Time: T0 + 12050ms
Location: agent.py, Line 458
Event: Image Assignment

┌─────────────────────────────────────────────────────────────┐
│ Step 6.8: Semantic Image Matching                           │
│ for idea in validated_ideas:                                │
│   search_query = idea.get('visual_prompt', target_component)│
│   mg_img_path, mg_label = self._evaluate_visual_match(      │
│     search_query, "mg", used_images                         │
│   )                                                          │
│                                                              │
│ Algorithm (for first idea):                                 │
│   1. Encode: "Close up of brake disc showing rotor"         │
│      → query_vec: (1, 384)                                  │
│   2. Cosine similarity:                                     │
│      scores = np.dot(query_vec, image_embeddings.T)         │
│      → scores: (N_images,)                                  │
│   3. Sort: top_indices = np.argsort(scores)[::-1]           │
│   4. Find first unused above threshold (0.35):              │
│      → idx=567, score=0.72, filename="120.jpg"              │
│   5. Return: "/static/images/mg/120.jpg", "Brake disc..."   │
│                                                              │
│   idea['mg_vehicle_image'] = "/static/images/mg/120.jpg"    │
│   idea['current_scenario_image'] = "/static/images/mg/120.jpg"│
│   used_images.add("120.jpg")                                │
│                                                              │
│ Repeat for all 8 ideas (ensuring uniqueness)                │
└─────────────────────────────────────────────────────────────┘
```

#### Phase 7: Stream C - Web Mining Engine

```
Time: T0 + 12100ms
Location: agent.py, Line 402
Event: Start Web Mining

┌─────────────────────────────────────────────────────────────┐
│ Step 7.1: Generate Search Queries                          │
│ search_queries = [                                          │
│   "automotive brake disc cost reduction... filetype:pdf",   │
│   "site:sae.org brake disc lightweighting MG Hector",      │
│   "automotive brake disc material substitution 2024 2025",  │
│   "reduce cost brake disc manufacturing...",               │
│   "heavy suv brake disc thermal management..."              │
│ ]                                                            │
└─────────────────────────────────────────────────────────────┘

Time: T0 + 12110ms
Location: agent.py, Line 426
Event: Parallel Web Search

┌─────────────────────────────────────────────────────────────┐
│ Step 7.2: ThreadPoolExecutor (3 workers)                    │
│ with ThreadPoolExecutor(max_workers=3) as executor:         │
│   futures = {                                                │
│     executor.submit(perform_web_search, q1, 3): q1,         │
│     executor.submit(perform_web_search, q2, 3): q2,         │
│     executor.submit(perform_web_search, q3, 3): q3,        │
│     executor.submit(perform_web_search, q4, 3): q4,         │
│     executor.submit(perform_web_search, q5, 3): q5          │
│   }                                                          │
│                                                              │
│ Thread Execution (tools.py, Line 405):                      │
│   For each query:                                            │
│     1. Sanitize query (max 400 chars)                       │
│     2. Random wait: 3-8 seconds (rate limiting)             │
│     3. DuckDuckGo search (DDGS().text())                    │
│     4. Format results: "[1] Title: Body (Source: URL)"      │
│     5. Return context string                                │
│                                                              │
│ Time: ~10-20 seconds (parallel, but rate-limited)            │
└─────────────────────────────────────────────────────────────┘

Time: T0 + 32000ms (32 seconds total)
Location: agent.py, Line 442
Event: Physics-Adapted Synthesis

┌─────────────────────────────────────────────────────────────┐
│ Step 7.3: LLM Synthesis                                      │
│ system_prompt = f"""                                        │
│   {engineering_context}  # Vehicle×Component constraints    │
│   You are a Senior Research Scientist...                   │
│   CRITICAL: If paper is for generic vehicle, ADAPT IT      │
│   to MG Hector using constraints...                        │
│ """                                                          │
│                                                              │
│ prompt = f"""                                               │
│   TARGET COMPONENT: "brake disc"                            │
│   ACADEMIC CONTEXT:                                         │
│   {raw_web_context[:18000]}  # Truncated to 18k chars      │
│   Output 4 Adapted Research Proposals as JSON.              │
│ """                                                          │
│                                                              │
│ raw_response = self.call_llm(prompt, system_prompt)        │
│   → Gemini API call (~5-10 seconds)                         │
│                                                              │
│ Result: 4 adapted research proposals                        │
└─────────────────────────────────────────────────────────────┘
```

#### Phase 8: Image Enrichment (VLM Engine)

```
Time: T0 + 42000ms
Location: agent.py, Line 586
Event: Enrich All Ideas with Images

┌─────────────────────────────────────────────────────────────┐
│ Step 8.1: VLM Enrichment for Each Stream                    │
│ if self.vlm:                                                │
│   existing_ideas = self._enrich_ideas_with_images(          │
│     existing_ideas, "Existing Database", used_images        │
│   )                                                          │
│   new_ideas = self._enrich_ideas_with_images(               │
│     new_ideas, "AI Innovation", used_images                │
│   )                                                          │
│   web_ideas = self._enrich_ideas_with_images(               │
│     web_ideas, "Web Source", used_images                    │
│   )                                                          │
└─────────────────────────────────────────────────────────────┘

Time: T0 + 42010ms
Location: agent.py, Line 599
Event: Process Single Idea (AI Innovation)

┌─────────────────────────────────────────────────────────────┐
│ Step 8.2: VLM Image Generation (for AI idea)               │
│ idea_text = "Reduce brake disc thickness..."                │
│ origin = "AI Innovation"                                    │
│ extra_context = {                                           │
│   "idea_text": idea_text,                                   │
│   "visual_prompt": "Close up of brake disc...",             │
│   "vehicle_name": "MG Hector",                              │
│   "component_key": "brake",                                 │
│   "mg_vehicle_image": "/static/images/mg/120.jpg"          │
│ }                                                            │
│                                                              │
│ images = vlm.get_images_for_idea(idea_text, origin, extra_context)│
└─────────────────────────────────────────────────────────────┘

Time: T0 + 42020ms
Location: vlm_engine.py, Line 35
Event: VLM Image Resolution

┌─────────────────────────────────────────────────────────────┐
│ Step 8.3: Current Scenario Image                            │
│ current_img = extra_context.get('mg_vehicle_image')          │
│   → "/static/images/mg/120.jpg" (already assigned)          │
│ images['current_scenario_image'] = current_img              │
└─────────────────────────────────────────────────────────────┘

Time: T0 + 42025ms
Location: vlm_engine.py, Line 73
Event: Proposal Image Generation (AI Innovation)

┌─────────────────────────────────────────────────────────────┐
│ Step 8.4: CAD Overlay Prompt Construction                   │
│ if origin == "AI Innovation":                                │
│   overlay_prompt = self._construct_overlay_prompt(          │
│     idea_text="Reduce brake disc thickness...",              │
│     vehicle_name="MG Hector",                               │
│     component_hint="brake disc"                             │
│   )                                                          │
│                                                              │
│ Result:                                                     │
│   "Technical engineering CAD overlay of brake disc for       │
│    MG Hector, highlighting the proposed change: Reduce     │
│    brake disc thickness. Schematic wireframe style,         │
│    mounting points highlighted in neon blue, white          │
│    background, clean annotations, no people, no branding."  │
└─────────────────────────────────────────────────────────────┘

Time: T0 + 42030ms
Location: vlm_engine.py, Line 79
Event: Pollinations API Call

┌─────────────────────────────────────────────────────────────┐
│ Step 8.5: AI Image Generation                               │
│ images['proposal_scenario_image'] =                          │
│   self._generate_cloud_image_pollinations(overlay_prompt)   │
│                                                              │
│ Algorithm:                                                  │
│   1. Sanitize prompt: quote() for URL encoding             │
│   2. Construct URL:                                         │
│      "https://image.pollinations.ai/prompt/{prompt}?nologo=true"│
│   3. HTTP GET request (timeout=20s)                          │
│   4. Save image to static/generated/ai_gen_xxx.jpg          │
│   5. Return path: "static/generated/ai_gen_xxx.jpg"         │
│                                                              │
│ Time: ~2-5 seconds per image                                │
└─────────────────────────────────────────────────────────────┘
```

#### Phase 9: Data Normalization & Formatting

```
Time: T0 + 47000ms
Location: agent.py, Line 763
Event: Normalize All Data

┌─────────────────────────────────────────────────────────────┐
│ Step 9.1: Normalize Each Stream                             │
│ normalized_existing = self._normalize_data(                 │
│   existing_ideas, "Existing DB"                              │
│ )                                                            │
│ normalized_new = self._normalize_data(                       │
│   new_ideas, "AI Innovation"                                │
│ )                                                            │
│ normalized_web = self._normalize_data(                       │
│   web_ideas, "World Wide Web"                               │
│ )                                                            │
│                                                              │
│ _normalize_data() Algorithm:                                │
│   For each idea dict:                                        │
│     - Map keys: snake_case → Title Case                     │
│     - Handle missing keys with defaults                      │
│     - Ensure all required fields present                    │
│     - Convert types (float, int, str)                       │
└─────────────────────────────────────────────────────────────┘

Time: T0 + 47050ms
Location: agent.py, Line 807
Event: Format Final Response

┌─────────────────────────────────────────────────────────────┐
│ Step 9.2: Response Text Generation                          │
│ response_text = self._format_final_response(                │
│   all_data, user_query                                      │
│ )                                                            │
│                                                              │
│ Result:                                                     │
│   "Analyzed query: 'Reduce weight of MG Hector Brake Disc'. │
│                                                              │
│   **Existing Database Matches (3):**                        │
│   - Reduce rotor thickness... (Status: OK)                  │
│   ...                                                       │
│                                                              │
│   **AI Generated Innovations (8):**                         │
│   - Reduce brake disc thickness...                          │
│   ...                                                       │
│                                                              │
│   **World Wide Web Insights (4):**                          │
│   - Composite brake disc material...                        │
│   ..."                                                       │
└─────────────────────────────────────────────────────────────┘

Time: T0 + 47100ms
Location: agent.py, Line 421
Event: Cache Results

┌─────────────────────────────────────────────────────────────┐
│ Step 9.3: Store Results                                     │
│ self._last_result_data = (                                   │
│   normalized_existing +                                     │
│   normalized_new +                                          │
│   normalized_web                                            │
│ )                                                            │
│   → Total: ~15-17 ideas                                     │
└─────────────────────────────────────────────────────────────┘
```

#### Phase 10: Response to Frontend

```
Time: T0 + 47110ms
Location: app.py, Line 4843
Event: Retrieve Structured Data

┌─────────────────────────────────────────────────────────────┐
│ Step 10.1: Get Cached Results                               │
│ table_data = vave_agent.get_last_result_data()              │
│   → Returns self._last_result_data                          │
└─────────────────────────────────────────────────────────────┘

Time: T0 + 47120ms
Location: app.py, Line 4846
Event: Image Path Conversion

┌─────────────────────────────────────────────────────────────┐
│ Step 10.2: Convert Paths to URLs                            │
│ table_data = convert_image_paths_to_urls(table_data)        │
│                                                              │
│ Algorithm:                                                  │
│   For each idea:                                            │
│     - "static/images/mg/120.jpg" → "/static/images/mg/120.jpg"│
│     - Ensure all image fields have valid URLs               │
└─────────────────────────────────────────────────────────────┘

Time: T0 + 47130ms
Location: app.py, Line 4871
Event: Return JSON Response

┌─────────────────────────────────────────────────────────────┐
│ Step 10.3: HTTP Response                                     │
│ return {                                                     │
│   "success": True,                                           │
│   "response_text": response_text,                           │
│   "table_data": table_data,  # List of 15-17 idea dicts    │
│   "image_urls": [],                                          │
│   "excel_url": "",                                           │
│   "ppt_url": ""                                              │
│ }                                                            │
│                                                              │
│ Flask serializes to JSON and sends HTTP 200 response        │
└─────────────────────────────────────────────────────────────┘

Time: T0 + 47140ms
Location: Frontend (app.js)
Event: Render Results

┌─────────────────────────────────────────────────────────────┐
│ Step 10.4: Frontend Rendering                               │
│ 1. Display response_text in chat area                        │
│ 2. Populate table_data into Analysis Results table           │
│ 3. Render images in table cells (lazy loading)              │
│ 4. Enable Excel/PPT export buttons                           │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Total Execution Time Breakdown

```
Phase 1: HTTP Request Reception          ~10ms
Phase 2: Agent Initialization            ~2000ms (image embeddings)
Phase 3: Target Extraction               ~5ms
Phase 4: Context Resolution               ~10ms
Phase 5: Stream A (RAG)                   ~150ms
Phase 6: Stream B (AI Innovation)         ~10000ms (LLM call)
Phase 7: Stream C (Web Mining)            ~20000ms (web search + LLM)
Phase 8: Image Enrichment                 ~5000ms (VLM generation)
Phase 9: Data Normalization               ~50ms
Phase 10: Response Formatting             ~30ms
─────────────────────────────────────────────────
Total:                                    ~47 seconds
```

**Optimization Opportunities**:
- Stream B and C can run in parallel (currently sequential)
- Image enrichment can be parallelized
- VLM generation can be cached

---

## 4. AI Agent Architecture - Comprehensive

### 4.1 VAVEAgent: Master Orchestrator Architecture

**File**: `agent.py`, Class: `VAVEAgent` (Lines 92-994)

#### 4.1.1 Overall Agent Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VAVEAgent (Master Orchestrator)                      │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  INITIALIZATION LAYER                                                 │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │  │
│  │  │ Image Index  │  │ VLM Engine   │  │ Dependencies │              │  │
│  │  │ Loader       │  │ Initializer  │  │ Validator    │              │  │
│  │  │ _load_       │  │ (conditional)│  │              │              │  │
│  │  │ image_       │  │              │  │              │              │  │
│  │  │ index()      │  │              │  │              │              │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  MAIN ORCHESTRATION LAYER (run())                                    │  │
│  │  ┌──────────────────────────────────────────────────────────────┐   │  │
│  │  │ 1. Component Extractor                                        │   │  │
│  │  │    _smart_extract_target()                                     │   │  │
│  │  │    → Extracts "brake disc" from query                          │   │  │
│  │  └──────────────────────────────────────────────────────────────┘   │  │
│  │                                                                      │  │
│  │  ┌──────────────────────────────────────────────────────────────┐   │  │
│  │  │ 2. Three Parallel Streams (Executed Sequentially)             │   │  │
│  │  │    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │   │  │
│  │  │    │ Stream A     │  │ Stream B     │  │ Stream C     │     │   │  │
│  │  │    │ RAG Engine   │  │ Innovation   │  │ Web Mining   │     │   │  │
│  │  │    │              │  │ Engine       │  │ Engine       │     │   │  │
│  │  │    └──────────────┘  └──────────────┘  └──────────────┘     │   │  │
│  │  └──────────────────────────────────────────────────────────────┘   │  │
│  │                                                                      │  │
│  │  ┌──────────────────────────────────────────────────────────────┐   │  │
│  │  │ 3. Image Enrichment Layer                                     │   │  │
│  │  │    _enrich_ideas_with_images()                                 │   │  │
│  │  │    → Calls VLM Engine for each idea                            │   │  │
│  │  └──────────────────────────────────────────────────────────────┘   │  │
│  │                                                                      │  │
│  │  ┌──────────────────────────────────────────────────────────────┐   │  │
│  │  │ 4. Data Normalization & Formatting                            │   │  │
│  │  │    _normalize_data() + _format_final_response()               │   │  │
│  │  └──────────────────────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  SUPPORTING AGENTS/ENGINES                                           │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │  │
│  │  │ Context      │  │ LLM Caller   │  │ Image        │              │  │
│  │  │ Resolver     │  │ Agent        │  │ Matcher      │              │  │
│  │  │ _resolve_    │  │ call_llm()   │  │ _evaluate_   │              │  │
│  │  │ engineering_ │  │ _call_       │  │ visual_      │              │  │
│  │  │ context()    │  │ gemini()     │  │ match()      │              │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │  │
│  │                                                                      │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │  │
│  │  │ Validation   │  │ JSON Parser  │  │ Data         │              │  │
│  │  │ Agent        │  │ _parse_llm_  │  │ Normalizer   │              │  │
│  │  │ _validate_   │  │ json()       │  │ _normalize_   │              │  │
│  │  │ and_filter_  │  │              │  │ data()       │              │  │
│  │  │ ideas()      │  │              │  │              │              │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 4.1.2 Agent Component Hierarchy

```
VAVEAgent (Master)
│
├─> Component Extractor Agent
│   └─> _smart_extract_target()
│       Role: Extract meaningful component name from natural language
│       Input: "Reduce weight of MG Hector Brake Disc"
│       Output: "brake disc"
│
├─> Context Resolver Agent
│   └─> _resolve_engineering_context()
│       Role: Resolve Vehicle×Component matrix constraints
│       Input: User query
│       Output: Engineering context string + state variables
│
├─> Stream A: RAG Retrieval Agent
│   └─> vector_db_func() [External: app.py]
│       Role: Semantic search existing ideas
│       Input: Query, top_k=5
│       Output: List[Dict] existing ideas
│
├─> Stream B: Innovation Engine Agent
│   │
│   ├─> Idea Generator Agent
│   │   └─> _generate_with_llm()
│   │       Role: Generate 12 new ideas via LLM
│   │       Input: Query, context, target component
│   │       Output: 12 raw ideas (JSON)
│   │
│   ├─> Validation Agent
│   │   └─> _validate_and_filter_ideas()
│   │       Role: Filter and score ideas
│   │       Input: 12 raw ideas
│   │       Output: 6-8 validated ideas
│   │
│   └─> Image Assignment Agent
│       └─> _evaluate_visual_match()
│           Role: Assign unique images to ideas
│           Input: Idea text, used_images set
│           Output: Image path + label
│
├─> Stream C: Web Mining Engine Agent
│   │
│   ├─> Query Generator Agent
│   │   └─> run_web_mining_engine()
│   │       Role: Generate academic search queries
│   │       Input: Target component, vehicle name
│   │       Output: 5 search queries
│   │
│   ├─> Web Search Agent
│   │   └─> perform_web_search() [tools.py]
│   │       Role: Execute parallel web searches
│   │       Input: Search queries
│   │       Output: Aggregated web context
│   │
│   └─> Research Synthesis Agent
│       └─> call_llm() with physics-adapted prompt
│           Role: Synthesize research into ideas
│           Input: Web context, engineering constraints
│           Output: 4 adapted research proposals
│
├─> LLM Caller Agent
│   ├─> call_llm() [Unified Interface]
│   ├─> _call_gemini() [Primary]
│   └─> _call_ollama() [Fallback]
│       Role: Abstract LLM provider selection
│       Input: Prompt, system_prompt
│       Output: Raw LLM response
│
├─> JSON Parser Agent
│   └─> _parse_llm_json()
│       Role: Extract JSON from LLM response
│       Input: Raw response string
│       Output: Parsed List[Dict]
│
├─> Image Matcher Agent
│   ├─> _evaluate_visual_match() [Semantic]
│   └─> _get_smart_image() [Keyword Fallback]
│       Role: Find best matching image
│       Input: Query text, folder, used_images
│       Output: Image path + match label
│
├─> VLM Integration Agent
│   └─> _enrich_ideas_with_images()
│       Role: Enrich ideas with generated images
│       Input: Ideas list, origin, used_images
│       Output: Ideas with image paths
│
└─> Data Formatter Agent
    ├─> _normalize_data()
    └─> _format_final_response()
        Role: Standardize and format output
        Input: Raw idea lists
        Output: Normalized data + formatted text
```

### 4.2 Detailed Agent Architectures

#### 4.2.1 Component Extractor Agent

**Location**: `agent.py`, Lines 599-623

**Architecture**:
```
┌─────────────────────────────────────────────────────────────┐
│  Component Extractor Agent                                   │
│  Method: _smart_extract_target(query: str) -> str           │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Input: "Reduce weight of MG Hector Brake Disc"     │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Step 1: Tokenization                                │   │
│  │  re.findall(r'\b[a-zA-Z0-9]+\b', query.lower())     │   │
│  │  → ["reduce", "weight", "of", "mg", "hector",        │   │
│  │     "brake", "disc"]                                  │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Step 2: Stopword Filtering                          │   │
│  │  stopwords = {'reduce', 'weight', 'for', 'the', ...} │   │
│  │  → Filtered: ["mg", "hector", "brake", "disc"]      │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Step 3: Meaningful Word Selection                   │   │
│  │  meaningful_words = [w for w in words                │   │
│  │                      if w not in stopwords            │   │
│  │                      and len(w) > 3]                 │   │
│  │  → ["hector", "brake", "disc"]                       │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Step 4: Last N Words (Heuristic)                    │   │
│  │  return " ".join(meaningful_words[-2:])              │   │
│  │  → "brake disc"                                       │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  Output: "brake disc"                                        │
└─────────────────────────────────────────────────────────────┘
```

**Role**: Extracts the target component from natural language queries, ignoring stopwords and focusing on meaningful technical terms.

**Algorithm Complexity**:
- Time: O(M) where M = query length
- Space: O(M) for token list

**Design Pattern**: Heuristic-based extraction (not ML-based, but effective for automotive domain)

---

#### 4.2.2 Context Resolver Agent

**Location**: `agent.py`, Lines 321-392

**Architecture**:
```
┌─────────────────────────────────────────────────────────────┐
│  Context Resolver Agent                                      │
│  Method: _resolve_engineering_context(query: str) -> str     │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Input: "Reduce weight of MG Hector Brake Disc"     │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Phase 1: Vehicle Detection                          │   │
│  │  ┌──────────────────────────────────────────────┐   │   │
│  │  │ vehicle_aliases = {                          │   │   │
│  │  │   "hector": ["mg hector", "hector"],         │   │   │
│  │  │   "astor": ["mg astor", "astor"],            │   │   │
│  │  │   "zsev": ["zs ev", "mg zs ev", ...],        │   │   │
│  │  │   ...                                         │   │   │
│  │  │ }                                             │   │   │
│  │  │                                               │   │   │
│  │  │ for key, aliases in vehicle_aliases.items(): │   │   │
│  │  │   if any(a in query.lower() for a in aliases):│   │   │
│  │  │     vehicle_key = key  # "hector"            │   │   │
│  │  │     break                                     │   │   │
│  │  └──────────────────────────────────────────────┘   │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Phase 2: Component Detection                        │   │
│  │  ┌──────────────────────────────────────────────┐   │   │
│  │  │ component_aliases = {                        │   │   │
│  │  │   "brake": ["brake", "brakes", "disc", ...], │   │   │
│  │  │   "hvac": ["hvac", "blower", "ac", ...],      │   │   │
│  │  │   "body": ["biw", "body", "door", ...],      │   │   │
│  │  │ }                                             │   │   │
│  │  │                                               │   │   │
│  │  │ for key, aliases in component_aliases.items():│   │   │
│  │  │   if any(a in query.lower() for a in aliases):│   │   │
│  │  │     component_key = key  # "brake"           │   │   │
│  │  │     break                                     │   │   │
│  │  └──────────────────────────────────────────────┘   │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Phase 3: Dictionary Lookup                         │   │
│  │  vehicle = VEHICLE_DB.get(vehicle_key, default)     │   │
│  │  component = COMPONENT_DB.get(component_key, default)│   │
│  │                                                      │   │
│  │  vehicle = {                                        │   │
│  │    "name": "MG Hector",                             │   │
│  │    "weight_kg": 1650,                               │   │
│  │    "type": "ICE",                                   │   │
│  │    "constraints": "High Kinetic Energy..."          │   │
│  │  }                                                  │   │
│  │                                                      │   │
│  │  component = {                                      │   │
│  │    "system": "Braking",                             │   │
│  │    "physics": "Thermal Mass, Friction...",          │   │
│  │    "regulations": "FMVSS 105, ECE R13.",           │   │
│  │    "focus": "Rotor thickness, Caliper material..."  │   │
│  │  }                                                  │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Phase 4: Special Rule Application                  │   │
│  │  if vehicle["type"] == "EV" and component_key == "hvac":│   │
│  │    extra_rules = "EV×HVAC RULE: Range impact..."     │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Phase 5: Context String Construction                │   │
│  │  context = f"""                                       │   │
│  │    VEHICLE: {vehicle['name']} | WEIGHT: {vehicle['weight_kg']}│   │
│  │    VEHICLE CONSTRAINTS: {vehicle['constraints']}     │   │
│  │    COMPONENT SYSTEM: {component['system']}            │   │
│  │    COMPONENT PHYSICS: {component['physics']}         │   │
│  │    COMPONENT REGULATIONS: {component['regulations']} │   │
│  │    COMPONENT FOCUS: {component['focus']}             │   │
│  │    {extra_rules}                                     │   │
│  │  """                                                 │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Phase 6: State Update                               │   │
│  │  self._last_vehicle_key = "hector"                    │   │
│  │  self._last_vehicle_name = "MG Hector"               │   │
│  │  self._last_vehicle_weight = 1650                    │   │
│  │  self._last_component_key = "brake"                  │   │
│  │  self._last_vehicle_type = "ICE"                      │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Output: Engineering Context String                   │   │
│  │  "VEHICLE: MG Hector | WEIGHT: 1650 | ..."           │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Role**: Resolves the Vehicle×Component matrix to generate physics-informed engineering constraints that prevent physically impossible suggestions.

**Key Features**:
- **Pattern Matching**: Fast keyword-based detection (O(M*K))
- **Default Fallback**: Defaults to "hector" + "general" if no match
- **State Caching**: Stores resolved values for downstream use
- **Special Rules**: EV×HVAC range impact constraints

**Design Pattern**: Strategy Pattern (different resolution strategies per vehicle/component)

---

#### 4.2.3 Innovation Engine Agent

**Location**: `agent.py`, Lines 544-598

**Architecture**:
```
┌─────────────────────────────────────────────────────────────┐
│  Innovation Engine Agent                                     │
│  Method: run_innovation_engine(query, context_ideas, used_images)│
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Input:                                               │   │
│  │  - query: "Reduce weight of MG Hector Brake Disc"    │   │
│  │  - context_ideas: [5 existing ideas from DB]        │   │
│  │  - used_images: set()                                │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Step 1: Extract Target Component                    │   │
│  │  target_component = _smart_extract_target(query)     │   │
│  │  → "brake disc"                                       │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Step 2: Idea Generator Agent                        │   │
│  │  generated_ideas = _generate_with_llm(              │   │
│  │    query, context_ideas, target_component            │   │
│  │  )                                                    │   │
│  │  → 12 raw ideas (JSON)                               │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Step 3: Validation Agent                            │   │
│  │  validated_ideas = _validate_and_filter_ideas(       │   │
│  │    generated_ideas,                                  │   │
│  │    target_component="brake disc",                    │   │
│  │    query=query,                                      │   │
│  │    min_score=25                                      │   │
│  │  )                                                    │   │
│  │  → 6-8 validated ideas                                │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Step 4: Image Assignment Agent                      │   │
│  │  for idea in validated_ideas:                         │   │
│  │    search_query = idea.get('visual_prompt')          │   │
│  │    mg_img, label = _evaluate_visual_match(           │   │
│  │      search_query, "mg", used_images                │   │
│  │    )                                                  │   │
│  │    idea['mg_vehicle_image'] = mg_img                  │   │
│  │    idea['current_scenario_image'] = mg_img           │   │
│  │    used_images.add(filename)                          │   │
│  │                                                       │   │
│  │    comp_img, _ = _evaluate_visual_match(              │   │
│  │      search_query, "competitor", used_images         │   │
│  │    )                                                  │   │
│  │    idea['competitor_image'] = comp_img                │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Output: List[Dict] (6-8 validated ideas with images)│   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Role**: Generates new, innovative cost reduction ideas using LLM, validates them against physics constraints, and assigns contextually relevant images.

**Sub-Agents**:

**A. Idea Generator Agent** (`_generate_with_llm`)
- **Input**: Query, context ideas (4 examples), target component
- **Process**:
  1. Builds system prompt with Vehicle×Component context
  2. Constructs user prompt with reference examples
  3. Calls LLM (Gemini) to generate 12 ideas
  4. Parses JSON response
- **Output**: 12 raw idea dictionaries

**B. Validation Agent** (`_validate_and_filter_ideas`)
- **Input**: 12 raw ideas, target component, query
- **Process**:
  1. Normalizes scores (0-100)
  2. Scope validation (target tokens in text?)
  3. Impossible pattern detection
  4. Physics heuristics (heavy vehicle brake risk, EV HVAC range impact)
  5. Scoring gate (min_score >= 25)
  6. Classification (Auto-Approved vs Needs Review)
- **Output**: 6-8 validated ideas

**C. Image Assignment Agent** (`_evaluate_visual_match`)
- **Input**: Visual prompt, folder name, used_images set
- **Process**:
  1. Semantic embedding of prompt
  2. Cosine similarity search
  3. Threshold filtering (0.35)
  4. Uniqueness enforcement
- **Output**: Image path + match label

**Design Pattern**: Pipeline Pattern (Generator → Validator → Enricher)

---

#### 4.2.4 Web Mining Engine Agent

**Location**: `agent.py`, Lines 408-474

**Architecture**:
```
┌─────────────────────────────────────────────────────────────┐
│  Web Mining Engine Agent                                     │
│  Method: run_web_mining_engine(query, target_component)     │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Step 1: Context Resolution                          │   │
│  │  engineering_context = _resolve_engineering_context(query)│   │
│  │  vehicle_name = self._last_vehicle_name              │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Step 2: Query Generator Agent                       │   │
│  │  search_queries = [                                   │   │
│  │    f"automotive {target_component} cost reduction...",│   │
│  │    f"site:sae.org {target_component} lightweighting {vehicle_name}",│   │
│  │    f"automotive {target_component} material substitution...",│   │
│  │    f"reduce cost {target_component} manufacturing...",│   │
│  │    f"heavy suv {target_component} thermal management..."│   │
│  │  ]                                                    │   │
│  │  → 5 specialized academic/industry queries            │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Step 3: Parallel Web Search Agent                   │   │
│  │  with ThreadPoolExecutor(max_workers=3) as executor:  │   │
│  │    futures = {                                        │   │
│  │      executor.submit(perform_web_search, q, 3): q    │   │
│  │      for q in search_queries                          │   │
│  │    }                                                  │   │
│  │                                                       │   │
│  │    for future in as_completed(futures):              │   │
│  │      result = future.result()                        │   │
│  │      raw_web_context += f"--- Source ({q}) ---\n{result}"│   │
│  │                                                       │   │
│  │  → Aggregated web context (18k chars max)           │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Step 4: Research Synthesis Agent                    │   │
│  │  system_prompt = f"""                                │   │
│  │    {engineering_context}                             │   │
│  │    You are a Senior Research Scientist...            │   │
│  │    CRITICAL: If paper is for generic vehicle,        │   │
│  │    ADAPT IT to {vehicle_name} using constraints...  │   │
│  │  """                                                 │   │
│  │                                                       │   │
│  │  prompt = f"""                                       │   │
│  │    TARGET COMPONENT: "{target_component}"            │   │
│  │    ACADEMIC CONTEXT:                                 │   │
│  │    {raw_web_context[:18000]}                          │   │
│  │    Output 4 Adapted Research Proposals as JSON.      │   │
│  │  """                                                 │   │
│  │                                                       │   │
│  │  raw_response = call_llm(prompt, system_prompt)      │   │
│  │  → LLM generates 4 adapted research proposals         │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Step 5: JSON Parsing & Metadata Attachment          │   │
│  │  ideas = _parse_llm_json(raw_response)                │   │
│  │  for idea in ideas:                                   │   │
│  │    idea['vehicle_name'] = vehicle_name               │   │
│  │    idea['vehicle_key'] = self._last_vehicle_key      │   │
│  │    idea['component_key'] = self._last_component_key   │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Output: List[Dict] (4 adapted research proposals)   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Role**: Mines web research, adapts generic findings to vehicle-specific constraints, and synthesizes research-grade proposals.

**Key Innovation**: **Adaptation Logic** - Instead of rejecting generic research, the agent rewrites it to fit vehicle constraints (e.g., "Plastic pedal" → Rejected for ROI, "Composite reinforcement" → Accepted as material tech transfer).

**Sub-Agents**:

**A. Query Generator Agent**
- Generates 5 specialized search queries targeting:
  - PDF academic papers
  - SAE.org case studies
  - Industry trends (2024-2025)
  - Manufacturing optimization
  - Thermal management

**B. Parallel Web Search Agent**
- Uses ThreadPoolExecutor (3 workers)
- Calls `perform_web_search()` from `tools.py`
- Aggregates results with source attribution
- Handles rate limiting (3-8s random delay per query)

**C. Research Synthesis Agent**
- Injects Vehicle×Component constraints into LLM prompt
- Instructs LLM to **adapt** generic research
- Generates 4 research proposals with feasibility scores

**Design Pattern**: Producer-Consumer Pattern (Query Generator → Web Search → Synthesis)

---

#### 4.2.5 Validation Agent

**Location**: `agent.py`, Lines 792-916

**Architecture**:
```
┌─────────────────────────────────────────────────────────────┐
│  Validation Agent                                           │
│  Method: _validate_and_filter_ideas(ideas, target_component, │
│                                   query, min_score=25)       │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Input: 12 raw ideas from LLM                         │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Phase 1: Pre-processing                              │   │
│  │  - Resolve engineering context (for physics heuristics)│   │
│  │  - Extract target tokens                              │   │
│  │  - Initialize validated[] and needs_review[] lists   │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Phase 2: Per-Idea Validation Loop                    │   │
│  │  for idea in ideas:                                   │   │
│  │    ┌──────────────────────────────────────────────┐ │   │
│  │    │ 2.1: Score Normalization                     │ │   │
│  │    │   for key in score_keys:                      │ │   │
│  │    │     val = int(float(idea.get(key, 0)))       │ │   │
│  │    │     idea[key] = max(0, min(100, val))         │ │   │
│  │    └──────────────────────────────────────────────┘ │   │
│  │    ┌──────────────────────────────────────────────┐ │   │
│  │    │ 2.2: Scope Validation                        │ │   │
│  │    │   text_blob = idea['cost_reduction_idea'] +   │ │   │
│  │    │              idea['way_forward']              │ │   │
│  │    │   scope_ok = any(tok in text_blob            │ │   │
│  │    │                  for tok in target_tokens)   │ │   │
│  │    │   if not scope_ok:                           │ │   │
│  │    │     → Reject (Out of Scope)                  │ │   │
│  │    └──────────────────────────────────────────────┘ │   │
│  │    ┌──────────────────────────────────────────────┐ │   │
│  │    │ 2.3: Impossible Pattern Detection            │ │   │
│  │    │   patterns = [                                │ │   │
│  │    │     r"100%\s*weight\s*reduction",            │ │   │
│  │    │     r"zero\s*cost\s*manufacturing",           │ │   │
│  │    │     ...                                       │ │   │
│  │    │   ]                                           │ │   │
│  │    │   impossible = any(re.search(p, text_blob)     │ │   │
│  │    │                    for p in patterns)        │ │   │
│  │    │   if impossible:                              │ │   │
│  │    │     → Reject (Physically Impossible)         │ │   │
│  │    └──────────────────────────────────────────────┘ │   │
│  │    ┌──────────────────────────────────────────────┐ │   │
│  │    │ 2.4: Physics-Informed Heuristics             │ │   │
│  │    │   # Heavy Vehicle Brake Risk                  │ │   │
│  │    │   if (component_key == "brake" and            │ │   │
│  │    │       vehicle_weight >= 1600 and             │ │   │
│  │    │       "reduce weight" in text_blob and        │ │   │
│  │    │       "disc" in text_blob):                  │ │   │
│  │    │     heavy_brake_risk = True                   │ │   │
│  │    │                                                 │ │   │
│  │    │   # EV×HVAC Range Impact                      │ │   │
│  │    │   if (vehicle_type == "EV" and                │ │   │
│  │    │       component_key == "hvac" and              │ │   │
│  │    │       "increase blower" in text_blob):        │ │   │
│  │    │     ev_hvac_range_risk = True                 │ │   │
│  │    └──────────────────────────────────────────────┘ │   │
│  │    ┌──────────────────────────────────────────────┐ │   │
│  │    │ 2.5: Scoring Gate                            │ │   │
│  │    │   min_score = min(idea.get(k) for k in score_keys)│   │
│  │    │   scores_ok = min_score >= min_score          │ │   │
│  │    └──────────────────────────────────────────────┘ │   │
│  │    ┌──────────────────────────────────────────────┐ │   │
│  │    │ 2.6: Classification                         │ │   │
│  │    │   if scores_ok and not flags:               │ │   │
│  │    │     idea['validation_status'] = "Auto-Approved"│   │
│  │    │     validated.append(idea)                    │ │   │
│  │    │   else:                                      │ │   │
│  │    │     idea['validation_status'] = "Needs Human Review"│   │
│  │    │     idea['validation_notes'] = explanation   │ │   │
│  │    │     needs_review.append(idea)                 │ │   │
│  │    └──────────────────────────────────────────────┘ │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Phase 3: Merge & Limit                               │   │
│  │  final_list = validated + needs_review                │   │
│  │  return final_list[:8]  # Top 8 ideas                  │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  Output: List[Dict] (6-8 validated ideas)                    │
└─────────────────────────────────────────────────────────────┘
```

**Role**: Autonomous validation engine that filters ideas based on:
1. **Scope**: Idea must target the requested component
2. **Physical Possibility**: Rejects impossible claims
3. **Physics Heuristics**: Flags risky proposals (heavy vehicle brake thinning, EV HVAC power increase)
4. **Scoring Threshold**: Minimum 25/100 across all categories

**Validation Criteria**:

| Criterion | Check | Action |
|-----------|-------|--------|
| Scope | Target tokens in idea text? | Reject if No |
| Impossible Patterns | "100% reduction", "zero cost"? | Reject if Yes |
| Heavy Vehicle Brake | Weight >= 1600kg + brake + "reduce weight" + "disc"? | Flag for Review |
| EV×HVAC Range | EV + HVAC + "increase power"? | Flag for Review |
| Scoring | min_score >= 25? | Auto-Approve if Yes + no flags |

**Design Pattern**: Chain of Responsibility (each validation step can reject/flag/approve)

---

#### 4.2.6 LLM Caller Agent

**Location**: `agent.py`, Lines 234-308

**Architecture**:
```
┌─────────────────────────────────────────────────────────────┐
│  LLM Caller Agent                                            │
│  Method: call_llm(prompt, system_prompt) -> str               │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Input: prompt, system_prompt                       │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Step 1: Provider Selection                          │   │
│  │  provider = os.getenv("LLM_PROVIDER", "gemini")     │   │
│  │                                                       │   │
│  │  if provider == "gemini":                            │   │
│  │    → _call_gemini()                                  │   │
│  │  elif provider == "ollama":                           │   │
│  │    → _call_ollama()                                  │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Step 2: Gemini Call (Primary)                      │   │
│  │  _call_gemini(prompt, system_prompt)                 │   │
│  │    ┌──────────────────────────────────────────────┐ │   │
│  │    │ 2.1: Model Fallback Chain                     │ │   │
│  │    │   candidate_models = [                        │ │   │
│  │    │     "gemini-2.5-flash-lite",  # Try first     │ │   │
│  │    │     "gemini-2.5-flash",       # Fallback      │ │   │
│  │    │     "gemini-3-flash"          # Last resort   │ │   │
│  │    │   ]                                           │ │   │
│  │    │                                               │ │   │
│  │    │   for model_name in candidate_models:         │ │   │
│  │    │     try:                                      │ │   │
│  │    │       model = genai.GenerativeModel(model_name)│ │   │
│  │    │       full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"│   │
│  │    │       config = GenerationConfig(temperature=0.7)│   │
│  │    │       response = model.generate_content(full_prompt, config)│   │
│  │    │       if response.text:                        │ │   │
│  │    │         return response.text                   │ │   │
│  │    │     except Exception as e:                     │ │   │
│  │    │       if "429" in str(e):  # Rate limit        │ │   │
│  │    │         time.sleep(5)                          │ │   │
│  │    │         continue                               │ │   │
│  │    │       elif "404" in str(e):  # Model not found │ │   │
│  │    │         continue                               │ │   │
│  │    │       else:                                    │ │   │
│  │    │         logger.error(...)                      │ │   │
│  │    │         continue                               │ │   │
│  │    └──────────────────────────────────────────────┘ │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Output: Raw LLM response string (JSON or text)      │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Role**: Unified interface for LLM calls with automatic fallback and error handling.

**Features**:
- **Provider Abstraction**: Supports Gemini (primary) and Ollama (fallback)
- **Model Fallback Chain**: Tries 3 models in order (fastest to slowest)
- **Rate Limit Handling**: Sleeps 5s on 429 errors, retries
- **Error Recovery**: Continues to next model on failure

**Design Pattern**: Strategy Pattern (different LLM providers) + Chain of Responsibility (model fallback)

---

#### 4.2.7 Image Matcher Agent

**Location**: `agent.py`, Lines 148-191

**Architecture**:
```
┌─────────────────────────────────────────────────────────────┐
│  Image Matcher Agent                                         │
│  Method: _evaluate_visual_match(query_text, folder_name,    │
│                                used_images, threshold=0.35) │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Input:                                               │   │
│  │  - query_text: "Close up of brake disc showing rotor"│   │
│  │  - folder_name: "mg"                                  │   │
│  │  - used_images: {"120.jpg", "85.jpg"}                │   │
│  │  - threshold: 0.35                                    │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Step 1: Fallback Check                               │   │
│  │  if image_embeddings is None:                        │   │
│  │    → _get_smart_image() (keyword matching)           │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Step 2: Query Embedding                             │   │
│  │  query_vec = sentence_model.encode([query_text])     │   │
│  │    → Shape: (1, 384)                                 │   │
│  │  query_norm = query_vec / ||query_vec||              │   │
│  │    → L2-normalized for cosine similarity             │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Step 3: Cosine Similarity Computation                │   │
│  │  scores = np.dot(query_norm, image_embeddings.T)      │   │
│  │    → query_norm: (1, 384)                            │   │
│  │    → image_embeddings: (N, 384)                     │   │
│  │    → scores: (1, N) = cosine similarities           │   │
│  │                                                       │   │
│  │  Example scores:                                      │   │
│  │    [0.72, 0.65, 0.58, 0.45, 0.32, ...]             │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Step 4: Sort by Similarity                          │   │
│  │  top_indices = np.argsort(scores)[::-1]             │   │
│  │    → [567, 234, 891, 123, ...]  # Highest first     │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Step 5: Find First Valid Match                      │   │
│  │  for idx in top_indices:                             │   │
│  │    score = scores[idx]                                │   │
│  │    if score < threshold:  # 0.35                     │   │
│  │      break  # Early termination                      │   │
│  │                                                       │   │
│  │    filename = image_filenames[idx]                   │   │
│  │    if filename not in used_images:                   │   │
│  │      matched_label = image_index[filename]           │   │
│  │      return f"/static/images/{folder}/{filename}",    │   │
│  │             matched_label                             │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Output: (image_path, matched_label)                 │   │
│  │  Example: ("/static/images/mg/120.jpg",               │   │
│  │            "Silver alloy wheel rim showing brake caliper")│   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Role**: Semantic image matching using cosine similarity to find the most contextually relevant image for each idea.

**Algorithm**:
1. **Embedding**: Encode query text to 384-dim vector
2. **Similarity**: Compute cosine similarity with all pre-computed image embeddings
3. **Ranking**: Sort by similarity (descending)
4. **Filtering**: Apply threshold (0.35) and uniqueness check
5. **Return**: First valid match

**Fallback**: If embeddings unavailable, uses keyword-based matching (`_get_smart_image`)

**Design Pattern**: Template Method (semantic matching with keyword fallback)

---

#### 4.2.8 JSON Parser Agent

**Location**: `agent.py`, Lines 309-320

**Architecture**:
```
┌─────────────────────────────────────────────────────────────┐
│  JSON Parser Agent                                           │
│  Method: _parse_llm_json(raw_response) -> List[Dict]        │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Input: Raw LLM response string                      │   │
│  │  Example:                                            │   │
│  │    "Here are the ideas:\n```json\n[{...}]\n```\n..." │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Step 1: Clean Markdown Markers                      │   │
│  │  clean_json = raw_response                           │   │
│  │    .replace("```json", "")                            │   │
│  │    .replace("```", "")                                │   │
│  │    .strip()                                           │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Step 2: Extract JSON Array via Regex               │   │
│  │  match = re.search(r'\[\s*\{.*\}\s*\]',              │   │
│  │                    clean_json,                       │   │
│  │                    re.DOTALL)                         │   │
│  │    → Finds first JSON array in text                  │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Step 3: Parse JSON                                   │   │
│  │  if match:                                            │   │
│  │    ideas = json.loads(match.group(0))                 │   │
│  │    return ideas                                       │   │
│  │  else:                                                │   │
│  │    return []                                          │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Output: List[Dict] (parsed ideas)                   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Role**: Robustly extracts JSON arrays from LLM responses that may contain markdown formatting, explanatory text, or errors.

**Error Handling**:
- Handles markdown code blocks (```json ... ```)
- Extracts JSON even if surrounded by text
- Returns empty list on parse failure (graceful degradation)

**Design Pattern**: Adapter Pattern (adapts LLM output format to internal data structure)

---

#### 4.2.9 Data Formatter Agent

**Location**: `agent.py`, Lines 917-991

**Architecture**:
```
┌─────────────────────────────────────────────────────────────┐
│  Data Formatter Agent                                        │
│  Methods: _normalize_data(), _format_final_response()        │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  _normalize_data(data_list, origin)                  │   │
│  │  Role: Standardize data structure across all streams  │   │
│  │                                                       │   │
│  │  Input: List[Dict] with mixed key formats            │   │
│  │    - snake_case: {"cost_reduction_idea": "..."}      │   │
│  │    - Title Case: {"Cost Reduction Idea": "..."}      │   │
│  │                                                       │   │
│  │  Process:                                            │   │
│  │    for item in data_list:                            │   │
│  │      n = {}                                          │   │
│  │      # Map all possible key variations               │   │
│  │      n["Idea Id"] = item.get("idea_id") or           │   │
│  │                     item.get("Idea Id") or "N/A"     │   │
│  │      n["Cost Reduction Idea"] = item.get("cost_reduction_idea") or│   │
│  │                                item.get("Cost Reduction Idea")│   │
│  │      ... (all fields)                                 │   │
│  │      n["Origin"] = origin  # "Existing DB", "AI Innovation", etc.│   │
│  │      normalized.append(n)                             │   │
│  │                                                       │   │
│  │  Output: List[Dict] with consistent Title Case keys  │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  _format_final_response(all_data, query)             │   │
│  │  Role: Generate human-readable response text          │   │
│  │                                                       │   │
│  │  Process:                                            │   │
│  │    existing = [d for d in all_data                   │   │
│  │                if d["Origin"] == "Existing DB"]      │   │
│  │    new_ai = [d for d in all_data                     │   │
│  │              if d["Origin"] == "AI Innovation"]       │   │
│  │    web_data = [d for d in all_data                   │   │
│  │                if d["Origin"] == "World Wide Web"]   │   │
│  │                                                       │   │
│  │    res = f"Analyzed query: '{query}'.\n\n"           │   │
│  │    res += f"**Existing Database Matches ({len(existing)}):**\n"│   │
│  │    for x in existing[:3]:                            │   │
│  │      res += f"- {x['Cost Reduction Idea']}...\n"      │   │
│  │    ... (similar for new_ai and web_data)             │   │
│  │                                                       │   │
│  │  Output: Formatted markdown-style text                │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Role**: Standardizes data formats and generates human-readable summaries.

**Key Features**:
- **Key Normalization**: Handles multiple key formats (snake_case, Title Case, mixed)
- **Default Values**: Provides sensible defaults for missing fields
- **Origin Tagging**: Tags each idea with its source (DB/AI/Web)
- **Response Formatting**: Creates markdown-style summary text

**Design Pattern**: Facade Pattern (simplifies complex data transformation)

---

### 4.3 Agent Interaction Flow

#### 4.3.1 Complete Agent Orchestration

```
User Query: "Reduce weight of MG Hector Brake Disc"
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  VAVEAgent.run(user_query)                                   │
│     │                                                        │
│     ├─> Component Extractor Agent                            │
│     │     _smart_extract_target()                            │
│     │     → "brake disc"                                     │
│     │                                                        │
│     ├─> Context Resolver Agent                               │
│     │     _resolve_engineering_context()                    │
│     │     → Vehicle: hector, Component: brake               │
│     │     → Context string + state variables                │
│     │                                                        │
│     ├─> Stream A: RAG Retrieval                              │
│     │     vector_db_func(query, top_k=5)                    │
│     │     → 5 existing ideas                                 │
│     │                                                        │
│     ├─> Stream B: Innovation Engine                          │
│     │     │                                                  │
│     │     ├─> Idea Generator Agent                           │
│     │     │     _generate_with_llm()                        │
│     │     │       ├─> Context Resolver (uses cached state)  │
│     │     │       ├─> LLM Caller Agent                      │
│     │     │       │     call_llm() → _call_gemini()         │
│     │     │       └─> JSON Parser Agent                     │
│     │     │             _parse_llm_json()                     │
│     │     │     → 12 raw ideas                               │
│     │     │                                                  │
│     │     ├─> Validation Agent                               │
│     │     │     _validate_and_filter_ideas()                 │
│     │     │       ├─> Context Resolver (for physics checks) │
│     │     │       ├─> Score normalization                    │
│     │     │       ├─> Scope/impossible checks                │
│     │     │       └─> Physics heuristics                    │
│     │     │     → 6-8 validated ideas                         │
│     │     │                                                  │
│     │     └─> Image Assignment Agent                         │
│     │           _evaluate_visual_match() (for each idea)     │
│     │           → Images assigned                            │
│     │                                                        │
│     ├─> Stream C: Web Mining Engine                         │
│     │     │                                                  │
│     │     ├─> Query Generator Agent                          │
│     │     │     Generate 5 search queries                    │
│     │     │                                                  │
│     │     ├─> Parallel Web Search Agent                      │
│     │     │     ThreadPoolExecutor → perform_web_search()    │
│     │     │     → Aggregated web context                     │
│     │     │                                                  │
│     │     └─> Research Synthesis Agent                      │
│     │           ├─> Context Resolver (uses cached state)    │
│     │           ├─> LLM Caller Agent                         │
│     │           └─> JSON Parser Agent                        │
│     │           → 4 adapted research proposals              │
│     │                                                        │
│     ├─> Image Enrichment                                    │
│     │     _enrich_ideas_with_images()                        │
│     │       → VLM Engine (external)                          │
│     │       → Images generated/retrieved                    │
│     │                                                        │
│     └─> Data Formatter Agent                                 │
│           ├─> _normalize_data() (for each stream)          │
│           └─> _format_final_response()                      │
│           → Normalized data + formatted text                 │
│                                                              │
│  Output: response_text + table_data (15-17 ideas)            │
└─────────────────────────────────────────────────────────────┘
```

#### 4.3.2 Agent Communication Patterns

**1. Direct Method Calls** (Synchronous):
- `run()` → `run_innovation_engine()` → `_generate_with_llm()`
- Fast, predictable execution order

**2. Function Injection** (Dependency Injection):
- `vector_db_func` injected from `app.py`
- Allows testing with mock functions

**3. State Sharing** (Instance Variables):
- `_last_vehicle_key`, `_last_component_key` set by Context Resolver
- Used by Validation Agent, Web Mining Engine, VLM Engine

**4. External Service Calls** (Asynchronous I/O):
- LLM API calls (network I/O, ~5-15s)
- Web search (network I/O, ~10-20s)
- Image generation (network I/O, ~2-5s)

**5. Parallel Execution** (Threading):
- Web search uses ThreadPoolExecutor
- Other streams execute sequentially (can be parallelized)

---

### 4.4 Agent Decision-Making Logic

#### 4.4.1 Context Resolution Decision Tree

```
Query: "Reduce weight of MG Hector Brake Disc"
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  Vehicle Detection                                           │
│     │                                                        │
│     ├─> "mg hector" in query? → YES → vehicle_key = "hector"│
│     │                                                        │
│     └─> No match? → vehicle_key = "hector" (default)         │
│                                                              │
│  Component Detection                                         │
│     │                                                        │
│     ├─> "brake" in query? → YES → component_key = "brake"    │
│     │                                                        │
│     ├─> "disc" in query? → YES → component_key = "brake"   │
│     │                                                        │
│     └─> No match? → component_key = "general" (default)       │
│                                                              │
│  Special Rule Check                                          │
│     │                                                        │
│     ├─> vehicle["type"] == "EV" AND component_key == "hvac"?  │
│     │   → NO (ICE + brake)                                  │
│     │   → extra_rules = ""                                  │
│     │                                                        │
│     └─> YES? → extra_rules = "EV×HVAC RULE: ..."            │
│                                                              │
│  Output: Context string with constraints                     │
└─────────────────────────────────────────────────────────────┘
```

#### 4.4.2 Validation Decision Tree

```
Idea: "Reduce brake disc thickness to reduce weight"
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  Scope Check                                                │
│     │                                                        │
│     ├─> "brake" in idea text? → YES → scope_ok = True      │
│     │                                                        │
│     └─> NO → Reject (Out of Scope)                          │
│                                                              │
│  Impossible Pattern Check                                    │
│     │                                                        │
│     ├─> "100% reduction" in text? → NO                      │
│     ├─> "zero cost" in text? → NO                            │
│     │                                                        │
│     └─> Match found? → Reject (Impossible)                  │
│                                                              │
│  Physics Heuristics                                          │
│     │                                                        │
│     ├─> Heavy Vehicle Brake Risk?                            │
│     │     │                                                  │
│     │     ├─> component_key == "brake"? → YES                │
│     │     ├─> vehicle_weight >= 1600? → YES (1650)          │
│     │     ├─> "reduce weight" in text? → YES                │
│     │     ├─> "disc" in text? → YES                         │
│     │     │                                                  │
│     │     └─> heavy_brake_risk = True                       │
│     │                                                        │
│     └─> EV×HVAC Range Impact? → NO (ICE + brake)            │
│                                                              │
│  Scoring Gate                                               │
│     │                                                        │
│     ├─> min_score >= 25? → YES (assume 78)                  │
│     │                                                        │
│     └─> NO → Flag for Review                                │
│                                                              │
│  Final Classification                                        │
│     │                                                        │
│     ├─> scores_ok AND not heavy_brake_risk?                 │
│     │   → NO (heavy_brake_risk = True)                      │
│     │                                                        │
│     └─> → "Needs Human Review"                               │
│         validation_notes = "MG Hector: Heavy-vehicle brake  │
│                            thermal risk. Avoid rotor/disc    │
│                            thinning unless validated by      │
│                            thermal CFD + fade/DVPR."        │
└─────────────────────────────────────────────────────────────┘
```

#### 4.4.3 Image Matching Decision Tree

```
Query: "Close up of brake disc showing rotor"
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  Embedding Availability Check                               │
│     │                                                        │
│     ├─> image_embeddings is None? → NO                      │
│     │                                                        │
│     └─> YES → Fallback to keyword matching                  │
│                                                              │
│  Semantic Matching                                           │
│     │                                                        │
│     ├─> Encode query → query_vec (1, 384)                   │
│     ├─> Cosine similarity → scores (N_images,)              │
│     ├─> Sort → top_indices                                  │
│     │                                                        │
│     ├─> For each idx in top_indices:                        │
│     │     │                                                  │
│     │     ├─> score < 0.35? → Break (no good match)         │
│     │     │                                                  │
│     │     ├─> filename in used_images? → Skip               │
│     │     │                                                  │
│     │     └─> Return (image_path, label)                    │
│     │                                                        │
│     └─> No match found? → ("NaN", "No Matching Image Found")│
└─────────────────────────────────────────────────────────────┘
```

---

### 4.5 Agent State Management

#### 4.5.1 State Variables

**Persistent State** (Initialized once, reused):
```python
self.image_index: Dict[str, str]           # Loaded from JSON
self.image_embeddings: np.ndarray          # Computed on init
self.image_filenames: List[str]            # Ordered list
self.vlm: Optional[VLMEngine]              # Initialized if deps available
```

**Runtime State** (Set per query):
```python
self._last_vehicle_key: str                # Set by _resolve_engineering_context
self._last_component_key: str              # Set by _resolve_engineering_context
self._last_vehicle_name: str               # Cached for VLM prompts
self._last_vehicle_type: str               # For physics heuristics
self._last_vehicle_weight: Optional[int]   # For physics heuristics
self._last_result_data: List[Dict]         # Cached normalized results
```

**Temporary State** (Per method call):
```python
global_used_images: set[str]               # Tracks assigned images
validated: List[Dict]                      # Auto-approved ideas
needs_review: List[Dict]                    # Flagged ideas
```

#### 4.5.2 State Flow

```
Query Arrives
     │
     ▼
_resolve_engineering_context()
     │
     ├─> Sets: _last_vehicle_key, _last_component_key, ...
     │
     └─> Returns: context string
     │
     ▼
run_innovation_engine()
     │
     ├─> Uses: _last_vehicle_name (from state)
     │
     └─> _validate_and_filter_ideas()
           │
           └─> Uses: _last_vehicle_weight, _last_component_key (from state)
     │
     ▼
run_web_mining_engine()
     │
     └─> Uses: _last_vehicle_name (from state)
     │
     ▼
_enrich_ideas_with_images()
     │
     └─> Uses: _last_vehicle_name (from state, passed to VLM)
     │
     ▼
_normalize_data()
     │
     └─> Stores in: _last_result_data (cached for get_last_result_data())
```

---

### 4.6 Agent Error Handling & Resilience

#### 4.6.1 Error Handling Strategy

**1. LLM Call Failures**:
```python
try:
    response = model.generate_content(...)
except Exception as e:
    if "429" in str(e):  # Rate limit
        time.sleep(5)
        continue  # Try next model
    elif "404" in str(e):  # Model not found
        continue  # Try next model
    else:
        logger.error(...)
        continue  # Try next model
# If all models fail, return error string
```

**2. JSON Parsing Failures**:
```python
try:
    ideas = json.loads(match.group(0))
    return ideas
except Exception as e:
    logger.error(f"JSON Parsing Exception: {e}")
    return []  # Graceful degradation
```

**3. Image Matching Failures**:
```python
try:
    # Semantic matching
    ...
except Exception as e:
    logger.error(f"Semantic Engine Failure: {e}")
    return "NaN", "Error"  # Fallback to keyword matching
```

**4. Validation Failures**:
```python
try:
    # Validation logic
    ...
except Exception as e:
    logger.error(f"Validation error for idea: {e}")
    # Continue to next idea (don't crash entire pipeline)
```

#### 4.6.2 Resilience Patterns

1. **Graceful Degradation**: Returns empty lists/placeholders on failure
2. **Fallback Chains**: Semantic → Keyword → Placeholder
3. **Error Logging**: All errors logged, execution continues
4. **Partial Results**: Returns whatever was successfully processed

---

### 4.7 Agent Performance Characteristics

| Agent | Time Complexity | Space Complexity | Typical Time |
|-------|----------------|-----------------|--------------|
| Component Extractor | O(M) | O(M) | <1ms |
| Context Resolver | O(M*K) | O(1) | <5ms |
| Idea Generator | O(1) + Network | O(R) | 5-15s |
| Validation Agent | O(N*M) | O(N) | 50-100ms |
| Image Matcher | O(N log N) | O(1) | 10-50ms |
| Web Mining | O(Q*T) + Network | O(C) | 10-20s |
| JSON Parser | O(R) | O(R) | <1ms |
| Data Formatter | O(N*F) | O(N) | 10-50ms |

**Where**:
- M = query/idea text length
- K = max aliases per vehicle/component
- N = number of ideas
- R = LLM response length
- Q = number of search queries
- T = search time per query
- C = web context size
- F = number of fields per idea

---

### 4.8 Agent Testing & Validation

#### 4.8.1 Test Cases

**Component Extractor**:
- Input: "find ideas for brake disc" → Output: "brake disc"
- Input: "MG Hector HVAC blower" → Output: "hvac blower"
- Input: "reduce cost" → Output: "Automotive Component" (fallback)

**Context Resolver**:
- Input: "MG Hector brake" → Vehicle: hector, Component: brake
- Input: "ZS EV blower" → Vehicle: zsev, Component: hvac, Extra: EV×HVAC rule
- Input: "generic query" → Vehicle: hector (default), Component: general (default)

**Validation Agent**:
- Test Case A: Hector brake disc weight reduction → Needs Review (thermal risk)
- Test Case B: Comet brake disc weight reduction → Auto-Approved (light vehicle)
- Test Case C: ZS EV blower power increase → Needs Review (range impact)

**Image Matcher**:
- Query: "brake disc" → Matches image with "brake" in caption
- Query: "nonexistent component" → Returns "NaN" if no match above threshold
- Uniqueness: Same query twice → Different images (used_images tracking)

---

## Summary: AI Agent Architecture

The VAVE AI System employs a **multi-agent architecture** with:

1. **Master Orchestrator**: VAVEAgent coordinates all sub-agents
2. **Specialized Agents**: Each agent has a single, well-defined responsibility
3. **State Management**: Shared state enables agent coordination
4. **Error Resilience**: Graceful degradation at every level
5. **Physics-Informed**: Context resolution prevents impossible suggestions
6. **Autonomous Validation**: Multi-criteria filtering reduces manual review

**Key Architectural Patterns**:
- **Pipeline Pattern**: Generator → Validator → Enricher
- **Strategy Pattern**: LLM provider selection, validation strategies
- **Chain of Responsibility**: Validation steps, model fallback
- **Facade Pattern**: Data normalization simplifies complexity
- **Adapter Pattern**: JSON parsing adapts LLM output

**Agent Communication**:
- **Synchronous**: Direct method calls for fast operations
- **Asynchronous**: Network I/O for LLM, web search, image generation
- **State Sharing**: Instance variables for coordination
- **Function Injection**: Dependency injection for testability

This architecture enables the system to generate, validate, and present cost reduction ideas with physics-informed constraints and autonomous quality control.

---

## 5. Internal State Management

### 4.1 Global State (app.py)

```python
# Module-level globals (initialized at startup)
embedding_model: SentenceTransformer      # Loaded once, reused
faiss_index: faiss.IndexFlatL2            # Loaded from disk or built
idea_texts: List[str]                      # Cached text for embeddings
idea_rows: List[Dict]                      # Cached DB rows
vave_agent: Optional[VAVEAgent]            # Lazy initialized
```

### 4.2 Agent Instance State

```python
# VAVEAgent instance variables (per request, but agent is reused)
self._last_vehicle_key: str                # Set by _resolve_engineering_context
self._last_component_key: str              # Set by _resolve_engineering_context
self._last_vehicle_name: str               # Cached for VLM prompts
self._last_result_data: List[Dict]         # Cached for get_last_result_data()
```

### 4.3 Thread Safety

**Current Implementation**:
- **Not thread-safe**: Global `vave_agent` shared across requests
- **Risk**: Concurrent requests may overwrite `_last_result_data`
- **Mitigation**: Flask development server is single-threaded

**Production Considerations**:
- Use thread-local storage for `_last_result_data`
- Or create agent instance per request (higher memory)

---

## 5. Data Structures & Algorithms

### 5.1 FAISS Index Structure

```python
# Index Type: IndexFlatL2 (Exact L2 distance)
dimension: int = 384  # SentenceTransformer embedding size
index: faiss.IndexFlatL2 = faiss.IndexFlatL2(dimension)

# Storage:
# - In-memory: numpy array of float32 vectors
# - On-disk: model/faiss_index.bin (serialized)

# Search Algorithm:
# - Brute-force L2 distance computation
# - Time: O(N * D) where N = vectors, D = dimension
# - Space: O(N * D * 4 bytes) ≈ 1.5MB per 1000 vectors
```

### 5.2 Image Embedding Matrix

```python
# Pre-computed on agent initialization
image_embeddings: np.ndarray
# Shape: (N_images, 384)
# Type: float32
# Normalized: L2-normalized for cosine similarity

# Memory: N_images * 384 * 4 bytes
# Example: 1000 images = 1.5MB
```

### 5.3 Used Images Tracking

```python
# Set data structure for O(1) lookup
used_images: set[str] = set()

# Operations:
# - Add: used_images.add(filename)  # O(1)
# - Check: filename in used_images   # O(1)
# - Size: len(used_images)           # O(1)
```

---

## 6. Concurrency & Threading Model

### 6.1 Current Threading Usage

**1. Web Search Parallelization**:
```python
# agent.py, Line 426
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = {executor.submit(perform_web_search, q, 3): q 
               for q in search_queries}
    # 5 queries, 3 workers → 2 batches
    # Batch 1: queries 1-3 execute in parallel
    # Batch 2: queries 4-5 execute after batch 1 completes
```

**2. Flask Threading**:
- Development server: Single-threaded (sequential requests)
- Production (Gunicorn): Multi-threaded (configurable workers)

### 6.2 Potential Parallelization

**Opportunity 1: Stream Parallelization**
```python
# Current (Sequential):
existing_ideas = stream_a()      # ~150ms
new_ideas = stream_b()           # ~10000ms
web_ideas = stream_c()           # ~20000ms
# Total: ~30 seconds

# Optimized (Parallel):
with ThreadPoolExecutor(max_workers=3) as executor:
    future_a = executor.submit(stream_a)
    future_b = executor.submit(stream_b)
    future_c = executor.submit(stream_c)
    existing_ideas = future_a.result()
    new_ideas = future_b.result()
    web_ideas = future_c.result()
# Total: ~20 seconds (bounded by slowest stream)
```

**Opportunity 2: Image Generation Parallelization**
```python
# Current (Sequential):
for idea in ideas:
    images = vlm.get_images_for_idea(...)  # ~2-5s each
# Total: 15 ideas * 3s = 45 seconds

# Optimized (Parallel):
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(vlm.get_images_for_idea, idea, ...)
               for idea in ideas]
    results = [f.result() for f in futures]
# Total: ~9 seconds (15 ideas / 5 workers * 3s)
```

---

## 7. Memory Management

### 7.1 Memory Footprint

**Startup (app.py initialization)**:
```
embedding_model (SentenceTransformer):     ~200MB
faiss_index (10k vectors):                 ~15MB
idea_texts (10k strings):                  ~50MB
idea_rows (10k dicts):                     ~100MB
image_embeddings (1k images):              ~1.5MB
─────────────────────────────────────────────────
Total:                                      ~366MB
```

**Per Request (VAVEAgent)**:
```
Agent instance:                             ~5MB
_last_result_data (15 ideas):               ~1MB
Temporary lists (validation, etc.):         ~2MB
─────────────────────────────────────────────────
Total per request:                          ~8MB
```

### 7.2 Memory Optimization

**1. Lazy Loading**:
- VAVEAgent initialized on first request (saves ~5MB at startup)

**2. Image Embedding Caching**:
- Pre-computed once, reused for all requests
- Trade-off: Higher startup memory, faster queries

**3. Garbage Collection**:
- Python GC handles temporary objects
- Large objects (embeddings) kept in memory for performance

---

## 8. API Call Sequences

### 8.1 Gemini API Call Sequence

```
Client (agent.py)
  │
  ├─> Configure API Key
  │     genai.configure(api_key=GOOGLE_API_KEY)
  │
  ├─> Initialize Model
  │     model = genai.GenerativeModel("gemini-2.5-flash-lite")
  │
  ├─> Construct Prompt
  │     full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"
  │
  ├─> HTTP POST Request
  │     POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent
  │     Headers:
  │       - Authorization: Bearer {api_key}
  │       - Content-Type: application/json
  │     Body:
  │       {
  │         "contents": [{"parts": [{"text": full_prompt}]}],
  │         "generationConfig": {"temperature": 0.7}
  │       }
  │
  ├─> Wait for Response (~5-15 seconds)
  │
  └─> Parse Response
        {
          "candidates": [{
            "content": {
              "parts": [{"text": "```json\n[...]\n```"}]
            }
          }]
        }
```

### 8.2 Pollinations API Call Sequence

```
Client (vlm_engine.py)
  │
  ├─> URL Encode Prompt
  │     prompt_encoded = quote(prompt_text)
  │
  ├─> Construct URL
  │     url = f"https://image.pollinations.ai/prompt/{prompt_encoded}?nologo=true"
  │
  ├─> HTTP GET Request
  │     GET {url}
  │     Timeout: 20 seconds
  │
  ├─> Wait for Response (~2-5 seconds)
  │
  └─> Save Image
        with open(filepath, 'wb') as f:
          f.write(response.content)
```

### 8.3 DuckDuckGo Search Sequence

```
Client (tools.py)
  │
  ├─> Rate Limiting
  │     time.sleep(random.uniform(3, 8))  # Human-like delay
  │
  ├─> Initialize DDGS
  │     with DDGS() as ddgs:
  │
  ├─> Execute Search
  │     results = ddgs.text(query, max_results=3, backend="html")
  │
  ├─> Format Results
  │     context = "\n".join([f"[{i}] {r['title']}: {r['body']}" 
  │                         for i, r in enumerate(results)])
  │
  └─> Return Context String
```

---

## Conclusion

This detailed system design document provides:

1. **High-Level Architecture**: Layered structure with component interactions
2. **Low-Level Design**: Data structures, algorithms, state management
3. **Complete Flow**: Step-by-step execution from user input to output
4. **AI Agent Architecture**: Comprehensive breakdown of all agents, their roles, flows, and interactions
5. **Internal State Management**: State variables, flow, and coordination
6. **Data Structures & Algorithms**: Detailed algorithm analysis with complexity
7. **Concurrency & Threading**: Current usage and optimization opportunities
8. **Memory Management**: Footprint analysis and optimization strategies
9. **API Call Sequences**: Detailed API interaction flows

**Key Insights**:
- Total execution time: ~47 seconds (can be reduced to ~25s with parallelization)
- Memory footprint: ~366MB at startup, ~8MB per request
- Main bottlenecks: LLM API calls (network I/O), web search (rate limiting)
- Thread safety: Current implementation is single-threaded (safe for dev server)

**Next Steps for Production**:
1. Implement stream parallelization
2. Add connection pooling for PostgreSQL
3. Implement Redis caching for frequent queries
4. Use async/await for I/O-bound operations
5. Add request queuing for rate-limited APIs

---

**Document Version**: 2.0 (Detailed)  
**Last Updated**: 2025-01-20  
**Status**: Comprehensive Low-Level System Design
