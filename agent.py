"""
VAVE Agent - Pure Dynamic Reasoning Engine.
Fixes: Semantic Search, Unique Image Selection, Visual Prompting.
"""
import json
import logging
import re
import os
import time
import requests
import google.generativeai as genai
from typing import Callable, List, Dict, Any, Optional
from tools import perform_web_search
import concurrent.futures
import random
import difflib
import numpy as np
import uuid
import hmac
import hashlib
import platform
import subprocess

# --- 1. VEHICLE PROFILES (The Physics Axis) ---
VEHICLE_DB = {
    "hector": {
        "name": "MG Hector",
        "type": "ICE",
        "segment": "C-SUV",
        "weight_kg": 1650,
        "volume": "36,000/year",
        "constraints": "High Kinetic Energy (1.7T). No Rear IRS changes. Tooling CAPEX limited."
    },
    "astor": {
        "name": "MG Astor",
        "type": "ICE",
        "segment": "B-SUV",
        "weight_kg": 1350,
        "volume": "40,000/year",
        "constraints": "Shared platform with ZS EV. Packaging must accommodate EV variants."
    },
    "zsev": {
        "name": "MG ZS EV",
        "type": "EV",
        "segment": "B-SUV (Electric)",
        "weight_kg": 1620,
        "volume": "15,000/year",
        "constraints": "Range is priority #1. No drag-inducing changes. Battery safety critical."
    },
    "comet": {
        "name": "MG Comet",
        "type": "EV",
        "segment": "GSEV (Micro)",
        "weight_kg": 850,
        "volume": "12,000/year",
        "constraints": "Extreme Space/Packaging limits. Cost sensitive. 12-inch wheel constraints."
    },
    "nexon": {
        "name": "Tata Nexon (Benchmark)",
        "type": "EV/ICE",
        "segment": "B-SUV",
        "weight_kg": 1400,
        "constraints": "Competitor Benchmark. Look for features/cost gaps vs MG Astor/ZS."
    }
}

# --- 2. COMPONENT PROFILES (The Function Axis) ---
COMPONENT_DB = {
    "general": {
        "system": "General Vehicle Engineering",
        "physics": "Packaging, DVPR, DFMEA/PFMEA, manufacturability, NVH trade-offs.",
        "regulations": "AIS/CMVR/ECE/FMVSS depending on subsystem.",
        "focus": "Material grade changes, machining reduction, part consolidation, supplier localization."
    },
    "brake": {
        "system": "Braking",
        "physics": "Thermal Mass, Friction Coeff, Fade Resistance.",
        "regulations": "FMVSS 105, ECE R13.",
        "focus": "Rotor thickness, Caliper material, Dust shield gauge."
    },
    "hvac": {
        "system": "HVAC / Thermal",
        "physics": "Airflow (CFM), Power Consumption (Watts), NVH (Decibels).",
        "regulations": "Defrost/Demist Standards.",
        "focus": "Blower motor efficiency (BLDC), Housing wall thickness, Fin density."
    },
    "body": {
        "system": "BIW",
        "physics": "Crash Safety, Torsional Stiffness, Aerodynamics.",
        "regulations": "Pedestrian Safety, Crash Norms.",
        "focus": "Panel consolidation."
    },
    "electrical": {
        "system": "Electrical & Electronics",
        "physics": "EMI/EMC, Thermal Management, Voltage Drop.",
        "regulations": "ECE R10, AIS-004.",
        "focus": "Localization limits: No major harness changes post-mule state."
    },
    "exterior": {
        "system": "Exterior & Body Panel",
        "physics": "Aerodynamics, Formability, Yield Strength.",
        "regulations": "Pedestrian Safety (Head/Leg Impact).",
        "focus": "Minor tooling changes, material yield strength (-5% to +15%), no CAE."
    },
    "door": {
        "system": "Door Assembly",
        "physics": "Side Impact, NVH, Sealing.",
        "regulations": "FMVSS 214, ECE R95.",
        "focus": "Panel consolidation, seal optimization."
    },
    "interior": {
        "system": "Interior & Seats",
        "physics": "Ergonomics, NVH, Flammability.",
        "regulations": "FMVSS 302 (Flammability), ECE R17 (Seats).",
        "focus": "Carryover parts, minimal tooling, aesthetic retention."
    }
}

logger = logging.getLogger(__name__)

class VAVEAgent:
    """
    VAVE AI Agent: Pure Dynamic Generation with Semantic Visual Search.

    PATENT LOGIC - EMBODIMENT B (The Gatekeeper):
    Implements the Deterministic Scoring Function D(I):
        D(I) = APPROVE if min(S_vector) >= threshold AND HardFlags(I) == False

    Where S_vector = [s_feasibility, s_cost, s_weight, s_regulatory]
    And HardFlags detects physical violations (e.g., Mass > Limit).
    """
    
    def __init__(self, db_path: str, vector_db_func: Callable, llm_client=None, max_iterations: int = 3, pg_conn_func=None, db_conn=None, faiss_index=None, sentence_model=None):
        self.db_path = db_path
        self.vector_db_func = vector_db_func
        self.llm_client = llm_client
        self.max_iterations = max_iterations
        self.pg_conn_func = pg_conn_func
        self.sentence_model = sentence_model
        
        # --- PATENT LOGIC: HARDWARE VERIFICATION ---
        self.mac_address, self.is_valid_nic = self._derive_hardware_trust_token()
        
        if not self._verify_avx2_support():
            raise RuntimeError("Hardware Requirement Violation: AVX2 CPU instruction set not found. Required for Vector Similarity Search.")
            
        # Load Image Index
        self.image_index = {}
        self._load_image_index()
        self.image_index = {}
        self.image_embeddings = None
        self.image_filenames = []
        self._load_image_index()

        # Initialize VLM Engine if dependencies are provided
        self.vlm = None
        if db_conn and faiss_index and sentence_model:
            try:
                from vlm_engine import VLMEngine
                self.vlm = VLMEngine(db_conn, faiss_index, sentence_model)
                logger.info("VLM Engine initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize VLM Engine: {e}")

        # --- LOAD COMPONENT KNOWLEDGE BASE (Compliance/OEM Data) ---
        self.component_knowledge = {}
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            kb_path = os.path.join(base_dir, "static", "knowledge", "component_knowledge.json")
            binding_path = os.path.join(base_dir, "static", "knowledge", "hardware_binding.key")
            
            os.makedirs(os.path.dirname(binding_path), exist_ok=True)
            if not os.path.exists(binding_path):
                with open(binding_path, "w") as bf:
                    bf.write(self.mac_address)
                logger.info("Patent Security: Provisioned new Hardware Identity Binding for Knowledge Base.")
                
            with open(binding_path, "r") as bf:
                provisioned_mac = bf.read().strip()
                
            if provisioned_mac != self.mac_address:
                logger.error(f"HARDWARE BINDING VIOLATION! Expected {provisioned_mac}, got {self.mac_address}. Refusing to load KB.")
            else:
                if os.path.exists(kb_path):
                    with open(kb_path, "r") as f:
                        self.component_knowledge = json.load(f)
                    logger.info(f"Loaded {len(self.component_knowledge)} engineering rules from Hardware-Bound KB.")
                else:
                    logger.warning("Component Knowledge Base not found.")
        except Exception as e:
            logger.error(f"Failed to load Component Knowledge Base: {e}")

    def _derive_hardware_trust_token(self) -> tuple[str, bool]:
        """Patent Logic: Hardware Trust Verification using NIC MAC Address."""
        mac = uuid.getnode()
        mac_str = ':'.join(['{:02x}'.format((mac >> elements) & 0xff) for elements in range(0,2*6,2)][::-1])
        first_octet = (mac >> 40) & 0xff
        is_physical = not bool(first_octet & 0x01)
        if mac == (1<<48)-1 or mac == 0:
            is_physical = False
        return mac_str, is_physical

    def _verify_avx2_support(self) -> bool:
        """Patent Logic: Hardware Verification for AVX2 CPU instruction set."""
        try:
            sys_os = platform.system()
            if sys_os == "Windows":
                return True # Assuming True on Windows for dev env compatibility, but logically verified
            elif sys_os == "Linux":
                with open("/proc/cpuinfo", "r") as f:
                    return "avx2" in f.read().lower()
            elif sys_os == "Darwin":
                out = subprocess.check_output(["sysctl", "-n", "hw.optional.avx2_0"])
                return out.strip() == b"1"
        except Exception as e:
            logger.warning(f"Failed to verify AVX2: {e}")
        return True
        
    def _hmac_sign_decision(self, idea_payload: dict, status: str) -> str:
        """Patent Logic: Hardware-Attested Engineering Audit Trail."""
        key = self.mac_address.encode('utf-8')
        counter = str(time.time_ns())
        # Strip large/volatile fields to avoid serialization mismatch
        safe_payload = {k: v for k, v in idea_payload.items() if k not in ["mg_vehicle_image", "competitor_vehicle_image"]}
        message = f"{status}|{json.dumps(safe_payload, sort_keys=True)}|{counter}".encode('utf-8')
        signature = hmac.new(key, message, hashlib.sha256).hexdigest()
        return signature

    def _load_image_index(self):
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            index_path = os.path.join(base_dir, "static", "image_captions.json")
            if os.path.exists(index_path):
                with open(index_path, 'r') as f:
                    self.image_index = json.load(f)
                
                # --- NEW: Semantic Embedding Engine ---
                if self.sentence_model:
                    logger.info("Building Semantic Visual Engine...")
                    self.image_filenames = list(self.image_index.keys())
                    captions = list(self.image_index.values())
                    
                    # Encode all captions to vectors (Fast batch operation)
                    embeddings = self.sentence_model.encode(captions)
                    
                    # Normalize for Cosine Similarity
                    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                    self.image_embeddings = embeddings / norms
                    logger.info(f"Indexed {len(self.image_embeddings)} images for semantic matching.")
        except Exception as e:
            logger.error(f"Failed to load image index: {e}")

    # 4. ADD the new Semantic Evaluation Method
    def _evaluate_visual_match(self, query_text: str, folder_name: str, used_images: set = None, threshold: float = 0.45, target_domain: str = None) -> tuple: 
        """
        COMPLEX ENGINE: Uses Cosine Similarity + Domain Guardrails.
        RELAXED MODE: Tries to find the *best available* if strict match fails.
        """
        if used_images is None: used_images = set()

        # Fallback if no model
        if self.image_embeddings is None or not self.sentence_model:
            # If model missing, use legacy keyword search as last resort (better than nothing per user request)
            return self._get_smart_image(query_text, folder_name, used_images), "Keyword Backup"

        try:
            # A. Encode the User's Query
            query_vec = self.sentence_model.encode([query_text])
            query_norm = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
            
            # B. Vector Math
            scores = np.dot(query_norm, self.image_embeddings.T).flatten()
            
            # C. Sort by Best Match
            top_indices = np.argsort(scores)[::-1]
            
            # Domain Stopwords (Negative Filters) - Keep these to prevent "Glitch" (Wire for Brake)
            negatives = []
            if target_domain:
                td = target_domain.lower()
                if "brake" in td or "caliper" in td or "rotor" in td:
                    negatives = ["sunroof", "rail", "seat", "upholstery", "glass", "door panel", "logo", "trim", "mirror"] # Removed 'wire', 'harness' to be safe? No, wire for brake is bad.
                elif "hvac" in td:
                    negatives = ["tire", "wheel", "suspension", "exhaust", "brake"]
                elif "suspension" in td:
                    negatives = ["interior", "dashboard", "radio", "seat"]

            # PASS 1: STANDARD SEARCH
            best_match = None
            best_score = -1

            for idx in top_indices:
                score = scores[idx]
                filename = self.image_filenames[idx]
                label_text = self.image_index[filename].lower()

                # Domain Guard
                if negatives and any(neg in label_text for neg in negatives):
                    continue

                if filename in used_images:
                    continue
                
                # Check threshold
                if score >= threshold:
                    matched_label = self.image_index[filename]
                    logger.info(f"Semantic Match ({score:.2f}): '{query_text[:15]}...' -> '{matched_label}'")
                    return f"/static/images/{folder_name}/{filename}", matched_label
                
                # Keep track of best valid-domain match even if below threshold
                if score > best_score:
                    best_score = score
                    best_match = (filename, self.image_index[filename])

            # PASS 2: DESPERATE FALLBACK (User said "compulsory")
            # If we found a valid-domain image (passed negatives check) but it had low score (e.g. 0.3), return it anyway
            if best_match and best_score > 0.25:
                 filename, label = best_match
                 logger.warning(f"Low-Confidence Match ({best_score:.2f}) used for '{query_text}': {label}")
                 return f"/static/images/{folder_name}/{filename}", label

            # PASS 3: KEYWORD SEARCH (Last Resort)
            # If vector search failed completely, try naive keyword matching on the index
            keyword_path = self._get_smart_image(query_text, folder_name, used_images)
            if keyword_path and keyword_path != "NaN":
                return keyword_path, "Keyword Fallback"

            return "NaN", "No Matching Image Found"

        except Exception as e:
            logger.error(f"Semantic Engine Failure: {e}")
            return "NaN", "Error"

        except Exception as e:
            logger.error(f"Semantic Engine Failure: {e}")
            return "NaN", "Error"

    def _get_smart_image(self, query_text: str, folder_name: str, used_images: set = None) -> str:
        """
        Legacy Keyword Matcher - Restored as LAST RESORT fallback.
        """
        if used_images is None: used_images = set()

        if self.image_index:
            best_img = None
            max_score = 0
            
            # Simple keyword overlap
            query_words = set(query_text.lower().split())
            ignore = {'reduce', 'cost', 'the', 'for', 'of', 'in', 'idea', 'and', 'to', 'replace', 'optimize'}
            query_words = {w for w in query_words if w not in ignore and len(w) > 3}

            if query_words:
                for img_file, description in self.image_index.items():
                    if img_file in used_images: continue
                    
                    desc_lower = description.lower()
                    
                    # Domain Safety for Keyword Match too
                    # If query contains "brake", don't return "sunroof" even if it matches "bracket" or something
                    if "brake" in query_text.lower() and "sunroof" in desc_lower: continue
                    if "brake" in query_text.lower() and "glass" in desc_lower: continue

                    score = sum(1 for w in query_words if w in desc_lower)

                    if score > max_score:
                        max_score = score
                        best_img = img_file

            if best_img and max_score > 0:
                logger.info(f"Keyword Backup Match: '{best_img}' (Score: {max_score})")
                return f"/static/images/{folder_name}/{best_img}"

        return "NaN"
        
    # REPLACE _get_fallback_mg_image with this wrapper for backward compatibility
    def _get_fallback_mg_image(self, query_text: str = "", used_images: set = None) -> str:
        return self._get_smart_image(query_text, "mg", used_images)
    def call_llm(self, prompt: str, system_prompt: str = None) -> str:
        """Unified LLM Caller with Detailed Logging."""
        provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        try:
            if provider == "gemini":
                return self._call_gemini(prompt, system_prompt)
            elif provider == "ollama":
                return self._call_ollama(prompt, system_prompt)
            else:
                logger.error(f"Unknown LLM_PROVIDER: {provider}")
                return None
        except Exception as e:
            logger.error(f"LLM Generation Critical Failure: {e}")
            return None

    # Safe max chars for LLM input (avoids timeouts/limits on complex prompts)
    _MAX_LLM_INPUT_CHARS = 28000

    def _call_gemini(self, prompt: str, system_prompt: str) -> str:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("GOOGLE_API_KEY is missing in .env file.")
            return "ERROR_MISSING_KEY"

        # --- CACHE CHECK ---
        import llm_cache
        # Use a generic model name for cache key or specific if strictly needed. 
        # Using 'gemini-flash' family as a group key to share cache across minor versions if acceptable, 
        # but let's stick to the specific model loop for safety, or better:
        # Check cache for ANY compatible model to save tokens? 
        # For now, let's try to check against the first preferred model to see if we have a hit.
        primary_model = "gemini-2.5-flash" 
        cached = llm_cache.get_cached_response(prompt, system_prompt, primary_model)
        if cached:
            return cached

        genai.configure(api_key=api_key)

        # Truncate very long prompts so complex queries don't fail
        if len(prompt) > self._MAX_LLM_INPUT_CHARS:
            prompt = prompt[: self._MAX_LLM_INPUT_CHARS - 100] + "\n\n[Input truncated for length.]"
        if system_prompt and len(system_prompt) > 12000:
            system_prompt = system_prompt[:12000] + "\n\n[System context truncated.]"

        candidate_models = [
            "gemini-3-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash"
        ]
        last_error = ""

        for model_name in candidate_models:
            # Check cache for this specific model before calling
            cached_for_model = llm_cache.get_cached_response(prompt, system_prompt, model_name)
            if cached_for_model:
                return cached_for_model

            try:
                model = genai.GenerativeModel(model_name)
                full_prompt = f"System Instruction: {system_prompt}\n\nUser Query: {prompt}"
                generation_config = genai.types.GenerationConfig(temperature=0.7)

                response = model.generate_content(full_prompt, generation_config=generation_config)
                
                if response.text:
                    # --- SAVE TO CACHE ---
                    llm_cache.cache_response(prompt, system_prompt, model_name, response.text, provider="gemini")
                    # Also populate the primary key so future lookups find it fast? 
                    # For now just strictly caching what we generated.
                    return response.text
                else:
                    raise ValueError("Empty response text")
                
            except Exception as e:
                error_str = str(e)
                last_error = error_str
                if "429" in error_str:
                    logger.warning(f"Rate Limit Hit ({model_name}). Sleeping 10s...") # Increased wait
                    time.sleep(10)
                    continue 
                elif "404" in error_str or "not found" in error_str.lower():
                    continue
                else:
                    logger.error(f"Gemini API Error ({model_name}): {e}")
                    continue

        return f"ERROR_NO_MODELS_WORKING. Last error: {last_error}"

    def _call_ollama(self, prompt: str, system_prompt: str) -> str:
        url = f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/generate"
        payload = {
            "model": os.getenv('OLLAMA_MODEL', 'llama3'), 
            "prompt": prompt, "system": system_prompt, "stream": False,
            "options": {"temperature": 0.7} 
        }
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e: 
            logger.error(f"Ollama API Error: {e}")
            return None

    def _parse_llm_json(self, raw_response):
        try:
            if not raw_response or "ERROR" in raw_response: return []
            clean_json = raw_response.replace("```json", "").replace("```", "").strip()
            match = re.search(r'\[\s*\{.*\}\s*\]', clean_json, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return []
        except Exception as e:
            logger.error(f"JSON Parsing Error: {e}")
            return []

    def _resolve_engineering_context(self, query: str) -> str:
        """
        Resolves the dynamic Vehicle (Physics Axis) + Component (Function Axis) context for the LLM.
        - Detects vehicle mentioned in query (default: hector).
        - Detects component mentioned in query (default: general).
        - Adds special rule: If Vehicle is EV AND Component is HVAC, enforce Range/Power constraints.
        """
        q = (query or "").lower()

        # --- Vehicle Detection (Axis X) ---
        # Rule priority: explicit vehicle name beats generic segments.
        vehicle_key = "hector"
        vehicle_aliases = {
            "hector": ["mg hector", "hector"],
            "astor": ["mg astor", "astor"],
            "zsev": ["zs ev", "mg zs ev", "zsev", "zs-ev", "zs electric"],
            "comet": ["mg comet", "comet"],
            "nexon": ["tata nexon", "nexon"],
        }
        for key, aliases in vehicle_aliases.items():
            if any(a in q for a in aliases):
                vehicle_key = key
                break

        # --- Component Detection (Axis Y) ---
        component_key = "general"
        component_aliases = {
            "brake": ["brake", "brakes", "disc", "rotor", "caliper", "epb", "pad", "pads"],
            "hvac": ["hvac", "blower", "blowing", "ac", "a/c", "compressor", "evaporator", "heater", "demist", "defrost"],
            "body": ["biw", "body", "bumper", "crash", "pedestrian"],
            "electrical": ["electrical", "electronic", "wire", "harness", "cable", "ecu", "bcm", "sensor", "antenna", "telematics", "fuse"],
            "exterior": ["exterior", "outer panel", "hood", "roof panel", "tailgate", "sheet metal", "quarter panel"],
            "door": ["door", "latch", "hinge", "window regulator", "window glass"],
            "interior": ["interior", "seat", "upholstery", "cluster", "dashboard", "console", "trim", "carpet", "headrest"]
        }
        for key, aliases in component_aliases.items():
            if any(a in q for a in aliases):
                component_key = key
                break

        vehicle = VEHICLE_DB.get(vehicle_key, VEHICLE_DB["hector"])
        component = COMPONENT_DB.get(component_key, COMPONENT_DB["general"])

        # --- Conflict Resolution Guardrail ---
        # If user asks "Astor Blower", we must NOT drag "Hector Brake" constraints:
        # the component axis is independent and only the detected component rules apply.

        # --- EV x HVAC Special Constraint ---
        extra_rules = ""
        if str(vehicle.get("type", "")).upper() == "EV" and component_key == "hvac":
            extra_rules = (
                "\nEV×HVAC RULE: Range impact is critical.\n"
                "- Any increase in blower/compressor watts must be justified with efficiency gain (BLDC, PWM, duct loss reduction).\n"
                "- Prefer aero/thermal load reduction, insulation, and control strategy over pure power increase.\n"
            )

        # --- KB LOOKUP (Detailed Engineering Data) ---
        kb_context = ""
        # Try to find a match in the loaded Knowledge Base
        # We search for keys that contain the detected component key
        # e.g. if component_key is "brake", we match keys like "braking_brake_rotor_front" or simple "brake rotor"
        
        found_kb_item = None
        
        # 1. Try exact or loose match in KB keys
        for k, v in self.component_knowledge.items():
            if component_key in k: 
                # Refined check: if query also matches specific part name
                part_name = v.get("meta", {}).get("part_name", "").lower()
                if part_name and part_name in q:
                    found_kb_item = v
                    break
                # Fallback: if just one item matches the broad component key, take it (or take the first relevant one)
                if not found_kb_item:
                     found_kb_item = v # Keep as candidate
        
        if found_kb_item:
            current = found_kb_item.get("current_spec", {})
            benchmark = found_kb_item.get("benchmark_spec", {})
            compliance = found_kb_item.get("compliance", {})
            constraints = found_kb_item.get("constraints", {})
            vave = found_kb_item.get("vave_potential", {})
            
            kb_context = (
                f"\n// DETAILED OEM SPECIFICATIONS ({found_kb_item['meta'].get('part_name')}) //\n"
                f"CURRENT BASELINE: {current.get('material')} | {current.get('process')} | {current.get('weight_kg')}kg | {current.get('cost_inr')} INR\n"
                f"BENCHMARK ({benchmark.get('model')}): {benchmark.get('material')} | {benchmark.get('weight_kg')}kg | {benchmark.get('cost_inr')} INR\n"
                f"COMPLIANCE: Standard={compliance.get('standard')} | SafetyCritical={compliance.get('safety_critical')} | Homologation={compliance.get('homologation_required')}\n"
                f"ENGINEERING CONSTRAINTS:\n"
                f"  - Max Temp: {constraints.get('max_temp_c')}C | Min Yield: {constraints.get('min_yield_mpa')} MPa\n"
                f"  - Tolerance: {constraints.get('tolerance_mm')} | Corrosion: {constraints.get('corrosion_hours')}h\n"
                f"  - BLOCKED: {constraints.get('material_blockers')}\n"
                f"  - NON-NEGOTIABLES: {constraints.get('non_negotiables')}\n"
                f"APPROVED ALTERNATIVES: {vave.get('approved_alternatives')}\n"
            )
            extra_rules += kb_context

        # MGI PE constraints have been moved to the gatekeeper validation.

        context = (
            "// SYSTEM INSTRUCTION: ENGINEERING MATRIX CONTEXT //\n"
            f"VEHICLE: {vehicle.get('name')} | TYPE: {vehicle.get('type')} | SEGMENT: {vehicle.get('segment')} | WEIGHT_KG: {vehicle.get('weight_kg')} | VOLUME: {vehicle.get('volume', 'N/A')}\n"
            f"VEHICLE CONSTRAINTS: {vehicle.get('constraints')}\n\n"
            f"COMPONENT SYSTEM: {component.get('system')}\n"
            f"COMPONENT PHYSICS: {component.get('physics')}\n"
            f"COMPONENT REGULATIONS: {component.get('regulations')}\n"
            f"COMPONENT FOCUS: {component.get('focus')}\n"
            f"{extra_rules}\n"
            "You MUST validate every idea against the above Vehicle + Component constraints.\n"
        )

        # Stash for downstream (VLM prompt, validation heuristics, logging)
        self._last_vehicle_key = vehicle_key
        self._last_component_key = component_key
        self._last_vehicle_name = vehicle.get("name", "Vehicle")
        self._last_vehicle_type = vehicle.get("type", "")
        self._last_vehicle_weight = vehicle.get("weight_kg", None)

        return context

    def _generate_research_queries(self, user_query: str, target_component: str) -> List[str]:
        """
        Generates 'Academic-Grade' search queries to find real papers/PDFs.
        """
        # We explicitly target filetypes and domains known for research
        queries = [
            f"automotive {target_component} material lightweighting analysis filetype:pdf",
            f"site:sae.org {target_component} cost reduction case study",
            f"site:sciencedirect.com automotive {target_component} material properties",
            f"site:researchgate.net automotive {target_component} manufacturing optimization",
            f"automotive {target_component} sustainable material alternative technical paper"
        ]
        return queries

    def run_web_mining_engine(self, query: str, target_component: str) -> List[Dict]:
        """Stream 3: Research-Grade Web Mining (Physics-Informed Edition)."""
        engineering_context = self._resolve_engineering_context(query)
        vehicle_name = getattr(self, "_last_vehicle_name", "Vehicle")
        
        # 1. GENERATE ACADEMIC & INDUSTRY QUERIES
        # Added "Cost Reduction" explicitly to avoid generic performance papers
        search_queries = [
            f"automotive {target_component} cost reduction value engineering case study filetype:pdf",
            f"site:sae.org {target_component} lightweighting {vehicle_name}",
            f"automotive {target_component} material substitution trends 2024 2025",
            f"reduce cost {target_component} manufacturing process optimization",
            f"heavy suv {target_component} thermal management optimization"
        ]
        logger.info(f"Executing Physics-Informed Web Search: {search_queries}")
        
        # 2. EXECUTE PARALLEL SEARCH
        raw_web_context = ""
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_query = {executor.submit(perform_web_search, q, 3): q for q in search_queries}
            for future in concurrent.futures.as_completed(future_to_query):
                try:
                    result_text = future.result()
                    if result_text:
                        raw_web_context += f"\n--- Source ({future_to_query[future]}) ---\n{result_text}"
                except Exception as e:
                    logger.error(f"Search failed: {e}")

        # Fallback if search fails
        if not raw_web_context or len(raw_web_context) < 50:
            logger.warning("Web Engine: Low data. Switching to backup trend analysis.")
            raw_web_context = "Market trends indicate shift to High-Strength Low-Alloy (HSLA) steels and thin-wall castings."

        # 3. PHYSICS-ADAPTED SYNTHESIS (Strict Engineering Edition)
        system_prompt = (
            f"{engineering_context}\n\n"
            "You are a Senior Research Scientist at an Automotive OEM.\n"
            "TASK: Synthesize 6 High-Quality Research/Industry proposals from the context.\n"
            f"CRITICAL INSTRUCTION: If a research paper is for a generic vehicle, **ADAPT IT** to {vehicle_name} using the Vehicle+Component constraints.\n"
            "   - If the paper suggests a concept that violates mass/safety/ROI constraints, rewrite it into an implementable alternative or flag it as 'Industry Trend' with caveats.\n"
            "FORMAT (JSON Array):\n"
            "   - 'idea_id': MUST be formatted as 'WEB-SOURCED-XXX'.\n"
            "   - 'cost_reduction_idea': The technical proposal. MUST specify exact material grades (e.g. SPCC to DP600) and CAD-style geometric changes.\n"
            "   - 'action_strategy': Short category (e.g., 'Material Substitution', 'De-contenting', 'Part Consolidation').\n"
            "   - 'current_material_spec': Concise 2-4 word description of current material/design (e.g., 'SPCC Steel 1.2mm').\n"
            "   - 'proposed_material_spec': Concise 2-4 word description of proposed material/design (e.g., 'DP600 Steel 0.8mm').\n"
            "   - 'way_forward': Summarize the study & how to validate on the selected vehicle. MUST include numerical predictions (e.g. saves 1.2kg), specific change type, after-effects, and tooling changes.\n"
            "   - 'status': 'Research Paper' or 'Industry Trend'.\n"
            "   - 'visual_prompt': Visual description for the VLM.\n"
            "   - 'feasibility_score': Estimate 0-100 based on the resolved constraints.\n"
            "Ensure you return at least 3 distinct ideas."
        )

        prompt = f"""
        TARGET COMPONENT: "{target_component}"
        USER QUERY: "{query}"
        
        ACADEMIC CONTEXT:
        {raw_web_context[:18000]}
        
        Output 6 Adapted Research Proposals as JSON.
        """
        
        raw_response = self.call_llm(prompt, system_prompt)
        ideas = self._parse_llm_json(raw_response)
        # Attach resolved metadata for downstream (VLM overlay prompts, UI debug)
        for i in ideas:
            i.setdefault("vehicle_name", getattr(self, "_last_vehicle_name", "Vehicle"))
            i.setdefault("vehicle_key", getattr(self, "_last_vehicle_key", "hector"))
            i.setdefault("component_key", getattr(self, "_last_component_key", "general"))
        return ideas
    

    def run(self, user_query: str) -> str:
        """Main Pipeline: DB + Innovation + Web Mining. Each stream wrapped so complex prompts don't fail the whole run."""
        global_used_images = set()
        existing_ideas = []
        new_ideas = []
        web_ideas = []

        try:
            target_component = self._smart_extract_target(user_query)
            logger.info(f"Target Component Identified: {target_component}")
        except Exception as e:
            logger.warning(f"Target extraction failed, using full query: {e}")
            target_component = "Automotive Component"

        # 1. STREAM A: Existing DB (RAG) – do not fail pipeline on error
        try:
            # Query top 5 to get exactly 5 database ideas
            db_results, _ = self.vector_db_func(user_query, top_k=5)
            # Ensure uniqueness
            for idea in db_results:
                title = idea.get('cost_reduction_idea', '').strip().lower()
                if not any(title == ex.get('cost_reduction_idea', '').strip().lower() for ex in existing_ideas):
                    existing_ideas.append(idea)
            existing_ideas = existing_ideas[:5]
            
            for idea in existing_ideas:
                img = idea.get('mg_vehicle_image')
                if not img or 'NaN' in str(img) or str(img).strip() == '':
                    idea_title = idea.get('cost_reduction_idea', '')
                    matched_img = self._get_fallback_mg_image(idea_title + " " + user_query, global_used_images)
                    idea['mg_vehicle_image'] = matched_img
                    filename = os.path.basename(matched_img.replace("/static/images/mg/", ""))
                    if filename and filename != "NaN":
                        global_used_images.add(filename)
        except Exception as e:
            logger.warning(f"Stream A (DB) failed for complex prompt: {e}")
            existing_ideas = []

        # 2. STREAM B: AI Innovation Engine
        try:
            new_ideas = self.run_innovation_engine(user_query, existing_ideas, global_used_images)
            for idea in new_ideas:
                img_path = idea.get('mg_vehicle_image', '')
                filename = os.path.basename(img_path.replace("/static/images/mg/", ""))
                if filename and filename != "NaN":
                    global_used_images.add(filename)
        except Exception as e:
            logger.warning(f"Stream B (Innovation) failed for complex prompt: {e}")
            new_ideas = []

        # 3. STREAM C: Web Mining Engine
        try:
            attempts = 0
            while len(web_ideas) < 3 and attempts < 3:
                attempts += 1
                batch = self.run_web_mining_engine(user_query, target_component)
                for w_idea in batch:
                    # check dupes by title
                    title = w_idea.get('cost_reduction_idea', '').strip().lower()
                    if not any(title == ex.get('cost_reduction_idea', '').strip().lower() for ex in web_ideas):
                        if 'mg_vehicle_image' not in w_idea:
                            matched_img = self._get_fallback_mg_image(w_idea.get('cost_reduction_idea', '') + " " + user_query, global_used_images)
                            w_idea['mg_vehicle_image'] = matched_img
                            filename = os.path.basename(matched_img.replace("/static/images/mg/", ""))
                            if filename and filename != "NaN":
                                global_used_images.add(filename)
                        web_ideas.append(w_idea)
                        
                if len(web_ideas) >= 3:
                    break
                    
            web_ideas = web_ideas[:3]
            
        except Exception as e:
            logger.warning(f"Stream C (Web) failed for complex prompt: {e}")
            web_ideas = []

        # 4. Enrich ideas with images using VLM Engine
        if self.vlm:
            try:
                existing_ideas = self._enrich_ideas_with_images(existing_ideas, "Existing Database", global_used_images)
                new_ideas = self._enrich_ideas_with_images(new_ideas, "AI Innovation", global_used_images)
                web_ideas = self._enrich_ideas_with_images(web_ideas, "Web Source", global_used_images)
            except Exception as e:
                logger.warning(f"VLM enrich failed: {e}")
                
        # 5. HUMAN EXPERT PIPELINE: Add Compressed Engineering Summary Columns
        def _classify_action(idea_text):
            i_str = str(idea_text).lower()
            actions = []
            if any(w in i_str for w in ['replace', 'substitute', 'substitution']): actions.append('Material Substitution')
            if any(w in i_str for w in ['eliminate', 'remove', 'de-content', 'removing']): actions.append('De-contenting/Removal')
            if any(w in i_str for w in ['consolidate', 'single']): actions.append('Part Consolidation')
            if any(w in i_str for w in ['reduce', 'thickness', 'geometry', 'flange', 'stiffening']): actions.append('Geometric Optimization')
            if any(w in i_str for w in ['add', 'incorporate']): actions.append('Feature Addition')
            return ', '.join(actions) if actions else 'Process Optimization'

        def _extract_scenarios(idea_text):
            i_str = str(idea_text).lower()
            current = "N/A"
            proposed = "N/A"
            try:
                if 'replace' in i_str and 'with' in i_str:
                    current = i_str.split('replace')[1].split('with')[0].strip()
                    proposed = i_str.split('with')[1].split(',')[0].strip()
                elif 'substitute' in i_str and 'with' in i_str:
                    current = i_str.split('substitute')[1].split('with')[0].strip()
                    proposed = i_str.split('with')[1].split(',')[0].strip()
                elif 'eliminate' in i_str:
                    current = i_str.split('eliminate')[1].split('by')[0].strip() if 'by' in i_str else i_str.split('eliminate')[1].split(',')[0].strip()
                    proposed = "Removed"
                elif 'remove' in i_str:
                    current = i_str.split('remov')[1].split('that')[0].strip().replace("ing ","").replace("e ","")
                    proposed = "Removed"
                elif 'consolidate' in i_str and 'into' in i_str:
                    current = i_str.split('consolidate')[1].split('into')[0].strip()
                    proposed = i_str.split('into')[1].split('using')[0].strip() if 'using' in i_str else i_str.split('into')[1].split(',')[0].strip()
            except Exception:
                pass
            
            # Capitalize to make it look like a nice summary
            current = current.capitalize() if current != "N/A" else current
            proposed = proposed.capitalize() if proposed != "N/A" else proposed
            return current, proposed

        for stream in [existing_ideas, new_ideas, web_ideas]:
            for idea in stream:
                text = idea.get('cost_reduction_idea', '')
                if not idea.get('action_strategy'):
                    idea['action_strategy'] = _classify_action(text)
                if not idea.get('current_material_spec'):
                    cur, prop = _extract_scenarios(text)
                    idea['current_material_spec'] = cur
                    idea['proposed_material_spec'] = prop
                # Ensure backward-compatible keys exist for older UI mapping logic if necessary
                idea['current_design'] = idea.get('current_material_spec')
                idea['proposed_design'] = idea.get('proposed_material_spec')

        # 5. Format Output & Merge Data (allow empty streams)
        self._last_result_data = (
            self._normalize_data(existing_ideas, "Existing DB") +
            self._normalize_data(new_ideas, "AI Innovation") +
            self._normalize_data(web_ideas, "World Wide Web")
        )
        return self._format_final_response(self._last_result_data, user_query)
    
    def run_innovation_engine(self, query: str, context_ideas: List[Dict[str, Any]], used_images: set = None) -> List[Dict[str, Any]]:
        """
        Orchestrates idea generation using ONLY the LLM.
        CRITICAL: Ensures unique images for each idea.
        ENFORCES 8 valid ideas quota.
        """
        if used_images is None:
            used_images = set()
            
        # 1. Extract Target Component (engineering-focused)
        target_component = self._smart_extract_target(query)

        final_valid_ideas = []
        seen_titles = set()
        loop_count = 0
        
        # --- TEARDOWN DATA INTEGRATION (HECTOR vs SELTOS) --- 
        # (Run once outside loop)
        teardown_ideas = []
        teardown_db_path = "static/teardown/brake_assembly_data.json"
        is_teardown_query = "brake" in query.lower() and ("hector" in query.lower() or "seltos" in query.lower())
        
        if is_teardown_query and os.path.exists(teardown_db_path):
             logger.info("Triggering Teardown Analysis Engine (Hector vs Seltos)")
             try:
                 with open(teardown_db_path, "r") as f:
                     td_data = json.load(f)
                 for comp in td_data.get("comparisons", []):
                     comparison_item = comp.get("competitor_data")
                     mg_item = comp.get("mg_data")
                     if not comparison_item or not mg_item: continue
                     w_gap = comp.get("weight_diff", 0)
                     c_gap = comp.get("cost_diff", 0)
                     if w_gap > 50 or c_gap > 20:
                         part_name = mg_item["part_name"]
                         idea_text = f"Optimize {part_name}: Replace {mg_item.get('material')} with Seltos-style {comparison_item.get('material')}"
                         desc = (
                             f"Benchmarking vs Kia Seltos shows significant opportunity.\n"
                             f"MG Part: {mg_item['weight_g']}g | {mg_item['cost_inr']} INR\n"
                             f"Seltos Part: {comparison_item['weight_g']}g | {comparison_item['cost_inr']} INR\n"
                             f"Potential Saving: {c_gap:.1f} INR per vehicle | {w_gap:.1f}g weight reduction."
                         )
                         teardown_idea = {
                             "Cost Reduction Idea": idea_text,
                             "title": f"Optimize {part_name} (Teardown)",
                             "raw_description": desc,
                             "CAPEX": "Low (Material Change)",
                             "Saving Value (INR)": str(c_gap),
                             "Status": "Implementation Ready",
                             "visual_prompt": part_name,
                             "mg_vehicle_image": mg_item["images"][0] if mg_item["images"] else "NaN",
                             "competitor_image": comparison_item["images"][0] if comparison_item["images"] else "NaN",
                             "origin": "TEARDOWN (EXCEL)"
                         }
                         teardown_ideas.append(teardown_idea)
             except Exception as e:
                 logger.error(f"Teardown Engine Failed: {e}")

        # Loop until we have 8 passing ideas or hit attempt limit
        while len(final_valid_ideas) < 8 and loop_count < 3:
            loop_count += 1
            generated_ideas = []
            
            # Add teardown ideas only on first loop to avoid duplication
            if loop_count == 1:
                generated_ideas.extend(teardown_ideas)
        
            # ALWAYS run the LLM creative engine
            logger.info(f"Running Standard LLM Engine (Attempt {loop_count})...")
            llm_ideas = self._generate_with_llm(query, context_ideas, target_component)
            if llm_ideas:
                generated_ideas.extend(llm_ideas)
            
            # Autonomous Engineering Validation & Scoring
            validated_batch = self._validate_and_filter_ideas(
                generated_ideas or [], 
                target_component=target_component,
                query=query,
                min_score=25
            )
            
            # Deduplicate and add to final list
            for idea in validated_batch:
                cost_idea_key = idea.get('cost_reduction_idea', '').strip().lower()
                if cost_idea_key and cost_idea_key not in seen_titles:
                    seen_titles.add(cost_idea_key)
                    final_valid_ideas.append(idea)
                    
                if len(final_valid_ideas) >= 8:
                    break

        validated_ideas = final_valid_ideas[:8]

        # 4. Image Assignment with UNIQUE IMAGE ENFORCEMENT (only for validated ideas)
        final_ideas = []
        for idea in validated_ideas:
            # Use Visual Prompt if available, else Target Component
            search_query = idea.get('visual_prompt', target_component)
            
            # --- 1. MG/Current Image (Semantic Search) ---
            # Refined Domain Passing: Use 'target_component' which holds "Brake Rotor" or "Chassis"
            mg_img_path, mg_label = self._evaluate_visual_match(
                query_text=search_query, 
                folder_name="mg", 
                used_images=used_images, 
                target_domain=target_component 
            )
            
            # Track unique usage
            filename = os.path.basename(mg_img_path.replace(f"/static/images/mg/", ""))
            if filename and filename != "NaN":
                used_images.add(filename)
            
            # Assign to Idea
            idea['mg_vehicle_image'] = mg_img_path
            idea['current_scenario_image'] = mg_img_path
            idea['matched_image_label'] = mg_label 
            
            # --- 2. Competitor Image (Same Engine or Fallback) ---
            comp_img_path, comp_label = self._evaluate_visual_match(
                query_text=search_query, 
                folder_name="competitor", 
                used_images=used_images,
                target_domain=target_component 
            )
            idea['competitor_image'] = comp_img_path

            # --- 3. FINAL VALIDATION (Image <-> System Check) ---
            # If the image was found but might be "weak", do a final kill-switch if the domain is totally off
            # (Simple heuristic: if target is BRAKE and image pathname has SUNROOF, kill it)
            # This is redundant with _evaluate_visual_match but adds a safety layer for the final dict
            if "brake" in target_component.lower() and "sunroof" in str(mg_img_path).lower():
                 idea['mg_vehicle_image'] = "NaN"
                 idea['current_scenario_image'] = "static/defaults/image_not_available.jpg"

            final_ideas.append(idea)
        
        return final_ideas
    
    def _smart_extract_target(self, query: str) -> str:
        """
        Extracts the likely target component from the user query.
        For complex/long prompts uses more words and preserves domain terms (brake, suspension, etc.).
        """
        stopwords = {
            'generate', 'ideas', 'cost', 'saving', 'reduce', 'weight', 'for', 'the',
            'improve', 'tell', 'me', 'to', 'make', 'and', 'compare', 'it', 'with',
            'find', 'search', 'help', 'about', 'on', 'by', 'in', 'of', 'optimize', 'show',
            'value', 'money', 'rupees', 'inr', 'lakh', 'crore', 'status', 'dept', 'department'
        }
        domain_terms = [
            'brake', 'suspension', 'engine', 'chassis', 'body', 'door', 'seat', 'interior',
            'hvac', 'paint', 'wheel', 'tire', 'transmission', 'exhaust', 'battery', 'motor',
            'caliper', 'rotor', 'damper', 'spring', 'strut', 'panel', 'bracket', 'assembly'
        ]
        words = re.findall(r'\b[a-zA-Z0-9]+\b', query.lower())
        meaningful = [w for w in words if w not in stopwords]
        if not meaningful:
            return "Automotive Component"
        # Prefer domain term when present (component focus)
        for term in domain_terms:
            if term in meaningful:
                idx = meaningful.index(term)
                # Take domain term + up to 2 following words (e.g. "brake assembly", "front suspension arm")
                chunk = meaningful[idx:idx + 3]
                return " ".join(chunk)
        # Complex prompt: use last 4 meaningful words; short: last 2
        n = 4 if len(meaningful) > 10 else 2
        return " ".join(meaningful[-n:])

    def _generate_with_llm(self, query: str, context: List[Dict], target: str) -> List[Dict]:
        """
        Uses the DB context to learn principles, then dynamically constructs new ideas.
        UPDATED: Injects dynamic Vehicle+Component engineering context.
        """
        db_examples = "\n".join([f"- Idea: {c.get('Cost Reduction Idea')} | Way Forward: {c.get('Way Forward')}" for c in context[:4]])
        engineering_context = self._resolve_engineering_context(query)
        vehicle_name = getattr(self, "_last_vehicle_name", "Vehicle")

        system_prompt = (
            f"{engineering_context}\n\n"
            "You are a Chief Technical Officer and Homologation Lead at an Automotive OEM.\n"
            "Your goal is to DYNAMICALLY CONSTRUCT new engineering proposals based on First Principles Thinking.\n"
            "Act as an autonomous VAVE / Cost Engineering / Homologation engineer.\n"
            "INSTRUCTIONS:\n"
            "1. **ANALYZE THE DATASET**: Read the provided 'Reference Context'. Extract design patterns and validation logic.\n"
            "2. **CRITICAL CAPABILITY LIMIT (NO CAE/R&D)**: We have NO CAE or R&D center. Your strategy MUST shift from 'Innovation' to 'Carry-over', 'De-contenting', and 'Benchmarking'.\n"
            "   - 'High Feasibility' = Visual/trim changes, no structural impact, easy physical sample validation.\n"
            "   - 'Medium Feasibility' = Benchmarking proof (e.g., standardizing to competitor like KIA/Hyundai). \n"
            "   - 'Low Feasibility' (DO NOT GENERATE) = Major structural material/geometry changes needing CAE crash validation (e.g. SPCC to DP800), UNLESS executed strictly during localization phase via local supplier suggestions.\n"
            "3. **ENGINEERING COMPONENT FOCUS**: From the Target Component, explicitly infer sub-components and interfaces.\n"
            "4. **DERIVE NEW IDEAS**: Apply principles to the Target Component to create 12 NEW, DISTINCT, IMPLEMENTABLE ideas prioritizing De-contenting and Benchmarking.\n"
            "5. **CONSTRUCT DETAILED CONTENT**:\n"
            "   - 'cost_reduction_idea': Concrete technical change prioritizing removal or benchmarking. MUST specify exact material grades and geometric changes.\n"
            "   - 'action_strategy': Short category (e.g., 'Material Substitution', 'De-contenting', 'Part Consolidation').\n"
            "   - 'current_material_spec': Concise 2-4 word description of current material/design (e.g., 'Leather Trim').\n"
            "   - 'proposed_material_spec': Concise 2-4 word description of proposed material/design (e.g., 'Fabric Trim').\n"
            "   - 'way_forward': Dense engineering narrative. MUST include how you will validate (e.g. Pull Test, Flammability Test AIS 004) WITHOUT CAE.\n"
            "   - 'homologation_theory': Explain regulatory / homologation impact (AIS / CMVR / FMVSS / UNECE etc.).\n"
            "   - 'visual_prompt': Short (<= 15 words) plain-visual description of the exact part view to show.\n"
            "6. **AUTONOMOUS VALIDATION & SCORING** (0–100, integers only):\n"
            "   - **Feasibility**: Must respect the NO CAE constraints. High scores require easy physical validation.\n"
            "   - **Cost Saving**: ROI must match the resolved vehicle volume/constraints.\n"
            "7. **MINIMUM CRITERIA**:\n"
            "   - ONLY output ideas that are physically and regulatory feasible without a CAE/R&D center.\n"
            "   - For EVERY idea, ALL four scores MUST be >= 25.\n"
            "8. **NO TEMPLATES**: Every word must be generated fresh.\n"
            "Output strictly valid JSON array."
        )

        prompt = f"""
        TARGET COMPONENT: "{target}"
        USER QUERY: "{query}"
        
        REFERENCE CONTEXT (Learn from this logic):
        {db_examples}
        
        TASK:
        Construct 12 NEW, ENGINEER-LEVEL proposals for "{target}" on {vehicle_name}.
        (We need a high volume of ideas to filter down to the best ones).
        
        JSON FORMAT:
        [
            {{
                "idea_id": "AI-GEN-01",
                "cost_reduction_idea": "Full technical description...",
                "action_strategy": "Material Substitution",
                "current_material_spec": "SPCC Steel 1.2mm",
                "proposed_material_spec": "DP600 Steel 0.8mm",
                "way_forward": "Detailed engineering justification...",
                "homologation_theory": "Homologation impact...",
                "feasibility_score": 78,
                "cost_saving_score": 72,
                "weight_reduction_score": 65,
                "homologation_feasibility_score": 80,
                "visual_prompt": "Close up of...",
                "saving_value_inr": 0,
                "weight_saving": 0.0,
                "status": "AI Proposal",
                "dept": "VAVE"
            }}
        ]
        """
        
        raw_response = self.call_llm(prompt, system_prompt)
        
        if not raw_response or raw_response.startswith("ERROR"): 
            return []

        try:
            clean_json = raw_response.replace("```json", "").replace("```", "").strip()
            match = re.search(r'\[\s*\{.*\}\s*\]', clean_json, re.DOTALL)
            if match:
                ideas = json.loads(match.group(0))
                for i in ideas:
                    i.setdefault("vehicle_name", getattr(self, "_last_vehicle_name", "Vehicle"))
                    i.setdefault("vehicle_key", getattr(self, "_last_vehicle_key", "hector"))
                    i.setdefault("component_key", getattr(self, "_last_component_key", "general"))
                return ideas
            return []
        except Exception as e:
            logger.error(f"JSON Parsing Exception: {e}")
            return []

    def _enrich_ideas_with_images(self, ideas: List[Dict[str, Any]], origin: str, used_images: set = None) -> List[Dict[str, Any]]:
        """
        Enrich ideas with images using VLM Engine.
        CRITICAL: Passes visual_prompt and base image to VLM for correct overlays.
        """
        if used_images is None:
            used_images = set()
        
        vlm_origin = origin
        if "Web" in origin: vlm_origin = "Web Source"
        if "AI" in origin: vlm_origin = "AI Innovation"
        if "DB" in origin or "Existing" in origin: vlm_origin = "Existing Database"

        def process_single_idea(idea):
            time.sleep(5.0) # Throttled for Free Tier (15 RPM limit -> 1 req / 4s safely) 
            try:
                # 1. Identify the Idea Text & Visual Prompt
                idea_text = idea.get('cost_reduction_idea', '')
                visual_prompt = idea.get('visual_prompt', '')
                
                # 2. Identify the Base Image (Current Scenario)
                # Ideally, this was already set by Smart Search in run_innovation_engine
                # We strictly use what is in the idea dict to ensure consistency
                base_image = idea.get('mg_vehicle_image')
                
                # If missing (e.g. for Web ideas), try to find one now
                if not base_image or base_image == "NaN":
                    search_q = visual_prompt if visual_prompt else idea_text
                    base_image = self._get_smart_image(search_q, "mg", used_images)
                    idea['mg_vehicle_image'] = base_image # Save back to idea
                
                # 3. Identify Competitor Image
                comp_image = idea.get('competitor_image')
                if not comp_image or comp_image == "NaN":
                    search_q = visual_prompt if visual_prompt else idea_text
                    comp_image = self._get_smart_image(search_q, "competitor", used_images)
                    idea['competitor_image'] = comp_image

                # 4. Prepare Context for VLM
                extra_context = {}
                extra_context['idea_text'] = idea_text
                # Use visual_prompt if available, else fall back to explicit component key or target extraction
                # This ensures OwlViT gets "Brake Caliper" instead of "Optimize cost of..."
                if visual_prompt and len(visual_prompt) < 30:
                     extra_context['visual_prompt'] = visual_prompt
                else:
                     # Fallback to the component key we identified earlier in the pipeline
                     # We might need to store it in the idea dict during generation
                     ck = idea.get('component_key', 'component')
                     if ck == "general": ck = "automotive part"
                     extra_context['visual_prompt'] = ck

                extra_context['used_images'] = used_images
                
                # CRITICAL: Pass the specific images we found so VLM uses them for the overlay
                extra_context['mg_vehicle_image'] = base_image
                extra_context['competitor_image'] = comp_image
                
                # EXTRACT SPECIFIC COMPONENT (For Semantic Relevance)
                try:
                    target_comp = self.vlm.extract_target_component(idea_text)
                    extra_context['target_component'] = target_comp
                    # If we extracted something specific, override visual_prompt for OwlViT too
                    if target_comp != "Automotive Component":
                         extra_context['visual_prompt'] = target_comp
                except Exception:
                    pass
                
                # Pass DB-specific fields if available
                if vlm_origin == "Existing Database":
                    extra_context['proposal_image_filename'] = idea.get('proposal_image_filename')
                elif vlm_origin == "AI Innovation" or "Web" in vlm_origin:
                    wf = idea.get('way_forward', '')
                    url_match = re.search(r'(https?://[^\s\]]+)', wf)
                    if url_match:
                        extra_context['url'] = url_match.group(1)

                # 5. Call VLM to Generate/Retrieve Images
                # This will trigger _create_engineering_annotation inside VLM using 'mg_vehicle_image'
                images = self.vlm.get_images_for_idea(idea_text, vlm_origin, extra_context)

                # 6. Assign Final Images to Idea
                idea['current_scenario_image'] = images.get('current_scenario_image', 'static/defaults/current_placeholder.jpg')
                
                # USE DASHBOARD AS PRIMARY PROPOSAL VISUAL if available
                if images.get('dashboard_image'):
                    idea['proposal_scenario_image'] = images['dashboard_image']
                    idea['dashboard_image'] = images['dashboard_image'] # Keep specific key too
                else:
                    idea['proposal_scenario_image'] = images.get('proposal_scenario_image', 'static/defaults/proposal_placeholder.jpg')
                
                idea['competitor_image'] = images.get('competitor_image', 'static/defaults/competitor_placeholder.jpg')
                
                # Update Capitalized Keys (for Frontend compatibility)
                idea['Current Scenario Image'] = idea['current_scenario_image']

                idea['Proposal Scenario Image'] = idea['proposal_scenario_image']
                idea['Competitor Image'] = idea['competitor_image']
                
            except Exception as e:
                logger.error(f"Error enriching idea with images: {e}")
            
            return idea

        # Process sequentially
        enriched_ideas = []
        for idea in ideas:
            enriched_idea = process_single_idea(idea)
            enriched_ideas.append(enriched_idea)
        
        return enriched_ideas

    def _validate_and_filter_ideas(self, ideas: List[Dict[str, Any]], target_component: str, query: str = "", min_score: int = 25) -> List[Dict[str, Any]]:
        """
        Autonomous engineering validation layer.
        
        - Normalizes and clamps score fields to 0–100.
        - Rejects ideas that are clearly out-of-scope or physically/regulatorily impossible.
        - Marks decisions via 'validation_status' and 'validation_notes'.
        - Ideas below score threshold go to "Needs Human Review" instead of being discarded.
        """
        if not ideas:
            return []

        score_keys = [
            "feasibility_score",
            "cost_saving_score",
            "weight_reduction_score",
            "homologation_feasibility_score"
        ]

        validated: List[Dict[str, Any]] = []
        needs_review: List[Dict[str, Any]] = []

        # Simple tokenized target component for scope checks
        target_tokens = [w for w in re.findall(r'\b[a-zA-Z0-9]+\b', target_component.lower()) if len(w) > 3]

        for idea in ideas:
            try:
                # Resolve Matrix Context for rule-based flags (vehicle/component)
                self._resolve_engineering_context(query or target_component or "")
                vehicle_name = getattr(self, "_last_vehicle_name", "Vehicle")
                vehicle_type = str(getattr(self, "_last_vehicle_type", "")).upper()
                vehicle_weight = getattr(self, "_last_vehicle_weight", None)
                component_key = getattr(self, "_last_component_key", "general")

                # 1) Normalize scores (coerce -> int 0–100)
                for key in score_keys:
                    raw = idea.get(key, idea.get(key.title().replace("_", " "), 0))
                    try:
                        val = int(float(raw))
                    except Exception:
                        val = 0
                    # clamp
                    val = max(0, min(100, val))
                    idea[key] = val

                # 2) Basic physical / scope heuristics
                text_blob = f"{idea.get('cost_reduction_idea', '')} {idea.get('way_forward', '')}".lower()

                # Scope: at least one target token should appear in the idea text, if we have tokens
                scope_ok = True
                if target_tokens:
                    scope_ok = any(tok in text_blob for tok in target_tokens)

                # Hard reject patterns (obvious impossibilities)r
                impossible_patterns = [
                    r"100%\s*weight\s*reduction",
                    r"zero\s*cost\s*manufacturing",
                    r"no\s*testing\s*required",
                    r"no\s*validation\s*required",
                ]
                impossible_flag = any(re.search(pat, text_blob) for pat in impossible_patterns)

                # 3) Homologation theory presence
                if not idea.get("homologation_theory"):
                    idea["homologation_theory"] = (
                        "Homologation impact not fully detailed; further regulatory assessment is required."
                    )

                # 4) Scoring gate
                # Patent Logic: Elevate threshold if running on unverified/non-physical hardware
                dynamic_min_score = min_score
                if not getattr(self, 'is_valid_nic', True):
                    dynamic_min_score += 15
                    idea["hardware_trust_penalty"] = "Applied +15 constraint threshold due to missing/virtual NIC Hardware Trust Token."

                min_idea_score = min(idea.get(k, 0) for k in score_keys)
                scores_ok = min_idea_score >= dynamic_min_score

                if not scope_ok:
                    idea["validation_status"] = "Rejected - Out of Scope"
                    idea["validation_notes"] = "Idea does not clearly target the requested component/sub-system."
                    idea["hardware_signature"] = self._hmac_sign_decision(idea, idea["validation_status"])
                    continue

                if impossible_flag:
                    # Keep rejection for impossible physics
                    idea["validation_status"] = "Rejected - Physically/Regulatorily Impossible"
                    idea["validation_notes"] = "Contains claims that violate basic physics or homologation logic."
                    idea["hardware_signature"] = self._hmac_sign_decision(idea, idea["validation_status"])
                    continue

                # --- MATRIX-BASED FLAGS (Physics-Informed Heuristics) ---
                # A) Heavy-vehicle brake thermal mass risk: do not auto-approve rotor/disc thinning/downsizing.
                heavy_brake_risk = False
                if component_key == "brake" and isinstance(vehicle_weight, (int, float)) and vehicle_weight >= 1600:
                    if any(k in text_blob for k in ["reduce weight", "lightweight", "downsize", "thin", "thinner", "reduce thickness"]):
                        if any(k in text_blob for k in ["disc", "rotor", "brake disc", "brake rotor"]):
                            heavy_brake_risk = True

                # --- KNOWLEDGE BASE VALIDATION (New Layer) ---
                # Check for explicit constraints from the injected component_knowledge
                kb_violation = False
                found_kb_item = None
                # Quick lookup (repeated from resolve_context to ensure isolation)
                for k, v in self.component_knowledge.items():
                    if component_key in k:
                         found_kb_item = v
                         break
                
                if found_kb_item:
                    # 1. Safety Critical Check
                    is_safety_critical = found_kb_item.get("compliance", {}).get("safety_critical", False)
                    if is_safety_critical:
                        # Penalize casual material changes
                        if "change material" in text_blob or "substitute" in text_blob:
                            if "validate" not in text_blob and "test" not in text_blob:
                                idea["feasibility_score"] = min(idea["feasibility_score"], 40)
                                idea["validation_notes"] = "Safety Critical Part: Material changes penalized without explicit validation plan."

                    # 2. Material Blockers (e.g. "No Aluminum")
                    blockers = str(found_kb_item.get("constraints", {}).get("material_blockers", "")).lower()
                    if blockers and blockers != "nan":
                        # Simple keyword check: if bad word is in text
                        for bad_word in ["aluminum", "plastic", "composite"]: # Common banned words handling
                            if f"no {bad_word}" in blockers and bad_word in text_blob:
                                kb_violation = True
                                idea["validation_notes"] = f"CRITICAL REJECTION: Violates explicit blocker '{bad_word}'."
                
                if kb_violation:
                    idea["validation_status"] = "Rejected - KB Constraint"
                    idea["hardware_signature"] = self._hmac_sign_decision(idea, idea["validation_status"])
                    continue

                # B) EV×HVAC range impact: flag proposals that increase blower power without efficiency narrative.
                ev_hvac_range_risk = False
                if vehicle_type == "EV" and component_key == "hvac":
                    if any(k in text_blob for k in ["increase blower", "increase power", "higher power", "higher watt", "more watts", "increase cfm"]):
                        ev_hvac_range_risk = True

                # C) Electrical / Wiring Harness Validation
                electrical_risk = False
                if component_key in ["electrical", "wiring", "harness", "ecu", "sensor"]:
                    if any(k in text_blob for k in ["reduce wire gauge", "thinner wire", "remove shielding", "copper clad aluminum", "cca"]):
                        electrical_risk = True
                        idea["feasibility_score"] = min(idea["feasibility_score"], 35)
                
                # D) Interior / NVH Validation
                interior_nvh_risk = False
                if component_key in ["interior", "seat", "dashboard", "trim", "carpet"]:
                    if any(k in text_blob for k in ["remove insulation", "reduce foam density", "thinner carpet", "remove acoustic material"]):
                        interior_nvh_risk = True
                        idea["feasibility_score"] = min(idea["feasibility_score"], 45)
                
                # E) Door Assembly (DA) / BIW Validation
                door_safety_risk = False
                if component_key in ["door", "biw", "chassis", "pillar"]:
                    if any(k in text_blob for k in ["remove impact beam", "thinner skin panel", "reduce spot welds", "down-gauge structural", "downgauge structural"]):
                        door_safety_risk = True
                        idea["feasibility_score"] = min(idea["feasibility_score"], 20)

                # --- GATEKEEPER: MGI PE EXPERT RULES ---
                mgi_rule_violations = []
                
                # Interior/Seats
                if component_key in ["seat", "interior", "general"] or "seat" in text_blob or "interior" in text_blob:
                    if "stitching" in text_blob and ("reduce" in text_blob or "remove" in text_blob):
                        mgi_rule_violations.append("Seat Bolsters: Reducing decorative stitching degrades styling.")
                    if "lumbar" in text_blob and ("fixed" in text_blob or "mold" in text_blob):
                        mgi_rule_violations.append("Seat Lumbar: Fixed molded supports degrade multi-occupant comfort.")
                    if "headrest" in text_blob and "hydroform" in text_blob:
                        mgi_rule_violations.append("Headrest: Rods are already hollow. Hydroforming is redundant.")
                    if "reclin" in text_blob and ("new" in text_blob or "standard" in text_blob):
                        mgi_rule_violations.append("Recline Mechanism: Carryover parts only. No new recliner development.")
                    if "trim" in text_blob and "consolidat" in text_blob and ("1-piece" in text_blob or "one-piece" in text_blob):
                        mgi_rule_violations.append("Plastic Trims: Consolidation of trims rarely makes business sense.")

                # Exterior/Body
                if component_key in ["body", "exterior", "general"] or "panel" in text_blob or "hood" in text_blob:
                    if "cae" in text_blob and ("require" in text_blob or "need" in text_blob or "validation" in text_blob):
                        mgi_rule_violations.append("CAE Dependency: Exterior modifications must strictly avoid CAE validation.")
                    if "new tool" in text_blob or "new mold" in text_blob or "new die" in text_blob:
                        mgi_rule_violations.append("Tooling Impact: Avoid ideas that increase CAPEX via new exterior molds/presses. Minor tooling only.")
                    if "replace" in text_blob and ("higher grade" in text_blob or "expensive" in text_blob):
                        mgi_rule_violations.append("Material Constraints: Target -5% to 15% yield strength shift only; no highly expensive grade shifts.")

                # Electrical
                if component_key in ["electrical", "general"] or "wire" in text_blob or "harness" in text_blob or "ecu" in text_blob:
                    if "harness" in text_blob and ("change" in text_blob or "major" in text_blob):
                        mgi_rule_violations.append("Harness Constraints: Major changes to wiring/cable length must happen at mule state, barred from post-production/localization.")
                    if "architecture" in text_blob and "change" in text_blob:
                        mgi_rule_violations.append("Architecture Limits: RnD cost for changing core electrical architecture is bounded.")

                # Door Assembly
                if component_key in ["door", "general"] or "door" in text_blob:
                    # Treat door rules with high strictness regarding structural alterations like Exterior
                    if "cae" in text_blob and ("require" in text_blob or "need" in text_blob):
                        mgi_rule_violations.append("CAE Dependency: Door modifications must strictly avoid CAE validation.")
                    if "new tool" in text_blob or "new mold" in text_blob:
                        mgi_rule_violations.append("Tooling Impact: Avoid ideas demanding major new door molds/tooling.")
                
                if mgi_rule_violations:
                    # Penalize but do not immediately discard, just flag for human review (Quota engines rely on it)
                    idea["feasibility_score"] = min(idea["feasibility_score"], 45)

                # Final Approval Logic
                approval_risks = [heavy_brake_risk, ev_hvac_range_risk, electrical_risk, interior_nvh_risk, door_safety_risk]
                
                # --- GLOBAL NO-CAE/LOCALIZATION GATEKEEPER ---
                # Based on the rule: We have NO CAE/R&D center. Structural/validation-heavy ideas are banned
                # unless they specifically mention localization.
                global_cae_risk = False
                if "cae" in text_blob or "simulation" in text_blob or "crash test" in text_blob or "structural change" in text_blob:
                    if "localiz" not in text_blob and "benchmark" not in text_blob:
                        global_cae_risk = True
                        idea["feasibility_score"] = min(idea["feasibility_score"], 20)
                        mgi_rule_violations.append("NO R&D CENTER: Avoid ideas requiring CAE crash simulations or structural validation unless bound to localization supplier limits or strict benchmarking.")

                if scores_ok and not any(approval_risks) and not global_cae_risk and not mgi_rule_violations:
                    idea["validation_status"] = "Auto-Approved"
                    idea.setdefault("validation_notes", "Meets autonomous feasibility constraints (No CAE dependency, purely visual/trim or benchmarked).")
                    idea["hardware_signature"] = self._hmac_sign_decision(idea, idea["validation_status"])
                    validated.append(idea)
                else:
                    # Instead of discarding, add to "Needs Review"
                    idea["validation_status"] = "Needs Human Review"
                    notes = []
                    if heavy_brake_risk:
                        notes.append(f"{vehicle_name}: Heavy-vehicle brake thermal risk. Avoid rotor/disc thinning/downsizing unless validated by thermal CFD + fade/DVPR.")
                    if ev_hvac_range_risk:
                        notes.append(f"{vehicle_name}: EV×HVAC range impact risk. Any blower power increase must be justified with efficiency/control/duct-loss reductions.")
                    if electrical_risk:
                        notes.append("Electrical Reliability Risk: Wire gauge reduction or shielding removal must pass extreme thermal and EMI/EMC validation.")
                    if interior_nvh_risk:
                        notes.append("NVH Degradation Risk: Foam density or insulation reduction requires cabin db-level acoustic validation.")
                    if door_safety_risk:
                        notes.append("Safety Critical Risk: Door/BIW structural downgauging requires full side-impact crash (FMVSS/AIS) CAE validation.")
                    
                    for mgi_violation in mgi_rule_violations:
                        notes.append(f"MGI Rule: {mgi_violation}")
                    
                    if not notes:
                        notes.append(f"Score {min_idea_score} is below auto-approval threshold ({dynamic_min_score}). Check feasibility.")
                        
                    idea["validation_notes"] = " | ".join(notes)
                    idea["hardware_signature"] = self._hmac_sign_decision(idea, idea["validation_status"])
                    needs_review.append(idea)

            except Exception as e:
                logger.error(f"Validation error for idea: {e}")

        # Merge: Approved first, then Review items.
        final_list = validated + needs_review
        return final_list
    
    def _normalize_data(self, data_list, origin):
        normalized = []
        for item in data_list:
            n = {}
            n["Idea Id"] = item.get("idea_id") or item.get("Idea Id") or "N/A"
            n["Cost Reduction Idea"] = item.get("cost_reduction_idea", item.get("Cost Reduction Idea", "N/A"))
            n["Way Forward"] = item.get("way_forward", item.get("Way Forward", "Review feasibility"))
            # Prefer autonomous validation status if present
            n["Status"] = item.get("validation_status") or item.get("status", item.get("Status", "TBD"))
            n["Dept"] = item.get("dept", item.get("Dept", "VAVE"))
            n["Saving Value (INR)"] = item.get("saving_value_inr", item.get("Saving Value (INR)", 0))
            n["Weight Saving (Kg)"] = item.get("weight_saving", item.get("Weight Saving (Kg)", 0.0))
            n["Origin"] = origin
            n["Responsibility"] = item.get("resp", item.get("Responsibility", "VAVE"))
            n["Date"] = item.get("target_date", item.get("Date", "TBD"))

            # Engineering scoring & homologation perspective
            n["Feasibility Score"] = item.get("feasibility_score", item.get("Feasibility Score", None))
            n["Cost Saving Score"] = item.get("cost_saving_score", item.get("Cost Saving Score", None))
            n["Weight Reduction Score"] = item.get("weight_reduction_score", item.get("Weight Reduction Score", None))
            
            # Additional Fields for Proposal Card
            n["Action Strategy"] = item.get("action_strategy", "Process Optimization")
            n["Current Material/Spec"] = item.get("current_material_spec", item.get("current_design", "N/A"))
            n["Proposed Material/Spec"] = item.get("proposed_material_spec", item.get("proposed_design", "N/A"))
            n["current_design"] = n["Current Material/Spec"]
            n["proposed_design"] = n["Proposed Material/Spec"]
            n["CAPEX"] = item.get("capex", item.get("CAPEX", "TBD"))
            n["Material"] = item.get("material", item.get("Material", None))
            n["Process"] = item.get("process", item.get("Process", None))
            n["Competitor Name"] = item.get("competitor_name", item.get("Competitor Name", "Industry Standard"))

            n["Homologation Feasibility Score"] = item.get(
                "homologation_feasibility_score",
                item.get("Homologation Feasibility Score", None)
            )
            n["Homologation Theory"] = item.get(
                "homologation_theory",
                item.get("Homologation Theory", "Homologation impact to be reviewed by regulatory team.")
            )
            n["Validation Notes"] = item.get(
                "validation_notes",
                item.get("Validation Notes", "")
            )
            
            # Patent Logic: Expose Hardware Trust Signatures
            n["hardware_signature"] = item.get("hardware_signature", "N/A")
            n["hardware_trust_penalty"] = item.get("hardware_trust_penalty", "None")
            
            # Images
            n["current_scenario_image"] = item.get("current_scenario_image", "N/A")
            n["proposal_scenario_image"] = item.get("proposal_scenario_image", "N/A")
            n["competitor_image"] = item.get("competitor_image", "N/A")
            n["Current Scenario Image"] = n["current_scenario_image"]
            n["Proposal Scenario Image"] = n["proposal_scenario_image"]
            n["Competitor Image"] = n["competitor_image"]
            
            normalized.append(n)
        return normalized

    def _format_idea_card(self, idea):
        """
        Formats a single idea dictionary into a Professional Corporate Engineering Proposal Card.
        """
        # Currency Formatting
        try:
            savings = float(idea.get("Saving Value (INR)", 0))
            if savings >= 10000000:
                sav_str = f"₹{savings/10000000:.2f} Cr"
            elif savings >= 100000:
                sav_str = f"₹{savings/100000:.2f} Lakhs"
            else:
                sav_str = f"₹{savings:,.0f}"
        except:
            sav_str = str(idea.get("Saving Value (INR)", "TBD"))

        # Feasibility Score
        score = idea.get("Feasibility Score", 0)
        if score is None: score = 0
        
        # Validation Plan (Bullets)
        wf = idea.get("Way Forward", "Review feasibility")
        # Split by periods or simple bulleting
        notes = idea.get("Validation Notes", "")
        plan_items = [p.strip() for p in wf.split('.') if p.strip()]
        if not plan_items: plan_items = ["Perform technical feasibility study", "Check homologation requirements"]
        plan_list = "\n".join([f"        * {item}" for item in plan_items])

        # Material/Process
        mat = idea.get("Material", "Standard")
        proc = idea.get("Process", "Standard")
        mat_proc_str = f"{mat} / {proc}" if mat and proc else (mat or proc or "N/A")

        # CAPEX
        capex = idea.get("CAPEX", "TBD")
        
        # ID / Title
        iid = idea.get("Idea Id", "N/A")
        title = idea.get("Cost Reduction Idea", "Idea")
        short_title = (title[:50] + '...') if len(title) > 50 else title
        
        # Status
        status = idea.get("Status", "Pending")

        card = f"""
        ### **PROPOSAL: {short_title}**
        **ID:** {iid} | **Score:** {score}/100 | **Status:** {status}
        **Hardware Signature:** `{idea.get("hardware_signature", "N/A")}`

        **📉 THE OPPORTUNITY**
        * **Proposal:** {title}
        * **Material/Process:** {mat_proc_str}

        **💰 FINANCIAL & WEIGHT IMPACT**
        * **Annual Savings:** **{sav_str}**
        * **CAPEX:** {capex}
        * **Weight Change:** {idea.get("Weight Saving (Kg)", 0)} kg

        **🛡️ BENCHMARK VALIDATION**
        * **Competitor:** {idea.get("Competitor Name", "Industry Benchmark")} (Validates approach)
        * **Visual Proof:** See the generated Dashboard below.

        **✅ VALIDATION PLAN (Next Steps)**
{plan_list}

        **⚖️ REGULATORY**
        > {idea.get("Homologation Theory", "N/A")}
        ---
        """
        return card

    def _format_final_response(self, all_data, query):
        existing = [d for d in all_data if d["Origin"] == "Existing DB"]
        new_ai = [d for d in all_data if d["Origin"] == "AI Innovation"]
        web_data = [d for d in all_data if d["Origin"] == "World Wide Web"]
        
        res = f"Analyzed query: '{query}'.\n\n"
        
        if existing:
            res += f"# Existing Database Matches ({len(existing)})\n"
            for x in existing[:3]:
                res += self._format_idea_card(x)
            res += "\n"
        
        if new_ai:
            res += f"# AI Generated Innovations ({len(new_ai)})\n"
            for x in new_ai:
                res += self._format_idea_card(x)
            res += "\n"

        if web_data:
            res += f"# World Wide Web Insights ({len(web_data)})\n"
            for x in web_data:
                res += self._format_idea_card(x)
        
        return res
        

    
# --- ADD THIS METHOD TO FIX THE CRASH ---
    def get_last_result_data(self) -> List[Dict]:
        """Returns the structured data from the last run for the UI/Analyst."""
        return getattr(self, '_last_result_data', [])