"""
VAVE VLM Engine - Smart Engineering Overlay with Pinpoint Location
Handles:
1. Retrieval of existing images.
2. Pinpoint location of design/material/implementation on vehicle (inferred + optional Vision).
3. Engineering overlay with highlight drawn at pinpoint region.
4. Fallback AI Generation.

PATENT LOGIC - EMBODIMENT C (Coordinate Inference):
The system calculates ROI(x,y,w,h) = T(Img, c_text) to define the specific
implementation locus of the engineering change.

Differential Overlay Equation:
    Img_output = Img_base (+) L_vec(ROI, delta_val)
    Where (+) is the composition operator and L_vec is the SVG/Vector annotation layer.
"""
import os
import logging
import requests
import random
import re
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
from urllib.parse import quote
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Pinpoint regions on vehicle image: (center_x, center_y, width_ratio, height_ratio) in 0-1.
# Typical vehicle photo: top=hood, middle=doors, bottom=wheels/ground.
COMPONENT_PINPOINT_REGIONS = {
    # Braking / wheels (lower third, center or wheel wells)
    "brake": (0.5, 0.78, 0.4, 0.22),
    "caliper": (0.5, 0.78, 0.35, 0.2),
    "rotor": (0.5, 0.78, 0.35, 0.2),
    "wheel": (0.5, 0.8, 0.35, 0.2),
    "tire": (0.5, 0.82, 0.4, 0.18),
    "hub": (0.5, 0.75, 0.25, 0.2),
    # Doors / body side
    "door": (0.5, 0.5, 0.45, 0.45),
    "panel": (0.5, 0.5, 0.5, 0.5),
    "trim": (0.5, 0.55, 0.4, 0.4),
    "fender": (0.35, 0.45, 0.25, 0.4),
    "quarter": (0.65, 0.5, 0.25, 0.45),
    # Front / hood
    "hood": (0.5, 0.28, 0.55, 0.28),
    "bonnet": (0.5, 0.28, 0.55, 0.28),
    "bumper": (0.5, 0.88, 0.65, 0.14),
    "grille": (0.5, 0.22, 0.4, 0.18),
    # Lighting
    "headlamp": (0.35, 0.2, 0.2, 0.15),
    "headlight": (0.35, 0.2, 0.2, 0.15),
    "lamp": (0.5, 0.2, 0.35, 0.18),
    "taillight": (0.35, 0.82, 0.2, 0.12),
    "lighting": (0.5, 0.2, 0.4, 0.2),
    # Suspension / underbody
    "suspension": (0.5, 0.72, 0.4, 0.28),
    "damper": (0.5, 0.7, 0.3, 0.25),
    "strut": (0.5, 0.68, 0.28, 0.3),
    "spring": (0.5, 0.72, 0.25, 0.25),
    "control arm": (0.5, 0.75, 0.35, 0.2),
    "subframe": (0.5, 0.78, 0.5, 0.2),
    # Interior / cabin
    "seat": (0.5, 0.45, 0.4, 0.35),
    "console": (0.5, 0.5, 0.3, 0.25),
    "dashboard": (0.5, 0.35, 0.5, 0.25),
    "headliner": (0.5, 0.25, 0.5, 0.2),
    "interior": (0.5, 0.45, 0.5, 0.45),
    # Engine / HVAC
    "engine": (0.5, 0.25, 0.45, 0.3),
    "hvac": (0.5, 0.35, 0.35, 0.25),
    "blower": (0.5, 0.32, 0.25, 0.2),
    "exhaust": (0.5, 0.82, 0.4, 0.18),
    # General / material (center)
    "material": (0.5, 0.5, 0.45, 0.45),
    "component": (0.5, 0.5, 0.4, 0.4),
    "assembly": (0.5, 0.5, 0.45, 0.45),
    "part": (0.5, 0.5, 0.35, 0.35),
}

class VLMEngine:
    def __init__(self, db_conn_func, faiss_index, sentence_model):
        self.db_conn_func = db_conn_func
        self.faiss_index = faiss_index
        self.sentence_model = sentence_model
        
        # Paths
        self.base_dir = Path(__file__).parent.resolve()
        self.static_gen_dir = self.base_dir / "static" / "generated"
        self.static_gen_dir.mkdir(parents=True, exist_ok=True)
        
        # Circuit Breaker state
        self._api_cooldown_until = 0

    def get_images_for_idea(self, idea_text: str, origin: str, extra_context: dict = None) -> dict:
        images = {
            'current_scenario_image': None,
            'proposal_scenario_image': None,
            'competitor_image': 'static/defaults/competitor_placeholder.jpg'
        }
        if not extra_context:
            extra_context = {}

        # 1. RESOLVE CURRENT IMAGE
        # Try to get it from context (Agent findings)
        current_img = extra_context.get('mg_vehicle_image')
        
        # If Agent returned None or "NaN", GENERATE a "Current" state image
        if not current_img or "NaN" in str(current_img):
            visual_prompt = extra_context.get('visual_prompt', idea_text)
            clean_prompt = f"Standard automotive {visual_prompt}, realistic photography"
            current_img = self._generate_cloud_image_pollinations(clean_prompt)
        
        images['current_scenario_image'] = current_img

        # 2. RESOLVE COMPETITOR IMAGE
        comp_img = extra_context.get('competitor_image')
        
        # FIX: Added Generation Fallback similar to 'Current Scenario'
        if not comp_img or "NaN" in str(comp_img):
             # Only generate if Agent failed to find a DB image
             visual_prompt = extra_context.get('visual_prompt', idea_text)
             clean_prompt = f"Competitor brand automotive {visual_prompt}, studio lighting"
             comp_img = self._generate_cloud_image_pollinations(clean_prompt)
        
        images['competitor_image'] = comp_img

        # 3. GENERATE PROPOSAL (Overlay-style visualization)
        try:
            vehicle_name = extra_context.get("vehicle_name", "Vehicle")
            component_hint = extra_context.get("visual_prompt") or extra_context.get("component_key") or ""

            # 3A. If we have both current and competitor images, try smart overlay composition.
            # This extracts the key component region from the competitor image and visually
            # lays it over the current MG vehicle to help engineers compare.
            try_comp_overlay = extra_context.get("enable_competitor_overlay", True)
            if (
                try_comp_overlay
                and images.get('current_scenario_image')
                and images.get('competitor_image')
                and "NaN" not in str(images['competitor_image'])
            ):
                overlay_path = self._compose_competitor_overlay(
                    current_image_ref=images['current_scenario_image'],
                    competitor_image_ref=images['competitor_image'],
                    idea_text=idea_text,
                    extra_context=extra_context
                )
                # If overlay succeeded, use it as the proposal visualization directly.
                if overlay_path:
                    images['proposal_scenario_image'] = overlay_path
                    return images

            # For AI ideas, prefer prompt-engineered CAD overlay visuals (instead of photo annotation).
            if origin == "AI Innovation":
                overlay_prompt = self._construct_overlay_prompt(
                    idea_text=idea_text,
                    vehicle_name=vehicle_name,
                    component_hint=str(component_hint)
                )
                images['proposal_scenario_image'] = self._generate_cloud_image_pollinations(overlay_prompt)
            else:
                # For DB/Web, keep the existing local overlay annotation path when possible.
                base_image_rel_path = images['current_scenario_image']
                if isinstance(base_image_rel_path, str) and base_image_rel_path.startswith("/"):
                    base_image_rel_path = base_image_rel_path[1:]
                base_image_abs_path = self.base_dir / str(base_image_rel_path)

                if base_image_abs_path.exists():
                    images['proposal_scenario_image'] = self._create_engineering_annotation(
                        base_image_path=base_image_abs_path,
                        idea_text=idea_text,
                        extra_context=extra_context
                    )
                else:
                    images['proposal_scenario_image'] = self._generate_cloud_image_pollinations(f"Concept sketch of {idea_text}")

        except Exception as e:
            logger.error(f"VLM Generation Error: {e}")
            images['proposal_scenario_image'] = self._generate_cloud_image_pollinations(idea_text)

        return images

    def _generate_cloud_image_pollinations(self, prompt: str) -> str:
        """
        Generates an image using Pollinations.ai (No API Key required).
        """
        try:
            # Clean prompt
            clean_prompt = re.sub(r'[^a-zA-Z0-9, ]', '', prompt)[:400]
            encoded = quote(clean_prompt)
            url = f"https://image.pollinations.ai/prompt/{encoded}?nologo=true"
            
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                safe_name = "".join(x for x in prompt[:15] if x.isalnum())
                timestamp = int(time.time() * 1000)
                filename = f"ai_gen_{safe_name}_{timestamp}.jpg"
                save_path = self.static_gen_dir / filename
                
                with open(save_path, "wb") as f:
                    f.write(response.content)
                
                logger.info(f"Generated AI Backup Image: {filename}")
                return f"static/generated/{filename}"
        except Exception as e:
            logger.warning(f"Pollinations AI Generation failed: {e}")
        
        return "static/defaults/ai_placeholder.jpg"

    def _infer_pinpoint_region(self, idea_text: str, extra_context: dict = None) -> dict:
        """
        Infer pinpoint location (design/material/implementation) on vehicle from idea text and context.
        Returns dict: center_x, center_y, width_ratio, height_ratio (0-1).
        """
        extra_context = extra_context or {}
        # 1. Explicit region from context (e.g. from UI or Vision API)
        region = extra_context.get("pinpoint_region")
        if isinstance(region, (list, tuple)) and len(region) >= 4:
            cx, cy, w, h = float(region[0]), float(region[1]), float(region[2]), float(region[3])
            return {"center_x": cx, "center_y": cy, "width_ratio": w, "height_ratio": h}
        # 2. Optional: Gemini Vision to get region from image
        current_img = extra_context.get("mg_vehicle_image") or extra_context.get("current_scenario_image")
        if current_img and os.getenv("GOOGLE_API_KEY"):
            vision_region = self._get_pinpoint_from_vision(current_img, idea_text, extra_context)
            if vision_region:
                return vision_region
        # 3. Keyword-based: match idea + visual_prompt to component region
        text = f" {(idea_text or '').lower()} {(extra_context.get('visual_prompt') or '').lower()} "
        for keyword, (cx, cy, w, h) in COMPONENT_PINPOINT_REGIONS.items():
            if keyword in text:
                return {"center_x": cx, "center_y": cy, "width_ratio": w, "height_ratio": h}
        # 4. Default: center of image
        return {"center_x": 0.5, "center_y": 0.5, "width_ratio": 0.4, "height_ratio": 0.4}

    def _get_pinpoint_from_vision(self, image_ref: str, idea_text: str, extra_context: dict = None) -> dict:
        """
        [DISABLED] Google Vision API call removed for optimization.
        Always returns None to force usage of _infer_pinpoint_region (heuristic fallback).
        """
        return None
        # --- CACHE CHECK ---
        try:
            import llm_cache
            # Create a deterministic key from image path and idea text
            cache_key_prompt = f"PINPOINT_VISION::{image_ref}::{idea_text}"[:4000] # Limit length
            cached_json = llm_cache.get_cached_response(cache_key_prompt, "VISION_API", "gemini-vision")
            if cached_json:
                import json
                return json.loads(cached_json)
        except Exception as e:
            logger.warning(f"Vision Cache Read Failed: {e}")

        try:
            import google.generativeai as genai
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                return None
            genai.configure(api_key=api_key)
            
            path = self._resolve_local_image(image_ref)
            if not path or not path.exists():
                return None
            with Image.open(path) as img:
                img = img.convert("RGB")
            if max(img.size) > 1024:
                img.thumbnail((1024, 1024), Image.LANCZOS)
            # Save to temp file so we can use genai.upload_file (project pattern)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                img.save(tmp, format="JPEG", quality=90)
                tmp_path = tmp.name
                
            uploaded_file = None
            try:
                uploaded_file = genai.upload_file(tmp_path, mime_type="image/jpeg")
                prompt = (
                    f"The user is implementing a cost reduction / design change on a vehicle. "
                    f"Description: {idea_text[:400]}.\n\n"
                    "Where on this vehicle image is the component or area that this change applies to? "
                    "Reply with ONLY four numbers in one line, space-separated: center_x_ratio center_y_ratio width_ratio height_ratio "
                    "(each 0.0 to 1.0, where 0.5 0.5 is image center). Example: 0.5 0.75 0.35 0.2"
                )
                
                # Retry across models if one is deprecated/missing
                candidate_models = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-3-flash"]
                last_error = None

                for model_name in candidate_models:
                    try:
                        model = genai.GenerativeModel(model_name)
                        response = model.generate_content([uploaded_file, prompt])
                        if response and response.text:
                            # Parse result
                            nums = re.findall(r"0?\.\d+|1\.0", response.text)
                            if len(nums) >= 4:
                                cx, cy, w, h = float(nums[0]), float(nums[1]), float(nums[2]), float(nums[3])
                                cx, cy = max(0, min(1, cx)), max(0, min(1, cy))
                                w, h = max(0.05, min(0.9, w)), max(0.05, min(0.9, h))
                                result = {"center_x": cx, "center_y": cy, "width_ratio": w, "height_ratio": h}
                                
                                # --- WRITE CACHE ---
                                try:
                                    import json
                                    llm_cache.cache_response(cache_key_prompt, "VISION_API", "gemini-vision", json.dumps(result))
                                except Exception as ce:
                                    logger.warning(f"Vision Cache Write Failed: {ce}")
                                    
                                return result
                    except Exception as e:
                        last_error = e
                        if "404" in str(e) or "not found" in str(e).lower():
                            continue # Try next model
                        # If meaningful error, maybe log but continue
                        continue
                
                if last_error:
                    logger.debug(f"All Vision models failed for pinpoint. Last error: {last_error}")

            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Vision pinpoint failed (using keyword fallback): {e}")
        return None

    def _construct_overlay_prompt(self, idea_text: str, vehicle_name: str, component_hint: str = "") -> str:
        """
        Builds an "engineering overlay" style prompt to simulate CAD/schematic proposal visuals.
        """
        # Keep it short and directive to improve consistency from generators.
        change = idea_text.strip()
        if len(change) > 160:
            change = change[:160].rsplit(" ", 1)[0] + "..."

        component = component_hint.strip()
        if not component:
            component = "component"

        return (
            f"Technical engineering CAD overlay of {component} for {vehicle_name}, "
            f"highlighting the proposed change: {change}. "
            "Schematic wireframe style, mounting points highlighted in neon blue, "
            "white background, clean annotations, no people, no branding."
        )

    def _create_engineering_annotation(self, base_image_path, idea_text, extra_context=None):
        """
        Draws professional CAD-style annotations with PINPOINT LOCATION of design/material/implementation
        on the vehicle. Infers region from idea text (and optional Vision), draws highlight box at that
        location, and places measurement/annotation at the pinpoint.
        """
        extra_context = extra_context or {}
        try:
            with Image.open(base_image_path) as img:
                img = img.convert("RGBA")
            width, height = img.size
            overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            # 1. Infer pinpoint region (design/material/implementation location on vehicle)
            region = self._infer_pinpoint_region(idea_text, extra_context)
            cx_n, cy_n = region["center_x"], region["center_y"]
            w_n, h_n = region["width_ratio"], region["height_ratio"]
            center_x = int(width * cx_n)
            center_y = int(height * cy_n)
            box_w = int(width * w_n)
            box_h = int(height * h_n)
            x1 = max(0, center_x - box_w // 2)
            y1 = max(0, center_y - box_h // 2)
            x2 = min(width, x1 + box_w)
            y2 = min(height, y1 + box_h)

            # 2. Draw PINPOINT HIGHLIGHT (neon-style box at implementation location)
            outline_color = (0, 255, 255, 255)  # Cyan
            outline_width = max(3, int(min(width, height) * 0.008))
            for d in range(outline_width):
                draw.rectangle([(x1 - d, y1 - d), (x2 + d, y2 + d)], outline=outline_color)
            # Semi-transparent fill so vehicle still visible
            draw.rectangle([(x1, y1), (x2, y2)], fill=(0, 255, 255, 35))
            # Label: "DESIGN / MATERIAL CHANGE HERE" above or beside the box
            label = "DESIGN / IMPLEMENTATION LOCATION"
            try:
                font_label = ImageFont.truetype("arialbd.ttf", max(12, int(height * 0.025)))
            except Exception:
                font_label = ImageFont.load_default()
            ly = max(10, y1 - int(height * 0.04))
            lx = max(10, min(x1, width - 320))
            draw.rectangle([(lx - 4, ly - 2), (lx + 310, ly + 22)], fill=(20, 20, 20, 230))
            draw.text((lx, ly), label, font=font_label, fill=(0, 255, 255, 255))

            # 3. Analyze Text for Numbers (The "Smart" Part)
            # Regex to find patterns like "160 to 110", "170 - 100", "reduces from X to Y"
            # Looks for: number + optional space + unit + spacer + number + optional space + unit
            
            # Pattern: (Number) ... (to/from/-) ... (Number)
            nums = re.findall(r'(\d+(?:\.\d+)?)', idea_text)
            units = re.findall(r'(mm|cm|kg|g|um|μm|%)', idea_text.lower())
            main_unit = units[0] if units else ""
            
            # Determine Before/After values
            val_old = "?"
            val_new = "?"
            
            if len(nums) >= 2:
                # Heuristic: Usually "from High to Low" for reduction
                v1, v2 = float(nums[0]), float(nums[1])
                if "reduc" in idea_text.lower():
                    val_old = str(max(v1, v2))
                    val_new = str(min(v1, v2))
                else:
                    val_old = str(v1)
                    val_new = str(v2)
            elif len(nums) == 1:
                val_new = nums[0]

            # 3. Setup Fonts (Try to load a nice font, fallback to default)
            font_size_header = int(height * 0.05) # 5% of image height
            font_size_body = int(height * 0.03)
            try:
                # Windows standard font
                font_header = ImageFont.truetype("arialbd.ttf", font_size_header)
                font_body = ImageFont.truetype("arial.ttf", font_size_body)
            except:
                font_header = ImageFont.load_default()
                font_body = ImageFont.load_default()

            # 4. Draw The "Implementation Card" (Bottom or Side)
            # Create a semi-transparent dark pane at the bottom
            pane_h = int(height * 0.25)
            draw.rectangle([(0, height - pane_h), (width, height)], fill=(20, 30, 40, 220))
            
            # Draw Title
            text_x = int(width * 0.05)
            text_y = height - pane_h + int(pane_h * 0.1)
            draw.text((text_x, text_y), "ENGINEERING IMPLEMENTATION", font=font_header, fill=(255, 165, 0, 255)) # Orange Title

            # Draw "Before -> After" if numbers exist
            text_y += int(pane_h * 0.3)
            if val_old != "?" and val_new != "?":
                # Draw the "Visual Scale"
                # "Before: 160um"  ---Arrow--->  "After: 100um"
                
                # Old Value (Red)
                draw.text((text_x, text_y), f"CURRENT: {val_old}{main_unit}", font=font_body, fill=(255, 100, 100, 255))
                
                # Arrow Symbol
                arrow_text = "  ➔  "
                w_old = draw.textlength(f"CURRENT: {val_old}{main_unit}", font=font_body)
                draw.text((text_x + w_old, text_y), arrow_text, font=font_body, fill=(255, 255, 255, 255))
                
                # New Value (Green)
                w_arrow = draw.textlength(arrow_text, font=font_body)
                draw.text((text_x + w_old + w_arrow, text_y), f"PROPOSED: {val_new}{main_unit}", font=font_body, fill=(100, 255, 100, 255))
                
                # 5. Draw On-Component Annotation at PINPOINT (measurement at design/material location)
                # Use inferred pinpoint center so the measurement line sits on the implementation area
                line_len = int(width * 0.18)
                # Clamp line to image bounds
                ly1 = max(0, center_y - line_len // 2)
                ly2 = min(height, center_y + line_len // 2)
                # Vertical Measurement Bar (caliper style) at pinpoint
                draw.line([(center_x, ly1), (center_x, ly2)], fill=(255, 255, 0, 220), width=3)
                cap_len = 20
                draw.line([(center_x - cap_len, ly1), (center_x + cap_len, ly1)], fill=(255, 255, 0, 220), width=3)
                draw.line([(center_x - cap_len, ly2), (center_x + cap_len, ly2)], fill=(255, 255, 0, 220), width=3)
                draw.text((center_x + cap_len + 10, center_y - 10), f"↓ {val_new}{main_unit}", font=font_header, fill=(255, 255, 0, 255))
                
                # Add a "Reduction" badge
                try:
                    diff = float(val_old) - float(val_new)
                    if diff > 0:
                        pct = (diff / float(val_old)) * 100
                        badge_text = f"-{int(pct)}%"
                        # Draw circle badge
                        badge_r = int(height * 0.06)
                        badge_x, badge_y = int(width * 0.85), int(height * 0.15)
                        draw.ellipse([(badge_x-badge_r, badge_y-badge_r), (badge_x+badge_r, badge_y+badge_r)], fill=(255, 50, 50, 255), outline=(255,255,255,255), width=2)
                        # Center text in badge
                        w_b = draw.textlength(badge_text, font=font_header)
                        draw.text((badge_x - w_b/2, badge_y - font_size_header/2), badge_text, font=font_header, fill=(255, 255, 255, 255))
                except:
                    pass

            else:
                # If no numbers found, just print the idea text nicely
                # Wrap text
                max_chars = 50
                wrapped_text = ""
                words = idea_text.split()
                line = ""
                for word in words:
                    if len(line + word) < max_chars:
                        line += word + " "
                    else:
                        wrapped_text += line + "\n"
                        line = word + " "
                wrapped_text += line
                
                draw.text((text_x, text_y), wrapped_text, font=font_body, fill=(220, 220, 220, 255))

            # 6. Composite and Save
            final_img = Image.alpha_composite(img, overlay).convert("RGB")
            
            # Create Filename
            safe_name = "".join(x for x in idea_text[:15] if x.isalnum())
            timestamp = int(time.time() * 1000)
            filename = f"impl_overlay_{safe_name}_{timestamp}.jpg"
            save_path = self.static_gen_dir / filename
            
            final_img.save(save_path, quality=90)
            logger.info(f"Generated Engineering Annotation: {filename}")
            
            return f"static/generated/{filename}"

        except Exception as e:
            logger.error(f"Annotation Failed: {e}", exc_info=True)
            # Fallback to AI gen if annotation fails
            return self._generate_cloud_image_pollinations(idea_text)

    def _resolve_local_image(self, image_ref: str):
        """
        Resolves a static/relative image reference (e.g. '/static/images/mg/x.jpg'
        or 'static/generated/x.jpg') into an absolute filesystem path.
        Returns None for HTTP URLs or invalid values.
        """
        if not image_ref:
            return None

        # Ignore remote URLs – current overlay only works with local files
        ref_str = str(image_ref)
        if ref_str.startswith("http://") or ref_str.startswith("https://"):
            return None

        rel = ref_str
        # Strip leading slashes (e.g. '/static/...' -> 'static/...')
        if rel.startswith("/"):
            rel = rel[1:]

        return self.base_dir / rel

    def _compose_competitor_overlay(self, current_image_ref: str, competitor_image_ref: str, idea_text: str, extra_context: dict = None) -> str:
        """
        Core "engineering overlay" for competitor comparison:
        - Takes the competitor scenario image
        - Extracts the component region (optionally using a normalized crop from context)
        - Soft-masks and lays it over the current scenario image.

        This is intentionally lightweight and works purely with PIL so that it
        can run inside the existing VAVE deployment without heavy new models.
        """
        extra_context = extra_context or {}

        try:
            current_path = self._resolve_local_image(current_image_ref)
            comp_path = self._resolve_local_image(competitor_image_ref)

            if not current_path or not comp_path or not current_path.exists() or not comp_path.exists():
                logger.warning(f"VLM overlay skipped – missing local files: current={current_path}, competitor={comp_path}")
                return None

            with Image.open(current_path) as base_img:
                base_img = base_img.convert("RGBA")

            with Image.open(comp_path) as comp_img:
                comp_img = comp_img.convert("RGBA")

            bw, bh = base_img.size
            cw, ch = comp_img.size

            # 1. Determine the component region on the competitor image
            #    a) If UI / agent provided a normalized crop [x, y, w, h] in 0–1 coords, use that
            crop_box = extra_context.get("competitor_crop")  # e.g. [0.2, 0.3, 0.4, 0.25]
            if (
                isinstance(crop_box, (list, tuple))
                and len(crop_box) == 4
                and all(isinstance(v, (int, float)) for v in crop_box)
            ):
                nx, ny, nw, nh = crop_box
                nx = max(0.0, min(1.0, float(nx)))
                ny = max(0.0, min(1.0, float(ny)))
                nw = max(0.01, min(1.0, float(nw)))
                nh = max(0.01, min(1.0, float(nh)))

                left = int(cw * nx)
                top = int(ch * ny)
                right = int(cw * (nx + nw))
                bottom = int(ch * (ny + nh))
                right = max(left + 1, min(cw, right))
                bottom = max(top + 1, min(ch, bottom))
            else:
                #    b) Fallback: Update to "Smart Context Crop" (Antigravity Protocol)
                #       Instead of small center crop, use a safe 80% crop logic
                #       [0.1, 0.1, 0.9, 0.9] -> x=10%, y=10%, w=80%, h=80%
                nx, ny, nw, nh = 0.1, 0.1, 0.8, 0.8
                
                left = int(cw * nx)
                top = int(ch * ny)
                right = int(cw * (nx + nw))
                bottom = int(ch * (ny + nh))

            comp_region = comp_img.crop((left, top, right, bottom))

            # 2. Resize the component patch relative to the current scenario
            target_width_ratio = float(extra_context.get("overlay_width_ratio", 0.35))
            target_width_ratio = max(0.1, min(0.8, target_width_ratio))

            target_w = int(bw * target_width_ratio)
            scale = target_w / max(1, comp_region.size[0])
            target_h = int(comp_region.size[1] * scale)
            comp_region = comp_region.resize((target_w, target_h), Image.LANCZOS)

            # 3. Create a soft alpha mask so the pasted component looks like a layer
            mask = Image.new("L", comp_region.size, 0)
            m_draw = ImageDraw.Draw(mask)
            mw, mh = comp_region.size
            inset_x = int(mw * 0.05)
            inset_y = int(mh * 0.05)
            m_draw.rectangle(
                (inset_x, inset_y, mw - inset_x, mh - inset_y),
                fill=255
            )
            mask = mask.filter(ImageFilter.GaussianBlur(radius=max(2, int(min(mw, mh) * 0.04))))
            comp_region.putalpha(mask)

            # 4. Decide where to place the component on the current image
            #    Allow normalized (x, y) in 0–1 via "overlay_position", default: bottom-right
            pos = extra_context.get("overlay_position")
            if (
                isinstance(pos, (list, tuple))
                and len(pos) == 2
                and all(isinstance(v, (int, float)) for v in pos)
            ):
                ox = int(bw * float(pos[0]))
                oy = int(bh * float(pos[1]))
            else:
                margin_x = int(bw * 0.05)
                margin_y = int(bh * 0.05)
                ox = bw - comp_region.size[0] - margin_x
                oy = bh - comp_region.size[1] - margin_y

            ox = max(0, min(bw - comp_region.size[0], ox))
            oy = max(0, min(bh - comp_region.size[1], oy))

            composed = base_img.copy()
            composed.alpha_composite(comp_region, dest=(ox, oy))

            # 5. Save composed proposal image into static/generated
            safe_name = "".join(x for x in idea_text[:15] if x.isalnum())
            timestamp = int(time.time() * 1000)
            filename = f"comp_overlay_{safe_name}_{timestamp}.jpg"
            save_path = self.static_gen_dir / filename

            composed.convert("RGB").save(save_path, quality=92)
            logger.info(f"Generated competitor overlay proposal: {filename}")

            return f"static/generated/{filename}"

        except Exception as e:
            logger.error(f"Competitor overlay generation failed: {e}", exc_info=True)
            return None

        return 'static/defaults/ai_placeholder.jpg'

    def _get_component_bbox_from_vision(self, image_ref: str, component_name: str) -> list:
        """
        [DISABLED] Vision API BBox detection removed. 
        Returns None to force usage of safe center crops or heuristics.
        """
        return None
        # 1. Circuit Breaker Check
        if time.time() < self._api_cooldown_until:
             logger.warning("Vision API Circuit Breaker active. Skipping API call.")
             return None

        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                import google.generativeai as genai
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key: return None
                genai.configure(api_key=api_key)
                
                path = self._resolve_local_image(image_ref)
                if not path or not path.exists():
                    return None

                with Image.open(path) as img:
                    # Resize if too large to save bandwidth/latentcy
                    if max(img.size) > 1024:
                        img.thumbnail((1024, 1024), Image.LANCZOS)
                    
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                        img.save(tmp, format="JPEG")
                        tmp_path = tmp.name
                
                uploaded_file = None
                try:
                    uploaded_file = genai.upload_file(tmp_path, mime_type="image/jpeg")
                    
                    candidate_models = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-3-flash"]
                    prompt = (
                        f"Return the bounding box for the '{component_name}' in this image. "
                        "Output ONLY four numbers: ymin, xmin, ymax, xmax (normalized 0 to 1). "
                        "If not found, return empty."
                    )

                    for model_name in candidate_models:
                        try:
                            model = genai.GenerativeModel(model_name)
                            response = model.generate_content([uploaded_file, prompt])
                            if response and response.text:
                                nums = re.findall(r"0?\.\d+|1\.0", response.text)
                                if len(nums) >= 4:
                                    return [float(n) for n in nums[:4]]
                        except Exception as e:
                            # 503/429/Socket check within the model loop? 
                            # Actually, we treat model errors as "try next model". 
                            # If ALL models fail, we hit the outer loop exception? 
                            # For simplicity, if a model specific error occurs, we try next model.
                            # If connection error occurs, we might want to wait.
                            # If connection error occurs, we might want to wait.
                            err_str = str(e).lower()
                            if "503" in err_str or "connection" in err_str:
                                raise e # Re-raise to trigger outer loop backoff
                            
                            # CIRCUIT BREAKER TRIGGER
                            if "429" in err_str or "quota" in err_str or "403" in err_str:
                                logger.warning(f"Vision API Quota Exceeded ({e}). Tripping Circuit Breaker for 60s.")
                                self._api_cooldown_until = time.time() + 60
                                return None

                            if "404" in err_str or "not found" in err_str:
                                continue
                            logger.debug(f"Model {model_name} failed: {e}")
                            
                finally:
                    try: os.unlink(tmp_path) 
                    except: pass
                
                # If we get here without returning, no models worked for this attempt
                # If it wasn't an exception, it was just "no valid response"
                
            except Exception as e:
                logger.warning(f"Vision BBox attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    sleep_time = 2 * (attempt + 1)
                    time.sleep(sleep_time)
                else:
                    logger.error("All Vision BBox attempts failed.")
        
        return None

    def _smart_merge_images_cv(self, current_path, comp_path, current_bbox, comp_bbox):
        """
        Antigravity Visual Fidelity:
        Uses OpenCV Poisson Blending (SeamlesClone) to merge component.
        """
        try:
            import cv2
            import numpy as np

            # Load images
            img_curr = cv2.imread(str(current_path))
            img_comp = cv2.imread(str(comp_path))

            if img_curr is None or img_comp is None:
                return None

            h_c, w_c = img_curr.shape[:2]
            h_s, w_s = img_comp.shape[:2]

            # 1. Extract Source Component
            # comp_bbox is [ymin, xmin, ymax, xmax]
            y1, x1, y2, x2 = comp_bbox
            sy1, sx1 = int(y1 * h_s), int(x1 * w_s)
            sy2, sx2 = int(y2 * h_s), int(x2 * w_s)
            
            # Clamp
            sy1, sx1 = max(0, sy1), max(0, sx1)
            sy2, sx2 = min(h_s, sy2), min(w_s, sx2)
            
            if (sy2 - sy1) < 10 or (sx2 - sx1) < 10:
                return None

            src_roi = img_comp[sy1:sy2, sx1:sx2]
            src_h, src_w = src_roi.shape[:2]

            # 2. Determine Target Region
            # current_bbox is center_x, center_y, w, h
            cx, cy = current_bbox["center_x"], current_bbox["center_y"]
            bw, bh = current_bbox["width_ratio"], current_bbox["height_ratio"]
            
            t_cx = int(w_c * cx)
            t_cy = int(h_c * cy)
            
            # Target Dimensions logic (Aspect Ratio Preserved)
            # instead of forcing fit to target_w/target_h, we fit INSIDE or COVER
            # Let's target a width match mainly
            target_roi_w = int(w_c * bw)
            # Calculate scale to match target width
            scale = target_roi_w / max(1, src_w)
            
            # Limit scale to prevent explosion
            scale = min(scale, 2.0)
            
            new_w = int(src_w * scale)
            new_h = int(src_h * scale)
            
            # Resize Source
            src_resized = cv2.resize(src_roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # 3. Prepare for Seamless Clone
            # Create a Mask (Soft Ellipse for organic blending)
            mask = np.zeros(src_resized.shape, dtype=np.uint8)
            # Draw white ellipse in center
            center = (new_w // 2, new_h // 2)
            axes = (int(new_w * 0.45), int(new_h * 0.45))
            cv2.ellipse(mask, center, axes, 0, 0, 360, (255, 255, 255), -1)
            
            # Blur the mask slightly
            mask = cv2.GaussianBlur(mask, (21, 21), 10)
            
            # Center point on target image
            center_point = (t_cx, t_cy)
            
            # Boundary checks for seamlessClone
            # src, mask, and center must allow placement roughly within img
            # OpenCV crashes if the source+center goes totally out of bounds.
            # Simple check:
            tl_x = t_cx - new_w // 2
            tl_y = t_cy - new_h // 2
            br_x = tl_x + new_w
            br_y = tl_y + new_h
            
            # If completely out, shift it back or abort
            # Realistically seamlessClone handles partial overlap well, but let's be safe
            if tl_x < 0: center_point = (new_w // 2, center_point[1])
            if tl_y < 0: center_point = (center_point[0], new_h // 2)
            if br_x > w_c: center_point = (w_c - new_w // 2, center_point[1])
            if br_y > h_c: center_point = (center_point[0], h_c - new_h // 2)

            # 4. Seamless Clone
            try:
                # MIXED_CLONE often looks better for parts integration
                # NORMAL_CLONE is safer for "covering" old parts
                blended = cv2.seamlessClone(src_resized, img_curr, mask, center_point, cv2.NORMAL_CLONE)
            except Exception as cv_err:
                logger.warning(f"SeamlessClone failed ({cv_err}), using manual overlay.")
                # Manual Fallback
                blended = img_curr.copy()
                # Simple paste with mask
                # Need to calculate exact ROI coordinates again based on center_point
                cpx, cpy = center_point
                tx1 = cpx - new_w // 2
                ty1 = cpy - new_h // 2
                tx2 = tx1 + new_w
                ty2 = ty1 + new_h
                
                # Clip to image bounds
                tx1_c, ty1_c = max(0, tx1), max(0, ty1)
                tx2_c, ty2_c = min(w_c, tx2), min(h_c, ty2)
                
                # Corresponding source coords
                sx1_c = tx1_c - tx1
                sy1_c = ty1_c - ty1
                sx2_c = sx1_c + (tx2_c - tx1_c)
                sy2_c = sy1_c + (ty2_c - ty1_c)
                
                if (tx2_c > tx1_c) and (ty2_c > ty1_c):
                    # Alpha blend manually
                    alpha = mask[sy1_c:sy2_c, sx1_c:sx2_c].astype(float) / 255.0
                    src_patch = src_resized[sy1_c:sy2_c, sx1_c:sx2_c].astype(float)
                    bg_patch = blended[ty1_c:ty2_c, tx1_c:tx2_c].astype(float)
                    
                    final_patch = (src_patch * alpha + bg_patch * (1.0 - alpha)).astype(np.uint8)
                    blended[ty1_c:ty2_c, tx1_c:tx2_c] = final_patch

            # Save result
            filename = f"cv_merge_{int(time.time()*1000)}.jpg"
            save_path = self.static_gen_dir / filename
            cv2.imwrite(str(save_path), blended)
            
            return f"static/generated/{filename}"

        except Exception as e:
            logger.error(f"CV Merge failed: {e}")
            return None

    def _compose_competitor_overlay(self, current_image_ref: str, competitor_image_ref: str, idea_text: str, extra_context: dict = None) -> str:
        """
        Enhanced "Smart Vision" Overlay:
        1. Detect component in Competitor Image.
        2. Detect target location in Current Image.
        3. Use Computer Vision to merge.
        """
        extra_context = extra_context or {}
        
        try:
            current_path = self._resolve_local_image(current_image_ref)
            comp_path = self._resolve_local_image(competitor_image_ref)

            if not current_path or not comp_path or not current_path.exists() or not comp_path.exists():
                return None

            # 1. Component Detection (Vision)
            idea_key = extra_context.get("visual_prompt") or idea_text[:20]
            
            # Try to get bbox for the KEY COMPONENT in the COMPETITOR image
            comp_bbox = self._get_component_bbox_from_vision(competitor_image_ref, idea_key)
            
            if not comp_bbox:
                # ANTIGRAVITY PROTOCOL: Fail-Safe "Gravity Assist"
                # If Vision fails, we LIFT the center component anyway.
                # We assume the "Center 60%" of the competitor image is the part.
                logger.warning(f"Smart Overlay: Vision failed for '{idea_key}'. Engaging Gravity Assist (Geometric Fallback).")
                comp_bbox = [0.2, 0.2, 0.8, 0.8] # [ymin, xmin, ymax, xmax]
            
            # 2. Target Location Detection (Vision or Context)
            # Use existing inference logic which tries context -> Vision -> keywords
            target_region = self._infer_pinpoint_region(idea_text, {"mg_vehicle_image": current_image_ref, **extra_context})
            
            # 3. Perform Merge
            # Try OpenCV first
            merged_path = self._smart_merge_images_cv(current_path, comp_path, target_region, comp_bbox)
            
            if merged_path:
                logger.info(f"Generated Smart CV Overlay: {merged_path}")
                return merged_path
            
            # Fallback to PIL overlay if CV fails
            logger.warning("Falling back to legacy PIL overlay")
            return self._compose_competitor_overlay_legacy(current_path, comp_path, target_region, comp_bbox)

        except Exception as e:
            logger.error(f"Smart Overlay Loop Failed: {e}", exc_info=True)
            return None

    def _compose_competitor_overlay_legacy(self, current_path, comp_path, target_region, comp_source_bbox):
        """
        Original PIL-based logic, refactored to accept bboxes.
        """
        # ... (simplified legacy logic using crop/paste) ...
        # For now, just return None to force generating a simple one or failing gracefully
        # Or re-implement the PIL logic quickly:
        try:
            with Image.open(current_path) as base_img:
                base_img = base_img.convert("RGBA")
            with Image.open(comp_path) as comp_img:
                comp_img = comp_img.convert("RGBA")
            
            # Crop Source
            # Ensure bbox indices are valid floats
            ys, xs, ye, xe = comp_source_bbox
            w_s, h_s = comp_img.size
            
            # Clamp to 0-1 just in case
            ys, xs = max(0, min(1, ys)), max(0, min(1, xs))
            ye, xe = max(0, min(1, ye)), max(0, min(1, xe))

            left, top = int(xs*w_s), int(ys*h_s)
            right, bottom = int(xe*w_s), int(ye*h_s)
            
            # Ensure valid dimensions
            if right <= left: right = left + 1
            if bottom <= top: bottom = top + 1
            
            comp_crop = comp_img.crop((left, top, right, bottom))
            
            # Target
            cx, cy = target_region["center_x"], target_region["center_y"]
            bw, bh = target_region["width_ratio"], target_region["height_ratio"]
            
            w_c, h_c = base_img.size
            target_w, target_h = int(w_c * bw), int(h_c * bh)
            
            comp_resized = comp_crop.resize((target_w, target_h), Image.LANCZOS)
            
            # Paste
            tx = int(w_c * cx - target_w // 2)
            ty = int(h_c * cy - target_h // 2)
            
            base_img.alpha_composite(comp_resized, (tx, ty))
            
            filename = f"legacy_overlay_{int(time.time())}.jpg"
            save_path = self.static_gen_dir / filename
            base_img.convert("RGB").save(save_path)
            return f"static/generated/{filename}"
        except:
            return None