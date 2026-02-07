I have balanced the retrieval logic to satisfy both "Strict Domain Safety" and "Compulsory Image Retrieval".

### **Adjustments Made:**

1.  **Relaxed Semantic Threshold (0.72 → 0.45)**:
    *   The "Perfect Match" threshold is now lowered to **0.45**. This allows "good enough" matches to pass through immediately.

2.  **Added "Desperate Fallback" Pass**:
    *   If no image meets the 0.45 threshold, the system now looks for *any* match with a score > **0.25** that **passes the domain guard**.
    *   *Result:* This ensures that if you ask for "Brake" and we have a "Brake Line" image with a lower similarity score, it will now be accepted instead of returning "Empty".

3.  **Restored Keyword Fallback**:
    *   I re-enabled the basic `_get_smart_image` text matcher as a "Last Resort".
    *   *Safety:* I added a basic "Domain Safety" check inside this keyword matcher too (so `brake` can't match `sunroof`), but it will happily match `caliper` or `disc`.

4.  **Preserved Domain Guard**:
    *   The negative keyword filter (e.g., Reject "Sunroof" for "Brake") remains active across ALL passes (Semantic High, Semantic Low, and Keyword). This prevents the "Gross Glitch" (Wires for Rotors) while allowing relevant images.

**Expected Outcome:**
*   **Significantly fewer "Empty/NaN" images.**
*   **No "Gross Hallucinations"** (e.g., Roof rails on Chassis).
*   **More "Acceptable" matches** (e.g., A generic brake system image for a specific rotor idea).

Please re-run the pipeline.
