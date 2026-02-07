I have completed the requested fixes for the VAVE AI Image Pipeline.

**Actions Taken:**

1.  **Strict Domain Validation (`agent.py`)**:
    *   **Strict Threshold**: Increased semantic match threshold to **0.72**.
    *   **Domain Guard**: Implemented logic to strictly reject images with negative keywords based on the target system (e.g., if looking for "Brake", it now aggressively rejects "sunroof", "wire", "interior", "roof rail").
    *   **Disabled Keyword Fallback**: Removed the loose "smart image" keyword matching to prevent blind hallucinations.

2.  **Fixed "Glitch" Images (`vlm_engine.py`)**:
    *   **Smart Overlay Validation**: In the `_compose_competitor_overlay` function, I removed the "fallback to center crop" logic.
    *   **Vision Check**: Now, if Gemini Vision cannot *specifically detect* the component (e.g., "Brake Caliper") in the competitor image, the overlay process **ABORTS** (returns `None`).
    *   **Result**: This prevents the system from pasting random parts (like a "roof rail") onto your brake concepts just because it had a file. It contributes to "cleaning up" the Implementation slides by showing only valid, verified overlays.

**Outcome:**
*   **AI-Generated Ideas**: Should no longer show unrelated "hallucinated" images (Wires for Rotors). If no valid image is found, it will fallback to "Image Not Available" or text-only, preserving credibility.
*   **Existing Ideas**: The "Implementation" slides will no longer have random inset glitches.

You can now re-run the pipeline to generate a clean report.
