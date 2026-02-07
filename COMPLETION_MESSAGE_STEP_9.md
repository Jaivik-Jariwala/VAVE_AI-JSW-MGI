I have implemented the **"Zero Truncation / Auto-Fit" logic**.

**How the new Engine Works:**
1.  **Pass 1 (Analyze):** The system first calculates the total height needed for *all* text content at standard 10pt font.
2.  **Pass 2 (Scale):** If the content is too long for the slide:
    *   It calculates a "Shrink Factor" (e.g., reduce by 15%).
    *   It reduces the font size dynamically (e.g., 10pt -> 8.5pt).
    *   It renders the text to fit exactly within the safe area.
3.  **Tighter Spacing:** I have reduced the gap between sections to **2mm** (0.08 inches), maximizing space for your content.

**Result:**
Generous text (like long Validation plans) will now simply appear slightly smaller to fit, rather than being cut off.

Please regenerate the PPT to see the result.
