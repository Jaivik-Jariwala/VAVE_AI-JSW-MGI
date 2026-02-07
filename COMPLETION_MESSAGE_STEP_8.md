I have resolved the Layout Overlap caused by long titles (like "Redesign the brake caliper mounting bracket...").

**The Fix:**
1.  **Adaptive Header Height**: The system now estimates the height of the Title Box based on character count (approx. 45 chars per line).
2.  **Dynamic Content Push**: The rest of the slide content ("Technical View", etc.) no longer starts at a fixed position. It calculates `Title Height + Padding` and starts strictly *below* the title.

This ensures that even if a title takes up 4 lines, it will simply push the Technical View down, rather than writing over it.

Please regenerate the PPT to see the clean, collision-free layout.
