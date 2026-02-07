I have fixed the `KeyError: 'competitor_image'` issue.

**The Fix:**
1.  **Normalization**: I updated `_normalize_ideas` to explicitly check for `competitor_image` and default to "NaN" if missing, so the key always exists.
2.  **Slide Generation**: I added a safety check. If `competitor_image` is missing or "NaN", it will try to fallback to `proposal_image`. if both are missing, it safely passes `None` to the image placer, which will render a grey "Image Not Available" placeholder instead of crashing.

Please try generating the PPT again. It should work without error now.
