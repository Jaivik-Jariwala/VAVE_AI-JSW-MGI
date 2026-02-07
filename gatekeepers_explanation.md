# The Two Gatekeepers of VAVE AI

Here is the explanation of the two critical "Gatekeepers" that ensure VAVE AI generates engineering-grade ideas.

### 1. Physics-Informed Matrix (Context Resolver)
*   **What it does:** It acts as the "Common Sense" filter before idea generation even starts.
*   **How it works:** It uses a **Vehicle × Component Matrix** to enforce fundamental engineering constraints.
*   **Example:** If the vehicle is an EV (MGMT EV), the Matrix *automatically blocks* any ideas related to "Exhaust Systems" or "Fuel Tanks" because they physically cannot exist on that platform. It prevents the AI from Hallucinating impossible components.

### 2. Autonomous Validation Engine (The Ruler)
*   **What it does:** It acts as the "inspector" *after* an idea is generated but before it reaches you.
*   **How it works:** It applies deterministic, hard-coded physics rules to validate feasibility.
*   **Example:** If an idea proposes "Thinning the B-Pillar by 20%" to save cost, this gatekeeper checks the vehicle's Curb Weight. If `Weight > 1600kg`, it knows thinning is dangerous for crash safety, so it **Stamps "REJECTED"** immediately. Only safe, physically valid ideas pass through.
