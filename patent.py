from fpdf import FPDF
import textwrap

# 1. Define the Content
# Note: Unicode characters have been replaced with ASCII equivalents for PDF compatibility.

patent_content = """
MASTER PATENT SPECIFICATION
Status: Ready for Filing (Provisional/Complete)
Jurisdiction: India (IPO) - Aligned with 2025 CRI Guidelines
Date: February 6, 2026

----------------------------------------------------------------------

FORM 2
THE PATENTS ACT, 1970
(39 OF 1970)
&
THE PATENTS RULES, 2003

1. TITLE OF THE INVENTION
SYSTEM AND METHOD FOR DETERMINISTIC PHYSICS-CONSTRAINED GENERATIVE VALUE ENGINEERING AND VALIDATION (VAVE AI)

2. APPLICANT(S)
Name: Jaivik Rajkumar Jariwala
Nationality: Indian
Status: Student Intern / AI Engineer
Affiliation: JSW MG Motor India / Pandit Deendayal Energy University (PDEU)

3. PREAMBLE TO THE DESCRIPTION
The following specification particularly describes the invention and the manner in which it is to be performed.

4. FIELD OF THE INVENTION
The present invention relates generally to the field of Artificial Intelligence in Automotive Engineering and Computer-Aided Engineering (CAE). More specifically, it relates to a Neuro-Symbolic Architecture that integrates Probabilistic Generative Models (Large Language Models) with Deterministic Physics Engines to automate Value Analysis/Value Engineering (VAVE) processes while strictly eliminating hallucination risks in safety-critical component validation.

5. BACKGROUND AND PRIOR ART
The paradigm of intellectual property protection for digital innovations has evolved significantly as of 2026. Under the 2025 Guidelines for the Examination of Computer-Related Inventions (CRI) in India, software must demonstrate a "further technical effect" to overcome Section 3(k) exclusions.

Current Limitations in the State of the Art:
1. Probabilistic Nature of LLMs: Conventional Generative AI systems (e.g., GPT-4, Gemini) operate on Next-Token Prediction. In engineering contexts, this leads to "hallucinations," where models invent non-existent material properties or violate thermodynamic laws.
2. Lack of Ground Truth: Generic models treat verified "Teardown Reports" and unverified "Forum Posts" with equal weight, leading to unreliable engineering insights.
3. Visual Fabrication: Standard Vision-Language Models (VLMs) generate new pixels when asked to modify a component (e.g., Stable Diffusion), often altering the structural geometry of a part (e.g., mounting points) in physically impossible ways.

The Technical Problem:
There is an urgent need for a system that leverages the semantic reasoning of LLMs for ideation but constrains their output through rigid, non-neural Physics Manifolds, ensuring that every generated Engineering Change Request (ECR) is mathematically valid before human review.

6. OBJECTS OF THE INVENTION
* Primary Object: To provide a system that enforces deterministic physics constraints on generative AI outputs.
* Secondary Object: To eliminate visual hallucinations by using coordinate inference and vector overlays instead of pixel generation.
* Tertiary Object: To provide a quantifiable "Technical Effect" through resource optimization (reduced validation time) and system-level reliability (safety compliance).

7. DETAILED DESCRIPTION OF THE INVENTION

7.1 Glossary of Technical Terms
* Probabilistic Generative AI: AI models that function by guessing the most likely next token based on statistical patterns (e.g., standard LLMs).
* Deterministic Engineering AI: A system where the output is governed by fixed, non-negotiable rules. If input is A and Rule is R, output must be B.
* Neuro-Symbolic Architecture: A hybrid design combining Neural Networks (for semantic understanding) and Symbolic Logic (for rule enforcement).
* Constraint Manifold (K): A mathematical representation of the "safe space" for a solution, defined by boundary conditions (e.g., stress limits, melting points).
* The "Gatekeeper": A non-neural physics engine that filters neural outputs against rigid mathematical formulas.

7.2 System Architecture (The Apparatus)
The invention is an end-to-end apparatus comprising four interconnected logic modules:

A. The Context Resolution & Constraint Injection Module
This module intercepts natural language queries (e.g., "Optimize the front brake caliper for weight") and maps them to a rigid Constraint Manifold (K). Unlike standard prompt engineering, this module retrieves specific boundary conditions from a structured SQL database and "injects" them as non-negotiable rules into the context window.

B. The Multi-Source Fusion Engine
The system fuses three distinct data streams with dynamic reliability weighting to generate engineering proposals:
1. Stream A (Rigid): XEPC Teardown Data (Excel/Structured). Reliability alpha=0.9.
2. Stream B (Internal): RAG Database of past Engineering Change Notes. Reliability beta=0.7.
3. Stream C (External): Semantic Web Mining for raw material costs. Reliability gamma=0.4.

C. The Deterministic Validation Gatekeeper
This is the core novelty. It is a Non-Neural Physics Engine. It receives the AI's proposed solution and runs it against standard engineering formulas (Stress, Cost, Weight). It acts as a logical interlock, preventing the AI from outputting dangerous suggestions.

D. The Visual Inference & Overlay Module
Instead of generating new images, this module uses Object Detection to identify the Region of Interest (ROI) coordinates (x, y, w, h). It then renders a vector-based "Delta Card" (a transparent UI layer) containing the proposed changes, superimposed over the original, unaltered engineering photograph.

7.3 Mathematical Modeling (The Inventive Steps)
The invention transforms the subjective VAVE process into an objective mathematical optimization problem.

I. The Constraint Manifold (K)
We define the engineering valid space K for a component C as:
K(C) = { x in R^n | g_i(x) <= 0, h_j(x) = 0 }

Where:
* g_i(x) represents inequality constraints (e.g., Weight < 5kg).
* h_j(x) represents equality constraints (e.g., Material == "Aluminum 6061").
The AI is mathematically restricted to search only within K(C).

II. The "Gatekeeper" Scoring Vector (S)
For every generated proposal (P), the system calculates a 4-dimensional feasibility vector S:
S(P) = [s_feaz, s_cost, s_wght, s_regs]

Where:
* s_feaz: Manufacturing Feasibility Score (0-1).
* s_cost: Cost Delta vs. Baseline.
* s_wght: Weight Delta vs. Baseline.
* s_regs: Regulatory Compliance Boolean (1 or 0).

III. The Deterministic Decision Function (D)
The final output is governed by a discrete step function, ensuring no "hallucinated" approvals:

D(P) = Auto-Approve      if min(S) >= tau_safe AND s_regs = 1
       Manual-Review     if tau_min <= min(S) < tau_safe
       Reject            otherwise

* tau_safe: The safety threshold (e.g., 0.95 confidence).

IV. Visual Coordinate Transformation (T)
To prevent visual hallucination, we claim the transformation T that maps semantic text to pixel coordinates without pixel generation:
T(Img_input, Text_query) -> (x_c, y_c, w, h)_ROI
View_final = Img_input (+) Render(Delta_data at (x_c, y_c))

8. STRATEGIC IMPLEMENTATION AND BEST MODE
Preferred Embodiment:
* Backend: Python (FastAPI) handling the logic flow.
* Database: Vector Database (Pinecone) for RAG embeddings + SQL for Constraint Manifolds.
* LLM Core: Gemini 1.5 Pro (via API) for semantic reasoning.
* Gatekeeper Engine: Custom Python module using SciPy for physics calculations.

Example Workflow:
1. Input: User uploads a photo of a "Steering Knuckle" and asks: "Reduce cost by changing material."
2. AI Proposal: AI suggests "Cast Iron" (Cheaper, but heavier).
3. Gatekeeper Calculation: System calculates Density of Iron vs. Aluminum. Result: Weight increases by 40%.
4. Vector Check: s_wght score drops below threshold tau_min.
5. Output: System auto-rejects: "Rejected: Cost target met, but violates Weight constraint by 40%."

9. CLAIMS
We Claim:

1. A System for Automated Engineering Value Analysis comprising:
(a) A Data Ingestion Module configured to parse structured teardown data and unstructured semantic data;
(b) A Context Resolution Unit capable of mapping natural language queries to a predefined physics constraint manifold;
(c) A Generative Fusion Engine for proposing engineering changes; and
(d) A Deterministic Validation Gatekeeper configured to filter said proposals using non-neural physics algorithms.

2. The Method of "Constraint Injection": wherein the Context Resolution Unit autonomously retrieves physical boundary conditions (e.g., melting point, tensile strength) from a rigid database and injects them into the generative model's context window prior to inference.

3. The Validation Algorithm: A method for validating AI-generated engineering designs utilizing a multi-dimensional scoring vector S = [s_feaz, s_cost, s_wght, s_regs], wherein the final approval is governed by a deterministic step function D(P) independent of the neural network's confidence score.

4. The Visual Inference Method: A method for displaying engineering changes characterized by identifying Region of Interest (ROI) coordinates (x,y) on an original 2D image and superimposing a vector-based information layer ("Delta Card"), thereby preventing pixel-level hallucination of the original component.

5. The Multi-Source Weighting Logic: A system according to Claim 1, wherein the fusion engine applies dynamic reliability coefficients (alpha, beta, gamma) to input data, prioritizing structured Teardown Data over web-scraped data during conflict resolution.

10. ABSTRACT
The present invention discloses "VAVE AI," a Neuro-Symbolic system for automated Value Analysis and Value Engineering. The system addresses the "hallucination" problem in Generative AI by introducing a Deterministic Validation Gatekeeper. Unlike standard Large Language Models (LLMs) that predict responses probabilistically, the VAVE AI system intercepts user queries to inject rigid physics constraints and filters all AI-generated outputs through a mathematical scoring vector (S) comprising feasibility, cost, weight, and regulatory parameters. The invention further includes a Visual Inference Module that projects engineering data onto original component images using coordinate overlays, eliminating visual fabrication. This provides a "Glass Box" engineering tool suitable for safety-critical automotive applications.
"""

presentation_content = """
SECTION 2: POWERPOINT PRESENTATION STRATEGY

Theme: Professional, Minimalist, Engineering-Focused (Blue/White/Grey theme).
Total Slides: 10-12
Target Audience: JSW MG Motor IP Committee & R&D Management.

Slide 1: Title
* Header: VAVE AI: Deterministic Neuro-Symbolic System for Automotive Value Engineering
* Sub-header: Transitioning from Probabilistic "Chatbots" to Physics-Constrained Engineering Validation
* Footer: Jaivik Rajkumar Jariwala, AI Engineer | Date: Feb 2026

Slide 2: The Problem (The "Trust Gap")
* Visual: A split screen.
    * Left: Standard LLM (ChatGPT) suggesting a "Plastic Brake Disc" (Hallucination).
    * Right: A text box saying "Safety Critical Failure."
* Bullet Points:
    * LLMs are Probabilistic (Next Token Prediction).
    * Automotive Engineering requires Determinism (Physics Laws).
    * Current AI lacks "Ground Truth" awareness.

Slide 3: The Solution - High Level
* Visual: Block Diagram (Inputs -> The VAVE Box -> Validated Output).
* Key Concept: "The Glass Box Approach."
* Text: We replace "Prompt Engineering" with "Constraint Injection" and "Physics Validation."

Slide 4: System Architecture (The "Machine")
* Visual: * Flow:
    1. Ingestion: XEPC Data (Excel) + Semantic Web.
    2. The Matrix: Context Resolution (Mapping text to constraints).
    3. The Gatekeeper: The Physics Engine filter.

Slide 5: The Core Innovation - The "Gatekeeper"
* Visual: The Equation (Rendered Large).
    D(P) = Approve if S > Safe; Reject otherwise
* Narration: "The AI proposes, but the Math disposes. The Neural Network does not have the authority to approve a design. Only the Physics Engine does."

Slide 6: Visual Inference (Anti-Hallucination)
* Visual: Comparison.
    * Traditional GenAI: Distorts the car part image to fit the text.
    * VAVE AI: Keeps original photo, overlays a sleek data card (Delta Card) on top.
* Tech: Coordinate Inference (x, y, w, h).

Slide 7: Comparative Advantage (The Moat)
* Table:
    * Generic LLM: Probabilistic, Hallucinates, Unsafe.
    * VAVE AI: Deterministic, Grounded, Safety-Critical.

Slide 8: Development Roadmap & Status
* Nov 2025 - Jan 2026: Research & Architecture (Complete).
* Feb 2026: Patent Filing (Provisional).
* Mar 2026: Prototype Deployment at JSW MG (V1.0).
* Aug 2026: Integration with Global Supply Chain Data.

Slide 9: Patent Strategy & Cost
* Filing Type: Utility Patent (System + Method).
* Applicant: Student Intern (Small Entity) OR Corporate Assignment.
* Cost: < 10,000 INR (Govt Fees for Individual).
* Value: Protects the "Neuro-Symbolic" Logic.

Slide 10: Conclusion
* Call to Action: Approval to file Provisional Application.
* Impact: Reduces VAVE time from Weeks to Minutes with 100% Safety Compliance.
"""

cost_content = """
SECTION 3: COST, TIMELINE & POSSIBILITIES

1. Financial Lifecycle (2026 Estimates - India)
Since you are a student/intern, you have two routes.

Route A: Filing as "Natural Person" (You are the Applicant)
* Benefit: Massive 80% fee reduction by the Indian Patent Office (IPO).
* Cost Breakdown:
    * Filing Fee (Form 1): 1,600 INR
    * Examination Request (Form 18): 4,000 INR
    * Total Govt Fee: 5,600 INR (approx).
    * Attorney Fees: 15k - 30k INR (if you hire one to review this draft).

Route B: Filing as "Large Entity" (JSW MG Motor is the Applicant)
* Benefit: Company pays all costs; you get "Inventor" credit on the patent certificate. Stronger enforcement resources.
* Cost Breakdown:
    * Filing Fee: 8,000 INR
    * Examination Request: 20,000 INR
    * Total Govt Fee: 28,000 INR.

2. Timeline (The "Fast Track")
* Feb 10, 2026: File Provisional Application (Secure the Priority Date immediately).
* Feb - Aug 2026: Build the working prototype at JSW MG. Collect real data logs (Evidence of "Technical Effect").
* Jan 2027: File Complete Specification (Within 12 months).
* Feb 2027: File Form 18A (Expedited Examination).
    * Possibility: As a Start-up/Student/Female inventor (if applicable to team), you can get a grant in <1 year.

3. Possibilities & Strategic Advice
* NotebookLLM Usage: Upload the "Master Patent Specification" text provided above into Google NotebookLLM. It will generate summaries, FAQs, and even an Audio Overview that you can play during your presentation to impress the stakeholders.
* The "Section 3(k)" Defense: Your strongest asset is the "Gatekeeper". By emphasizing that the software drives a physical validation process using physics equations, you move out of "Abstract Algorithms" and into "Technical Applications," which is the key to getting a software patent granted in India in 2026.
"""

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 10)
        self.cell(0, 10, 'VAVE AI Patent Specification & Strategy', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Times', '', 12)
        self.multi_cell(0, 7, body)
        self.ln()

# Create PDF
pdf = PDF()
pdf.alias_nb_pages()
pdf.add_page()

# Write content
pdf.chapter_title("SECTION 1: MASTER PATENT SPECIFICATION")
pdf.chapter_body(patent_content)
pdf.add_page()
pdf.chapter_title("SECTION 2: PRESENTATION STRATEGY")
pdf.chapter_body(presentation_content)
pdf.add_page()
pdf.chapter_title("SECTION 3: COST & TIMELINE")
pdf.chapter_body(cost_content)

# Output
pdf.output("VAVE_AI_Patent_Compilation.pdf")