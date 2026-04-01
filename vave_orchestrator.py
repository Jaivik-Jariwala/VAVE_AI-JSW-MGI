"""
VAVE AI Multi-Agent Orchestrator — Features 11, 13, 14
Implements:
  Feature 11 — Multi-Agent System: Orchestrator + specialized subagents
  Feature 13 — Visual Chain-of-Thought for BLIP VLM
  Feature 14 — Agentic RAG (ReAct loop within the Orchestrator)

Architecture:
  VAVEOrchestrator
    ├── RetrievalAgent   (hybrid_retrieval, metadata filter, multi-query)
    ├── AnalysisAgent    (SQL aggregations, cost calculations)
    ├── ReportAgent      (delegates to existing generate_excel / generate_ppt)
    └── VisionAgent      (BLIP Visual CoT, image analysis)

The Orchestrator uses a lightweight ReAct-style loop:
  Plan → Execute tools → Observe → Synthesize

Each subagent is a thin wrapper that uses the shared Gemini client but
operates on a focused, bounded task.
"""
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import google.generativeai as genai

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════
# FEATURE 13 — Vision Agent (Visual Chain-of-Thought for BLIP)
# ════════════════════════════════════════════════════════════════════════

class VisionAgent:
    """
    Feature 13: Wraps BLIP VLM with step-by-step Visual Chain-of-Thought.
    Replaces direct single-step captioning with a 4-step reasoning chain
    to dramatically reduce hallucination.
    """

    def __init__(self, vlm_model, vlm_processor, device: str = "cpu"):
        self.vlm_model = vlm_model
        self.vlm_processor = vlm_processor
        self.device = device

    def _blip_caption(self, image, question: str) -> str:
        """Single BLIP inference step."""
        try:
            from PIL import Image as PILImage
            import torch

            if not isinstance(image, PILImage.Image):
                return ""

            inputs = self.vlm_processor(image, question, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.vlm_model.generate(**inputs, max_new_tokens=150)
            return self.vlm_processor.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"[VisionAgent] BLIP inference error: {e}")
            return ""

    def visual_chain_of_thought(
        self,
        image,
        vehicle_context: str = "",
        component_context: str = "",
    ) -> Dict[str, Any]:
        """
        Feature 13: 4-step Visual CoT over a VAVE part image.

        Steps:
          1. Describe: What component is visible?
          2. Identify: Material, manufacturing process?
          3. VAVE: What modifications are possible?
          4. Impact: Weight/cost estimate?

        Returns dict with each step + final summary.
        """
        import prompt_registry

        steps = [
            ("describe", prompt_registry.get("visual_cot_step1_describe")),
        ]

        reasoning_chain = {}
        context_so_far = vehicle_context

        # Step 1 — Describe
        step1_q = prompt_registry.get("visual_cot_step1_describe")
        step1_ans = self._blip_caption(image, step1_q)
        reasoning_chain["step1_describe"] = step1_ans
        context_so_far += f" Component: {step1_ans}"

        # Step 2 — Identify material/process
        step2_q = prompt_registry.get(
            "visual_cot_step2_identify", prev_context=context_so_far[:300]
        )
        step2_ans = self._blip_caption(image, step2_q)
        reasoning_chain["step2_identify"] = step2_ans
        context_so_far += f" Material/Process: {step2_ans}"

        # Step 3 — VAVE opportunities
        step3_q = prompt_registry.get(
            "visual_cot_step3_vave", prev_context=context_so_far[:400]
        )
        step3_ans = self._blip_caption(image, step3_q)
        reasoning_chain["step3_vave_opportunities"] = step3_ans
        context_so_far += f" VAVE: {step3_ans}"

        # Step 4 — Impact estimate
        step4_q = prompt_registry.get(
            "visual_cot_step4_impact",
            vehicle_context=vehicle_context,
            prev_context=context_so_far[:500],
        )
        step4_ans = self._blip_caption(image, step4_q)
        reasoning_chain["step4_impact_estimate"] = step4_ans

        reasoning_chain["final_summary"] = (
            f"{step1_ans} | {step2_ans} | VAVE: {step3_ans} | Impact: {step4_ans}"
        )
        reasoning_chain["hallucination_mode"] = "chain_of_thought"

        logger.info(f"[VisionAgent] VCoT complete: {reasoning_chain['final_summary'][:80]}")
        return reasoning_chain

    def analyze_comparison(
        self,
        mg_image,
        competitor_image,
        component_label: str = "",
    ) -> Dict[str, Any]:
        """
        Analyse MG vs competitor images using Visual CoT on both,
        then compare the two chains.
        """
        mg_chain = self.visual_chain_of_thought(mg_image, component_context=component_label)
        comp_chain = self.visual_chain_of_thought(competitor_image, component_context=component_label)
        return {
            "mg_analysis": mg_chain,
            "competitor_analysis": comp_chain,
            "mg_summary": mg_chain.get("final_summary", ""),
            "competitor_summary": comp_chain.get("final_summary", ""),
        }


# ════════════════════════════════════════════════════════════════════════
# SUB-AGENT: RetrievalAgent
# ════════════════════════════════════════════════════════════════════════

class RetrievalAgent:
    """
    Feature 11: Dedicated retrieval subagent.
    Wraps hybrid_retrieval with metadata filtering + multi-query expansion.
    """

    def __init__(self, faiss_index, embedding_model, idea_texts, idea_rows, pg_conn_func):
        self.faiss_index = faiss_index
        self.embedding_model = embedding_model
        self.idea_texts = idea_texts
        self.idea_rows = idea_rows
        self.pg_conn_func = pg_conn_func

        # Import shared bm25_index singleton
        import hybrid_retrieval as hr
        self.bm25 = hr.bm25_index
        self.hr = hr

    def run(
        self,
        query: str,
        gemini_flash_model=None,
        top_k: int = 10,
        use_multi_query: bool = True,
        use_metadata_filter: bool = True,
        use_compression: bool = True,
    ) -> Tuple[List[Dict], str]:
        """
        Full retrieval pipeline:
        1. Extract metadata filters from query
        2. Get candidate IDs from PostgreSQL
        3. Run multi-query hybrid search or single hybrid search
        4. Optionally compress context
        Returns (ideas_list, compressed_context_string)
        """
        # Step 1: Metadata filter
        candidate_ids = None
        if use_metadata_filter and self.pg_conn_func:
            filters = self.hr.extract_metadata_filters(query)
            if filters:
                candidate_ids = self.hr.filter_candidate_ids(self.pg_conn_func, **filters)

        # Step 2: Search
        if use_multi_query and gemini_flash_model:
            results = self.hr.multi_query_hybrid_search(
                query=query,
                gemini_model=gemini_flash_model,
                faiss_index=self.faiss_index,
                embedding_model=self.embedding_model,
                idea_texts=self.idea_texts,
                idea_rows=self.idea_rows,
                bm25_index=self.bm25,
                top_k=top_k,
                candidate_ids=candidate_ids,
            )
        else:
            results = self.hr.hybrid_search(
                query=query,
                faiss_index=self.faiss_index,
                embedding_model=self.embedding_model,
                idea_texts=self.idea_texts,
                idea_rows=self.idea_rows,
                bm25_index=self.bm25,
                top_k=top_k,
                candidate_ids=candidate_ids,
            )

        # Step 3: Contextual compression
        compressed_ctx = ""
        if use_compression and gemini_flash_model and results:
            compressed_ctx = self.hr.compress_retrieved_context(
                retrieved_ideas=results,
                user_query=query,
                gemini_flash_model=gemini_flash_model,
            )
        else:
            # Build compact context without compression
            compressed_ctx = "\n".join([
                f"[{r.get('idea_id','N/A')}] {r.get('cost_reduction_idea','')} "
                f"| ₹{r.get('saving_value_inr','N/A')} | {r.get('dept','N/A')} | {r.get('status','N/A')}"
                for r in results
            ])

        return results, compressed_ctx


# ════════════════════════════════════════════════════════════════════════
# SUB-AGENT: AnalysisAgent
# ════════════════════════════════════════════════════════════════════════

class AnalysisAgent:
    """
    Feature 11: Dedicated analysis subagent.
    Runs SQL aggregations and cost calculations against PostgreSQL.
    """

    def __init__(self, pg_conn_func: Callable):
        self.pg_conn_func = pg_conn_func

    def run(self, analysis_type: str, ideas: List[Dict], query: str = "") -> Dict[str, Any]:
        """
        Run structured analysis on the retrieved ideas.
        analysis_type: 'aggregate_savings' | 'rank_ideas' | 'compare_depts' | 'sql'
        """
        if analysis_type == "aggregate_savings":
            return self._aggregate_savings(ideas)
        elif analysis_type == "rank_ideas":
            return self._rank_by_saving(ideas)
        elif analysis_type == "compare_depts":
            return self._compare_by_dept(ideas)
        elif analysis_type == "sql":
            return self._run_sql_analysis(query)
        return {"result": ideas}

    def _aggregate_savings(self, ideas: List[Dict]) -> Dict:
        total = sum(float(r.get("saving_value_inr") or 0) for r in ideas)
        total_weight = sum(float(r.get("weight_saving") or 0) for r in ideas)
        return {
            "total_ideas": len(ideas),
            "total_saving_inr": round(total, 2),
            "total_weight_saving_kg": round(total_weight, 2),
        }

    def _rank_by_saving(self, ideas: List[Dict]) -> Dict:
        ranked = sorted(ideas, key=lambda x: float(x.get("saving_value_inr") or 0), reverse=True)
        return {"ranked_ideas": ranked}

    def _compare_by_dept(self, ideas: List[Dict]) -> Dict:
        dept_totals: Dict[str, float] = {}
        for r in ideas:
            dept = r.get("dept", "Unknown")
            dept_totals[dept] = dept_totals.get(dept, 0) + float(r.get("saving_value_inr") or 0)
        return {"dept_savings": dict(sorted(dept_totals.items(), key=lambda x: x[1], reverse=True))}

    def _run_sql_analysis(self, sql: str) -> Dict:
        """Execute a safe read-only SQL query against PostgreSQL."""
        sql_lower = sql.lower().strip()
        if not sql_lower.startswith("select"):
            return {"error": "Only SELECT queries permitted"}
        try:
            conn = self.pg_conn_func()
            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description] if cur.description else []
            cur.close()
            conn.close()
            return {"columns": cols, "rows": [list(r) for r in rows]}
        except Exception as e:
            logger.error(f"[AnalysisAgent] SQL error: {e}")
            return {"error": str(e)}


# ════════════════════════════════════════════════════════════════════════
# FEATURE 14 — ReAct Orchestrator (Main Multi-Agent Loop)
# ════════════════════════════════════════════════════════════════════════

class VAVEOrchestrator:
    """
    Feature 11 + 14: Multi-Agent Orchestrator with ReAct planning.

    Replaces the monolithic VAVEAgent.run() with a structured pipeline:
      1. Plan: Understand query and decide which agents to activate
      2. Execute: Run activated subagents in order
      3. Synthesize: Merge all agent outputs into a final response

    Falls back to the original VAVEAgent for complex creative/web tasks.
    """

    def __init__(
        self,
        faiss_index,
        embedding_model,
        idea_texts: List[str],
        idea_rows: List[Dict],
        pg_conn_func: Callable,
        gemini_pro_model,
        gemini_flash_model,
        vave_agent=None,             # Original VAVEAgent for fallback
        vlm_model=None,
        vlm_processor=None,
        device: str = "cpu",
    ):
        self.gemini_pro = gemini_pro_model
        self.gemini_flash = gemini_flash_model
        self.vave_agent = vave_agent

        # Initialize subagents
        self.retrieval = RetrievalAgent(
            faiss_index, embedding_model, idea_texts, idea_rows, pg_conn_func
        )
        self.analysis = AnalysisAgent(pg_conn_func)
        self.pg_conn_func = pg_conn_func

        # Vision agent (optional — needs BLIP)
        self.vision: Optional[VisionAgent] = None
        if vlm_model and vlm_processor:
            self.vision = VisionAgent(vlm_model, vlm_processor, device)
            logger.info("[VAVEOrchestrator] VisionAgent initialized with Visual CoT.")

        logger.info("[VAVEOrchestrator] Initialized with all subagents.")

    # ── Feature 14: ReAct Planner ─────────────────────────────────────

    def _plan(self, user_query: str) -> Dict:
        """
        Use Gemini Flash to produce a lightweight execution plan.
        Falls back to a default plan if LLM fails.
        """
        import prompt_registry
        try:
            plan_prompt = prompt_registry.get("react_planner", query=user_query)
            raw = self.gemini_flash.generate_content(plan_prompt).text
            match = re.search(r'\{.*?\}', raw, re.DOTALL)
            if match:
                plan = json.loads(match.group(0))
                logger.info(f"[VAVEOrchestrator] ReAct plan: {json.dumps(plan)[:200]}")
                return plan
        except Exception as e:
            logger.warning(f"[VAVEOrchestrator] Planning failed, using default plan: {e}")

        # Default plan: always retrieve + analyse
        return {
            "thought": "Default: retrieve ideas and aggregate analysis",
            "steps": [
                {"tool": "vector_search", "input": user_query, "reason": "Find relevant ideas"},
                {"tool": "calculate", "input": "aggregate_savings", "reason": "Compute totals"},
            ],
            "needs_retrieval": True,
            "needs_calculation": True,
            "needs_report": False,
        }

    # ── Feature 14: Synthesizer ───────────────────────────────────────

    def _synthesize(self, user_query: str, tool_results: Dict) -> str:
        """
        Use Gemini Pro to synthesize a final answer from all tool results.
        """
        import prompt_registry
        try:
            synth_prompt = prompt_registry.get(
                "react_synthesizer",
                query=user_query,
                tool_results=json.dumps(tool_results, default=str, indent=2)[:4000],
            )
            response = self.gemini_pro.generate_content(synth_prompt)
            return response.text
        except Exception as e:
            logger.error(f"[VAVEOrchestrator] Synthesis error: {e}")
            # Fallback: return raw tool results summary
            return f"Retrieved {len(tool_results.get('ideas', []))} ideas. Analysis: {tool_results.get('analysis', {})}"

    # ── Main Entry Point ──────────────────────────────────────────────

    def process(self, user_query: str) -> dict:
        """
        Feature 11+14: Main ReAct orchestration loop.
        Returns the synthesized final response string and raw table data.
        """
        logger.info(f"[VAVEOrchestrator] Processing: {user_query[:80]}")
        tool_results: Dict[str, Any] = {}

        # Step 1: Plan
        plan = self._plan(user_query)

        # Step 2: Execute retrieval
        if plan.get("needs_retrieval", True):
            try:
                ideas, context = self.retrieval.run(
                    query=user_query,
                    gemini_flash_model=self.gemini_flash,
                    top_k=10,
                    use_multi_query=True,
                    use_metadata_filter=True,
                    use_compression=True,
                )
                tool_results["ideas"] = ideas
                tool_results["compressed_context"] = context
                logger.info(f"[VAVEOrchestrator] Retrieved {len(ideas)} ideas via hybrid search.")
            except Exception as e:
                logger.error(f"[VAVEOrchestrator] Retrieval failed: {e}")
                tool_results["ideas"] = []

        # Step 3: Analysis
        if plan.get("needs_calculation", False) and tool_results.get("ideas"):
            try:
                tool_results["analysis"] = self.analysis.run(
                    analysis_type="aggregate_savings",
                    ideas=tool_results["ideas"],
                )
                tool_results["ranked"] = self.analysis.run(
                    analysis_type="rank_ideas",
                    ideas=tool_results["ideas"],
                )
            except Exception as e:
                logger.error(f"[VAVEOrchestrator] Analysis failed: {e}")

        # Step 4: SQL-based analysis if plan specifies
        for step in plan.get("steps", []):
            if step.get("tool") == "sql_query":
                try:
                    sql_result = self.analysis.run(
                        analysis_type="sql",
                        ideas=[],
                        query=step.get("input", ""),
                    )
                    tool_results["sql_result"] = sql_result
                except Exception as e:
                    logger.error(f"[VAVEOrchestrator] SQL step failed: {e}")

        # Step 5: Synthesize final answer
        response_text = self._synthesize(user_query, tool_results)
        return {
            "response_text": response_text,
            "table_data": tool_results.get("ideas", [])
        }

    def analyze_image_with_cot(
        self,
        image,
        vehicle_context: str = "",
        component_label: str = "",
    ) -> Optional[Dict]:
        """Feature 13: Wrapper for Visual CoT image analysis."""
        if self.vision:
            return self.vision.visual_chain_of_thought(
                image=image,
                vehicle_context=vehicle_context,
                component_context=component_label,
            )
        logger.warning("[VAVEOrchestrator] VisionAgent not initialized; BLIP loaded?")
        return None
