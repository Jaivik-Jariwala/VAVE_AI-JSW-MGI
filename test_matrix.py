from agent import VAVEAgent


def _dummy_vector_db_func(query: str, top_k: int = 5):
    return [], None


def run_case(agent: VAVEAgent, query: str, target_component: str, idea_text: str):
    # Hand-crafted idea with strong scores so matrix heuristics can override if needed
    ideas = [
        {
            "idea_id": "TEST-01",
            "cost_reduction_idea": idea_text,
            "way_forward": "DFMEA + DVPR + validation plan.",
            "homologation_theory": "Assess AIS/CMVR impact and re-validate as needed.",
            "feasibility_score": 80,
            "cost_saving_score": 70,
            "weight_reduction_score": 60,
            "homologation_feasibility_score": 75,
        }
    ]

    validated = agent._validate_and_filter_ideas(  # noqa: SLF001 (test harness)
        ideas,
        target_component=target_component,
        query=query,
        min_score=25,
    )

    if not validated:
        print(f"\nQUERY: {query}\n-> No ideas returned (rejected).")
        return

    idea = validated[0]
    print(f"\nQUERY: {query}")
    print(f"STATUS: {idea.get('validation_status')}")
    print(f"NOTES:  {idea.get('validation_notes')}")


if __name__ == "__main__":
    agent = VAVEAgent(db_path=":memory:", vector_db_func=_dummy_vector_db_func)

    # Case A: Hector brake disc weight reduction should be warned/reviewed due to heavy vehicle thermal risk
    run_case(
        agent,
        query="Reduce weight of MG Hector Brake Disc",
        target_component="brake disc",
        idea_text="Reduce brake disc thickness to reduce weight and cost.",
    )

    # Case B: Comet brake disc weight reduction should be acceptable (much lighter vehicle)
    run_case(
        agent,
        query="Reduce weight of MG Comet Brake Disc",
        target_component="brake disc",
        idea_text="Reduce brake disc thickness to reduce weight and cost.",
    )

    # Case C: ZS EV blower power increase should be flagged for range impact
    run_case(
        agent,
        query="Increase Blower Power for MG ZS EV",
        target_component="blower motor",
        idea_text="Increase blower motor power (higher watt) to improve airflow.",
    )

