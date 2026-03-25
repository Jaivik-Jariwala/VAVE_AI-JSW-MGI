import re

with open("app.py", "r", encoding="utf-8") as f:
    text = f.read()

cte = """
        cur.execute('''
            CREATE TEMP VIEW all_ideas AS
            WITH chat_raw AS (
                SELECT jsonb_array_elements(
                     CASE WHEN jsonb_typeof(table_data) = 'array' THEN table_data ELSE '[]'::jsonb END
                ) as json_line
                FROM chat_history
                WHERE table_data IS NOT NULL AND table_data != 'null'::jsonb
            ),
            chat_ideas AS (
                SELECT 
                    COALESCE(NULLIF(json_line->>'Idea ID', ''), NULLIF(json_line->>'Idea Id', ''), 'AI-' || left(md5(random()::text), 6)) as idea_id,
                    COALESCE(NULLIF(json_line->>'Status', ''), 'TBD') as status,
                    COALESCE(NULLIF(json_line->>'Dept', ''), NULLIF(json_line->>'Department', ''), '?') as dept,
                    NULLIF(REGEXP_REPLACE(json_line->>'Saving (INR)', '[^\\d.]', '', 'g'), '')::numeric as saving_value_inr,
                    NULLIF(REGEXP_REPLACE(json_line->>'Weight Saving', '[^\\d.]', '', 'g'), '')::numeric as weight_saving,
                    NULLIF(REGEXP_REPLACE(json_line->>'Cost Saving', '[^\\d.]', '', 'g'), '')::numeric as capex,
                    COALESCE(json_line->>'Detailed Idea Description', json_line->>'Idea Description', json_line->>'Title', '') as cost_reduction_idea,
                    COALESCE(NULLIF(json_line->>'Origin', ''), NULLIF(json_line->>'Idea Source', ''), 
                             CASE WHEN json_line->>'Status' ILIKE '%AI Generated%' THEN 'AI Generated'
                                  WHEN json_line->>'Status' ILIKE '%Web Sourced%' THEN 'Web Sourced'
                                  ELSE 'AI Generated' END) as idea_generated_by,
                    COALESCE(NULLIF(json_line->>'MGI PE Feasibility', ''), NULLIF(json_line->>'PE Feasibility', ''), NULLIF(json_line->>'Feasibility', ''), 'Unknown') as mgi_pe_feasibility,
                    COALESCE(NULLIF(json_line->>'Financial Feasibility', ''), NULLIF(json_line->>'Homologation Feasibility', ''), 'Unknown') as financial_feasibility,
                    json_line->>'Way Forward' as way_forward,
                    COALESCE(NULLIF(json_line->>'Component Group', ''), NULLIF(json_line->>'Component', ''), 'Other') as group_id
                FROM chat_raw
                WHERE json_line->>'Idea ID' ILIKE 'AI-%' OR json_line->>'Idea ID' ILIKE 'WEB-%'
                   OR json_line->>'Origin' ILIKE '%AI%' OR json_line->>'Idea Source' ILIKE '%AI%'
                   OR json_line->>'Origin' ILIKE '%Web%' OR json_line->>'Idea Source' ILIKE '%Web%'
                   OR json_line->>'Status' ILIKE '%AI Generated%' OR json_line->>'Status' ILIKE '%Web Sourced%'
            )
            SELECT idea_id, status, dept, saving_value_inr, weight_saving, capex, cost_reduction_idea, idea_generated_by, mgi_pe_feasibility, financial_feasibility, group_id, way_forward
            FROM ideas
            UNION ALL
            SELECT idea_id, status, dept, saving_value_inr, weight_saving, capex, cost_reduction_idea, idea_generated_by, mgi_pe_feasibility, financial_feasibility, group_id, way_forward
            FROM chat_ideas;
        ''')
"""

lines = text.splitlines()

# Endpoint 1: analytics_ideas_detail
for i in range(3962, 4120):
    if lines[i].strip() == "FROM ideas":
        lines[i] = lines[i].replace("FROM ideas", "FROM all_ideas")

# Endpoint 2: analytics_ideas_table
for i in range(4126, 4200):
    if lines[i].strip() == "FROM ideas":
        lines[i] = lines[i].replace("FROM ideas", "FROM all_ideas")

new_text = "\\n".join(lines)

new_text = new_text.replace(
    "        cur = conn.cursor(cursor_factory=DictCursor)",
    "        cur = conn.cursor(cursor_factory=DictCursor)\\n" + cte
)

with open("app.py", "w", encoding="utf-8") as f:
    f.write(new_text)

print("Replacement successful")
