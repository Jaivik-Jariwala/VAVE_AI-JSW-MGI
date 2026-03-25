import pandas as pd
import io

data = """Idea ID	Visual Scenarios	Cost Reduction Idea	Way Forward	Saving (INR)	Weight Saving	Status	Feasibility	Cost Saving	Weight Reduction	Homologation Feasibility	Homologation Theory
AI-GEN-001	Competitor (KIA Seltos)	Replace current mild steel (e.g., SPCC) front fender outer panel with a higher strength steel (e.g., DP600) of 0.7mm thickness, reducing flange width by 10mm at the sill interface and eliminating a minor stiffening rib.	AI Innovation	480000	0.3	Auto-Approved	85	80	70	90	AIS 100
AI-GEN-003	Competitor (KIA Seltos)	Reduce the flange width of the A-pillar outer panel from 20mm to 15mm and replace the current steel grade (e.g., MS) with a slightly higher strength steel (e.g., CP800) with 5% lower yield strength, for cost savings.	AI Innovation	320000	0.2	Auto-Approved	70	75	60	70	FMVSS 208
AI-GEN-004	Competitor (KIA Seltos)	Eliminate the separate rain channel trim on the B-pillar outer by redesigning the B-pillar outer panel to incorporate a molded-in channel feature, using the same steel grade but a slightly thicker gauge (0.8mm to 0.9mm) for robustness.	AI Innovation	384000	-0.1	Auto-Approved	80	70	50	85	AIS 098
AI-GEN-005	Competitor (KIA Seltos)	Replace the current steel door outer panel (e.g., SPFC980) with a slightly lower strength grade (e.g., SPFC780) that has a 5% lower yield strength, and reduce the inner flange width by 8mm at the beltline.	AI Innovation	1280000	1.6	Auto-Approved	75	80	70	75	FMVSS 214
AI-GEN-006	Competitor (KIA Seltos)	Simplify the rear quarter panel inner reinforcement by consolidating two separate stamped parts into a single hydroformed or advanced stamping part using the same steel grade (e.g., DP600) but with optimized wall thickness distribution.	AI Innovation	800000	0.3	Auto-Approved	65	70	65	75	AIS 100
AI-GEN-007	Competitor (KIA Seltos)	Replace the current aluminum hood inner panel with a stamped steel panel (e.g., DP600) of optimized geometry and slightly increased thickness (1.0mm to 1.1mm) to match weight and cost targets, while eliminating a separate brace.	AI Innovation	960000	-0.5	Auto-Approved	80	75	55	85	FMVSS 219
AI-GEN-010	Competitor (KIA Seltos)	Consolidate the tailgate outer skin and inner panel into a single, hydroformed or advanced stamped part using a higher strength steel (e.g., DP1000) with optimized thickness, eliminating separate reinforcements and reducing welding.	AI Innovation	1280000	1	Auto-Approved	60	65	80	70	FMVSS 201
AI-GEN-011	Competitor (KIA Seltos)	Reduce the number of stiffening ribs on the roof outer panel by 30% and use a higher strength steel grade (e.g., DP800) with a 10% higher yield strength, optimizing bead geometry for stiffness.	AI Innovation	576000	0.3	Auto-Approved	70	75	65	75	FMVSS 216a
WEB-SOURCED-001	Competitor (KIA Seltos)	Replace current mild steel (e.g., SPCC) front fender panels with a higher strength steel like DP600 (Dual Phase steel). This substitution allows for a 0.2mm reduction in material thickness while maintaining or improving dent resistance and crash performance due to its higher yield strength (approx. 600 MPa vs. 240 MPa for SPCC). No changes to existing tooling or press lines are required as DP600 is compatible with standard stamping processes. Geometric changes are limited to minor radius adjustments on flanges for stamping optimization, not structural redesign.	World Wide Web	-	-	Industry Trend	95	-	-	-	Homologation impact
WEB-SOURCED-002	Competitor (KIA Seltos)	De-content the rear bumper fascia by removing two non-structural mounting tabs that are currently over-engineered for aesthetic alignment. These tabs do not contribute to crash performance or overall structural integrity. The remaining mounting points are sufficient for secure attachment. This involves a simple CAD modification to the existing bumper fascia design, removing approximately 50 grams of material and reducing injection molding cycle time slightly.	World Wide Web	-	-	Research Paper	90	-	-	-	Homologation impact
WEB-SOURCED-003	Competitor (KIA Seltos)	Consolidate the left and right B-pillar inner reinforcement stampings into a single, simpler stamped part. This part will be designed for compatibility with existing B-pillar outer panel tooling. The new inner reinforcement will use a readily available High Strength Steel (HSS) grade, such as DP600, ensuring comparable or improved torsional stiffness due to optimized ribbing and geometry. The consolidation reduces part count and assembly complexity.	World Wide Web	-	-	Research Paper	85	-	-	-	Homologation impact
WEB-SOURCED-004	Competitor (KIA Seltos)	Replace the current steel door inner panels (e.g., HC270P) with a thinner gauge of advanced High Strength Steel (AHSS) like DP600 or even CP800 (Complex Phase steel), with a maximum yield strength increase of 10-15% over the current material. This substitution, coupled with minor geometric stiffening features (e.g., slight increase in bead depth or radius), allows for a 0.3mm thickness reduction per panel. The material choice remains cost-effective and avoids high-grade materials that fail flammability and cost targets. Tooling remains compatible.	World Wide Web	-	-	Research Paper	88	-	-	-	Homologation impact"""

# Parse as tab-separated values
df = pd.read_csv(io.StringIO(data), sep='\t')

def classify_action(idea):
    idea = str(idea).lower()
    actions = []
    if 'replace' in idea or 'substitute' in idea or 'substitution' in idea: 
        actions.append('Material Substitution')
    if 'eliminate' in idea or 'remove' in idea or 'de-content' in idea or 'removing' in idea: 
        actions.append('De-contenting/Removal')
    if 'consolidate' in idea or 'single' in idea: 
        actions.append('Part Consolidation')
    if 'reduce' in idea or 'thickness' in idea or 'geometry' in idea or 'flange' in idea or 'stiffening' in idea: 
        actions.append('Geometric Optimization')
    if 'add' in idea or 'incorporate' in idea:
        actions.append('Feature Addition')
        
    if not actions:
        return 'Process Optimization'
    return ', '.join(actions)

def get_change_details(idea):
    idea_str = str(idea).lower()
    idea_ori = str(idea)
    
    parts = []
    if 'replace' in idea_str:
        try:
            # simple heurustic to grab what follows 'replace' and what follows 'with'
            target = idea_str.split('replace')[1].split('with')[0].strip()
            replacement = idea_str.split('with')[1].split(',')[0].strip()
            parts.append(f"From: '{target}' -> To: '{replacement}'")
        except:
            pass
            
    if 'eliminate' in idea_str:
        try:
            p = idea_str.split('eliminate')[1].split('by')[0].strip()
            parts.append(f"Removed: '{p}'")
        except:
            pass
            
    if 'removing' in idea_str or 'remove' in idea_str:
        try:
            p = idea_str.split('remov')[1].split('that')[0].strip().replace("ing ","").replace("e ","")
            parts.append(f"Removed: '{p}'")
        except: pass
            
    if 'consolidate' in idea_str:
        try:
            p = idea_str.split('consolidate')[1].split('into')[0].strip()
            dest = idea_str.split('into')[1].split('using')[0].strip()
            parts.append(f"Consolidated: '{p}' -> Into: '{dest}'")
        except:
            parts.append("Consolidated components into fewer parts")

    return ' | '.join(parts) if parts else 'N/A'

# Apply the transformations
df['Action Strategy'] = df['Cost Reduction Idea'].apply(classify_action)
df['Structural Change (From -> To)'] = df['Cost Reduction Idea'].apply(get_change_details)

# Re-arrange to make it readable: keeping ID, Strategy, Change, and original Idea
cols = ['Idea ID', 'Action Strategy', 'Structural Change (From -> To)', 'Cost Reduction Idea', 'Saving (INR)', 'Weight Saving']
df_out = df[cols]

output_file = 'advanced_sd3/compressed_ideas.xlsx'
df_out.to_excel(output_file, index=False)
print(f"Success! Compressed data exported to {output_file}")
