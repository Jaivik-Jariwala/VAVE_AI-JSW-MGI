import os
import re
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_file, url_for, render_template
from transformers import GPT2Tokenizer
import torch
from gtts import gTTS
from pptx import Presentation
from pptx.util import Inches
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
import markdown
from bs4 import BeautifulSoup
import traceback
from flask_cors import CORS
import logging
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
import faiss
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Pt
import sqlite3
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Define paths
BASE_DIR = Path(__file__).parent.resolve()
DB_PATH = BASE_DIR / "cost_reduction.db"
MODEL_DIR = BASE_DIR / "model"
TORCHSCRIPT_MODEL_PATH = MODEL_DIR / "gpt2.pt"
FAISS_INDEX_PATH = MODEL_DIR / "faiss_index.bin"
STATIC_DIR = BASE_DIR / "static"

STATIC_DIR.mkdir(exist_ok=True)

# Global variables
embedding_model = None
faiss_index = None
idea_texts = []
idea_rows = []
tokenizer = None
model = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
executor = ThreadPoolExecutor(max_workers=4)

# ------------------- Model and Vector DB Setup -------------------
def setup_model():
    global tokenizer, model
    try:
        # Load tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
        # Load TorchScript model
        model = torch.jit.load(TORCHSCRIPT_MODEL_PATH, map_location=device)
        model.eval()
        logger.info(f"TorchScript model loaded successfully from {TORCHSCRIPT_MODEL_PATH} on {device}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def build_vector_db():
    global embedding_model, faiss_index, idea_texts, idea_rows
    try:
        # Initialize SentenceTransformer
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        
        # Load FAISS index
        if not FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_PATH}")
        faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
        
        # Load idea texts and rows from SQLite
        with sqlite3.connect(DB_PATH) as conn:
            query = "SELECT * FROM ideas"
            df = pd.read_sql_query(query, conn)
        
        idea_texts.clear()
        idea_rows.clear()
        for _, row in df.iterrows():
            text = ' '.join(str(row.get(col, '')) for col in df.columns)
            idea_texts.append(text)
            idea_rows.append(row)
        
        logger.info(f"Loaded FAISS index from {FAISS_INDEX_PATH} and {len(idea_texts)} ideas from SQLite")
    
    except Exception as e:
        logger.error(f"Error building vector DB: {e}")
        raise

# ------------------- Query Parsing and Filtering -------------------
def parse_query_filters(query):
    filters = {}
    query_lower = query.lower()
    # Numeric filters
    for field in ['saving value', 'mgi estimated gross saving', 'estimated cost savings', 'investment', 'wtd. avg.']:
        if field in query_lower:
            match = re.search(r'(greater|less) than\s*([\d.]+)', query_lower)
            if match:
                filters[field] = (match.group(1), float(match.group(2)))
    if 'weight saving' in query_lower:
        match = re.search(r'(greater|less) than\s*([\d.]+)', query_lower)
        if match:
            filters['weight_saving'] = (match.group(1), float(match.group(2)))
        else:
            match = re.search(r'weight saving\s*([\d.]+)', query_lower)
            if match:
                filters['weight_saving'] = ('equal', float(match.group(1)))
    # Categorical filters
    for status in ['ok', 'tbd', 'ng']:
        if f'status {status}' in query_lower:
            filters['status'] = status.upper()
    for dept in ['engineering', 'manufacturing', 'procurement', 'design']:
        if dept in query_lower:
            filters['dept'] = dept
    for kd_lc in ['kd', 'lc']:
        if kd_lc in query_lower:
            filters['kd_lc'] = kd_lc.upper()
    # System/keyword filters
    for system in ['brake', 'suspension', 'engine', 'transmission', 'body', 'chassis']:
        if system in query_lower:
            filters['system'] = system
    return filters

def filter_ideas(rows, filters):
    filtered = []
    for row in rows:
        # Saving value filter
        if 'saving value' in filters:
            op, val = filters['saving value']
            saving = row.get('saving_value_inr', 0)
            if pd.isnull(saving): saving = 0
            if op == 'greater' and saving <= val: continue
            if op == 'less' and saving >= val: continue
        # MGI Estimated Gross Saving filter
        if 'mgi estimated gross saving' in filters:
            op, val = filters['mgi estimated gross saving']
            saving = row.get('mgi_gross_saving', 0)
            if pd.isnull(saving): saving = 0
            if op == 'greater' and saving <= val: continue
            if op == 'less' and saving >= val: continue
        # Estimated Cost Savings filter
        if 'estimated cost savings' in filters:
            op, val = filters['estimated cost savings']
            saving = row.get('estimated_cost_savings', 0)
            if pd.isnull(saving): saving = 0
            if op == 'greater' and saving <= val: continue
            if op == 'less' and saving >= val: continue
        # Investment filter
        if 'investment' in filters:
            op, val = filters['investment']
            investment = row.get('investment_cr', 0)
            if pd.isnull(investment): investment = 0
            if op == 'greater' and investment <= val: continue
            if op == 'less' and investment >= val: continue
        # Wtd. Avg. filter
        if 'wtd. avg.' in filters:
            op, val = filters['wtd. avg.']
            wtd_avg = row.get('wtd_avg', 0)
            if pd.isnull(wtd_avg): wtd_avg = 0
            if op == 'greater' and wtd_avg <= val: continue
            if op == 'less' and wtd_avg >= val: continue
        # Weight saving filter
        if 'weight_saving' in filters:
            op, val = filters['weight_saving']
            weight = row.get('weight_saving', 0)
            if pd.isnull(weight): weight = 0
            if op == 'greater' and weight <= val: continue
            if op == 'less' and weight >= val: continue
            if op == 'equal' and weight != val: continue
        # Status filter
        if 'status' in filters:
            status = filters['status']
            if str(row.get('status', '')).upper() != status: continue
        # Dept filter
        if 'dept' in filters:
            dept = filters['dept']
            if str(row.get('dept', '')).lower() != dept: continue
        # KD/LC filter
        if 'kd_lc' in filters:
            kd_lc = filters['kd_lc']
            if str(row.get('kd_lc', '')).upper() != kd_lc: continue
        # System filter
        if 'system' in filters:
            system = filters['system']
            proposal = str(row.get('cost_reduction_idea', '')).lower()
            if system not in proposal: continue
        filtered.append(row)
    return filtered

def retrieve_context(query, top_k=10):
    if embedding_model is None or faiss_index is None:
        logger.warning("Vector DB not initialized")
        return [], []
    filters = parse_query_filters(query)
    query_emb = embedding_model.encode([query], convert_to_numpy=True, device=device)
    D, I = faiss_index.search(query_emb, top_k)
    selected_rows = [idea_rows[i] for i in I[0]]
    filtered_rows = filter_ideas(selected_rows, filters)
    table_data = []
    contexts = []
    for row in filtered_rows:
        # Convert numeric fields to float, handling strings and invalid values
        def to_float(value, default=0.0):
            try:
                return float(value) if value is not None and not pd.isna(value) else default
            except (ValueError, TypeError):
                return default

        idea_data = {
            "Idea Id": str(row.get('idea_id', 'N/A')),
            "Group ID": str(row.get('group_id', 'N/A')),
            "Status": str(row.get('status', 'N/A')),
            "Way Forward": str(row.get('way_forward', 'N/A')),
            "Dept": str(row.get('dept', 'N/A')),
            "Target Date": str(row.get('target_date', 'N/A')),
            "KD/LC": str(row.get('kd_lc', 'N/A')),
            "Weight Saving (Kg)": f"{to_float(row.get('weight_saving')):,.2f} kg" if pd.notnull(row.get('weight_saving')) else 'N/A',
            "Saving Value (INR)": f"INR {to_float(row.get('saving_value_inr')):,.2f}" if pd.notnull(row.get('saving_value_inr')) else 'N/A',
            "Cost Reduction Idea": str(row.get('cost_reduction_idea', 'N/A')),
            "Reason of Cost Reduction Proposal": str(row.get('reason', 'N/A')),
            "MGI Estimated Gross saving": f"INR {to_float(row.get('mgi_gross_saving')):,.2f}" if pd.notnull(row.get('mgi_gross_saving')) else 'N/A',
            "Estimated Cost Savings": f"INR {to_float(row.get('estimated_cost_savings')):,.2f}" if pd.notnull(row.get('estimated_cost_savings')) else 'N/A',
            "Wtd. Avg.": f"{to_float(row.get('wtd_avg')):,.2f}" if pd.notnull(row.get('wtd_avg')) else 'N/A',
            "Est. Impl. Date": str(row.get('est_impl_date', 'N/A')),
            "Investment (Cr)": f"{to_float(row.get('investment_cr')):,.2f} Cr" if pd.notnull(row.get('investment_cr')) else 'N/A',
            "MGI Carline": str(row.get('mgi_carline', 'N/A')),
            "Benchmarking Carline": str(row.get('benchmarking_carline', 'N/A')),
            "MG Product Scenario": str(row.get('mg_product_scenario', 'N/A')),
            "Competitor Product Scenario": str(row.get('competitor_product_scenario', 'N/A')),
            "Purpose on Competitor Product": str(row.get('purpose_competitor_product', 'N/A')),
            "Purpose on MG Product": str(row.get('purpose_mg_product', 'N/A'))
        }
        table_data.append(idea_data)
        contexts.append("\n".join([f"- {k}: {v}" for k, v in idea_data.items()]))
    return table_data, "\n\n".join(contexts)

# ------------------- Response Generation -------------------
def generate_response(user_query):
    try:
        table_data, context_str = retrieve_context(user_query)
        if not table_data:
            return [], f"No relevant cost reduction ideas found for query: {user_query}"
        prompt = f"""
        ### Instruction:
        You are a cost optimization expert. Analyze the following cost reduction query and provide detailed insights based on the provided context.
        Query: {user_query}

        ### Context:
        {context_str}

        ### Guidelines:
        1. Analyze only the cost reduction ideas relevant to the query.
        2. Compare the proposed solutions, focusing on their feasibility and impact.
        3. Highlight potential savings (Saving Value (INR), MGI Estimated Gross saving, Estimated Cost Savings) and investment requirements (Investment (Cr)).
        4. Consider department, status, and implementation dates in recommendations.
        5. Provide actionable recommendations tailored to the query.

        ### Response:
        """
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            # Use TorchScript model to get logits
            logits = model(inputs['input_ids'], inputs['attention_mask'])
            # Sample from logits to generate token IDs
            probabilities = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probabilities[:, -1, :], num_samples=1)
            generated_ids = inputs['input_ids']
            for _ in range(512):  # max_new_tokens
                logits = model(generated_ids, inputs['attention_mask'])
                probabilities = torch.softmax(logits[:, -1, :], dim=-1)
                next_token_id = torch.multinomial(probabilities, num_samples=1)
                generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
                # Update attention mask
                inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.ones_like(next_token_id)], dim=-1)
                # Check for EOS token
                if next_token_id.item() == tokenizer.eos_token_id:
                    break
        response_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        response_text = response_text.split("### Response:")[-1].strip()
        total_savings = 0
        mgi_savings = 0
        est_cost_savings = 0
        total_investment = 0
        for idea in table_data:
            # Saving Value (INR)
            saving_str = str(idea.get('Saving Value (INR)', 'N/A')).replace('INR ', '').replace(',', '')
            if saving_str != 'N/A':
                try:
                    total_savings += float(saving_str)
                except ValueError:
                    pass
            # MGI Estimated Gross saving
            mgi_str = str(idea.get('MGI Estimated Gross saving', 'N/A')).replace('INR ', '').replace(',', '')
            if mgi_str != 'N/A':
                try:
                    mgi_savings += float(mgi_str)
                except ValueError:
                    pass
            # Estimated Cost Savings
            est_str = str(idea.get('Estimated Cost Savings', 'N/A')).replace('INR ', '').replace(',', '')
            if est_str != 'N/A':
                try:
                    est_cost_savings += float(est_str)
                except ValueError:
                    pass
            # Investment (Cr)
            inv_str = str(idea.get('Investment (Cr)', 'N/A')).replace(' Cr', '').replace(',', '')
            if inv_str != 'N/A':
                try:
                    total_investment += float(inv_str)
                except ValueError:
                    pass
        formatted_response = f"""
        ### Summary
        - Total Ideas Analyzed: {len(table_data)}
        - Total Potential Savings (INR): INR {total_savings:,.2f}
        - MGI Estimated Gross Saving: INR {mgi_savings:,.2f}
        - Estimated Cost Savings: INR {est_cost_savings:,.2f}
        - Total Investment: {total_investment:,.2f} Cr

        ### Analysis
        {response_text}

        ### Detailed Ideas Table
        {generate_html_table(table_data)}
        """
        logger.info(f"Generated response for query: {user_query}")
        return table_data, formatted_response
    except Exception as e:
        logger.error(f"Error in generate_response: {str(e)}")
        raise

def generate_html_table(table_data):
    if not table_data:
        return "<p>No relevant ideas found.</p>"
    headers = table_data[0].keys()
    html = ['<table class="data-table">', '<thead><tr>']
    html.extend(f'<th>{header}</th>' for header in headers)
    html.append('</tr></thead><tbody>')
    for row in table_data:
        html.append('<tr>')
        html.extend(f'<td>{row.get(header, "")}</td>' for header in headers)
        html.append('</tr>')
    html.append('</tbody></table>')
    return '\n'.join(html)

# ------------------- File Generation (CSV, PPT, Audio) -------------------
def generate_csv_from_table(table_data):
    try:
        if not table_data or len(table_data) == 0:
            logger.warning("No table data to generate CSV")
            return None
        df = pd.DataFrame(table_data)
        csv_filename = f"table_{id(table_data)}.csv"
        csv_path = STATIC_DIR / csv_filename
        df.to_csv(csv_path, index=False)
        logger.info(f"CSV file generated: {csv_filename}")
        return csv_filename
    except Exception as e:
        logger.error(f"Error generating CSV: {str(e)}")
        return None

def generate_ppt_from_response(response_text, table_data):
    try:
        prs = Presentation()
        # Set theme colors
        primary_color = RGBColor(66, 133, 244)  # Google Blue
        secondary_color = RGBColor(220, 53, 69)  # Red
        font_name = "Calibri"

        # Helper: Add footer
        def add_footer(slide, text="Generated by VAVE-AI"):
            left = Inches(0.0)
            top = Inches(6.8)
            width = Inches(10)
            height = Inches(0.3)
            txBox = slide.shapes.add_textbox(left, top, width, height)
            tf = txBox.text_frame
            p = tf.paragraphs[0]
            p.text = text
            p.font.size = Pt(10)
            p.font.color.rgb = RGBColor(150, 150, 150)
            p.alignment = PP_ALIGN.RIGHT

        # Title slide
        title_slide = prs.slides.add_slide(prs.slide_layouts[0])
        title = title_slide.shapes.title
        subtitle = title_slide.placeholders[1]
        title.text = "Cost Reduction Analysis"
        title.text_frame.paragraphs[0].font.size = Pt(44)
        title.text_frame.paragraphs[0].font.bold = True
        title.text_frame.paragraphs[0].font.name = font_name
        subtitle.text = "Corporate Report by VAVE-AI"
        subtitle.text_frame.paragraphs[0].font.size = Pt(24)
        subtitle.text_frame.paragraphs[0].font.name = font_name
        add_footer(title_slide)

        # Statistics slide
        if table_data and len(table_data) > 0:
            total_ideas = len(table_data)
            total_savings = 0
            mgi_savings = 0
            est_cost_savings = 0
            total_investment = 0
            for idea in table_data:
                # Saving Value (INR)
                saving_str = str(idea.get('Saving Value (INR)', 'N/A')).replace('INR ', '').replace(',', '')
                if saving_str != 'N/A':
                    try:
                        total_savings += float(saving_str)
                    except ValueError:
                        pass
                # MGI Estimated Gross saving
                mgi_str = str(idea.get('MGI Estimated Gross saving', 'N/A')).replace('INR ', '').replace(',', '')
                if mgi_str != 'N/A':
                    try:
                        mgi_savings += float(mgi_str)
                    except ValueError:
                        pass
                # Estimated Cost Savings
                est_str = str(idea.get('Estimated Cost Savings', 'N/A')).replace('INR ', '').replace(',', '')
                if est_str != 'N/A':
                    try:
                        est_cost_savings += float(est_str)
                    except ValueError:
                        pass
                # Investment (Cr)
                inv_str = str(idea.get('Investment (Cr)', 'N/A')).replace(' Cr', '').replace(',', '')
                if inv_str != 'N/A':
                    try:
                        total_investment += float(inv_str)
                    except ValueError:
                        pass

            stats_slide = prs.slides.add_slide(prs.slide_layouts[5])
            title = stats_slide.shapes.title
            title.text = "Key Statistics"
            title.text_frame.paragraphs[0].font.size = Pt(32)
            title.text_frame.paragraphs[0].font.bold = True
            title.text_frame.paragraphs[0].font.name = font_name

            # Add stats as bullet points
            left = Inches(0.5)
            top = Inches(1.5)
            width = Inches(4.5)
            height = Inches(2.5)
            txBox = stats_slide.shapes.add_textbox(left, top, width, height)
            tf = txBox.text_frame
            tf.word_wrap = True
            stats = [
                f"Total Ideas: {total_ideas}",
                f"Total Potential Savings (INR): INR {total_savings:,.2f}",
                f"MGI Estimated Gross Saving: INR {mgi_savings:,.2f}",
                f"Estimated Cost Savings: INR {est_cost_savings:,.2f}",
                f"Total Investment: {total_investment:,.2f} Cr"
            ]
            for stat in stats:
                p = tf.add_paragraph()
                p.text = stat
                p.font.size = Pt(18)
                p.font.name = font_name
            add_footer(stats_slide)

        # Summary slide
        summary_slide = prs.slides.add_slide(prs.slide_layouts[1])
        title = summary_slide.shapes.title
        title.text = "Analysis Summary"
        title.text_frame.paragraphs[0].font.size = Pt(32)
        title.text_frame.paragraphs[0].font.bold = True
        title.text_frame.paragraphs[0].font.name = font_name
        body_shape = summary_slide.placeholders[1]
        tf = body_shape.text_frame
        tf.word_wrap = True
        summary_text = response_text.split('### Analysis')[0] if '### Analysis' in response_text else response_text
        for line in summary_text.split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and line:
                p = tf.add_paragraph()
                p.text = line
                p.font.size = Pt(18)
                p.font.name = font_name
        add_footer(summary_slide)

        # Analysis slide
        if '### Analysis' in response_text:
            analysis_slide = prs.slides.add_slide(prs.slide_layouts[1])
            title = analysis_slide.shapes.title
            title.text = "Detailed Analysis"
            title.text_frame.paragraphs[0].font.size = Pt(32)
            title.text_frame.paragraphs[0].font.bold = True
            title.text_frame.paragraphs[0].font.name = font_name
            body_shape = analysis_slide.placeholders[1]
            tf = body_shape.text_frame
            tf.word_wrap = True
            analysis_text = response_text.split('### Analysis')[1].split('### Detailed Ideas Table')[0]
            for line in analysis_text.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    p = tf.add_paragraph()
                    p.text = line
                    p.font.size = Pt(16)
                    p.font.name = font_name
            add_footer(analysis_slide)

        # Table slide
        if table_data and len(table_data) > 0:
            table_slide = prs.slides.add_slide(prs.slide_layouts[5])
            title = table_slide.shapes.title
            title.text = "Cost Reduction Ideas"
            title.text_frame.paragraphs[0].font.size = Pt(28)
            title.text_frame.paragraphs[0].font.bold = True
            title.text_frame.paragraphs[0].font.name = font_name
            rows = min(len(table_data) + 1, 16)  # Limit to 15 ideas per slide for readability
            cols = len(table_data[0].keys())
            headers = list(table_data[0].keys())
            left = Inches(0.3)
            top = Inches(1.2)
            width = Inches(9.2)
            height = Inches(4.5)
            table = table_slide.shapes.add_table(rows, cols, left, top, width, height).table
            for i, header in enumerate(headers):
                cell = table.cell(0, i)
                cell.text = header
                cell.text_frame.paragraphs[0].font.size = Pt(14)
                cell.text_frame.paragraphs[0].font.bold = True
                cell.text_frame.paragraphs[0].font.name = font_name
                cell.fill.solid()
                cell.fill.fore_color.rgb = primary_color
            for row_idx, row_data in enumerate(table_data[:rows-1], 1):
                for col_idx, header in enumerate(headers):
                    cell = table.cell(row_idx, col_idx)
                    cell.text = str(row_data.get(header, ''))
                    cell.text_frame.paragraphs[0].font.size = Pt(12)
                    cell.text_frame.paragraphs[0].font.name = font_name
            add_footer(table_slide)

        ppt_filename = f"response_{id(response_text)}.pptx"
        ppt_path = STATIC_DIR / ppt_filename
        prs.save(ppt_path)
        logger.info(f"PPT file generated: {ppt_filename}")
        return ppt_filename
    except Exception as e:
        logger.error(f"Error generating PPT: {str(e)}")
        traceback.print_exc()
        return None

def generate_audio(text):
    try:
        with app.app_context():
            clean_text = re.sub(r'<[^>]+>', '', text)
            if not clean_text.strip():
                return None
            tts = gTTS(text=clean_text, lang='en')
            audio_path = STATIC_DIR / f"audio_{id(text)}.mp3"
            tts.save(audio_path)
            return url_for('static', filename=os.path.basename(audio_path))
    except Exception as e:
        logger.error(f"Audio generation failed: {e}")
        return None

# ------------------- Flask Endpoints -------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400
        user_input = data.get('input', '').strip()
        logger.info(f"User query: {user_input}")
        if not user_input:
            return jsonify({'error': 'No input provided'}), 400
        if model is None or tokenizer is None:
            setup_model()
        if embedding_model is None or faiss_index is None:
            build_vector_db()
        table_data, response_text = generate_response(user_query=user_input)
        audio_future = executor.submit(generate_audio, response_text)
        csv_filename = generate_csv_from_table(table_data)
        ppt_filename = generate_ppt_from_response(response_text, table_data)
        try:
            audio_url = audio_future.result(timeout=10)
        except Exception as e:
            logger.error(f"Audio generation error: {str(e)}")
            audio_url = None
        response_payload = {
            'response': response_text,
            'audio': audio_url,
            'format': 'table',
            'table': table_data,
            'csv_url': url_for('download_csv', filename=csv_filename) if csv_filename else None,
            'ppt_url': url_for('download_ppt', filename=ppt_filename) if ppt_filename else None
        }
        logger.info(f"Response payload: {response_payload}")
        return jsonify(response_payload)
    except Exception as e:
        logger.error(f"Chat endpoint error: {traceback.format_exc()}")
        return jsonify({
            'error': 'An error occurred while processing your request.',
            'details': str(e) if app.debug else None
        }), 500

@app.route('/download_csv/<path:filename>')
def download_csv(filename):
    try:
        return send_file(STATIC_DIR / filename, as_attachment=True, download_name="cost_reduction_data.csv")
    except Exception as e:
        logger.error(f"Error downloading CSV: {str(e)}")
        return jsonify({'error': 'File not found'}), 404

@app.route('/download_ppt/<path:filename>')
def download_ppt(filename):
    try:
        return send_file(STATIC_DIR / filename, as_attachment=True, download_name="cost_reduction_analysis.pptx")
    except Exception as e:
        logger.error(f"Error downloading PPT: {str(e)}")
        return jsonify({'error': 'File not found'}), 404

@app.route('/stats', methods=['GET'])
def stats():
    try:
        if embedding_model is None or faiss_index is None:
            build_vector_db()
        if not DB_PATH.exists():
            raise FileNotFoundError(f"SQLite database not found at {DB_PATH}")
        with sqlite3.connect(DB_PATH) as conn:
            query = """
            SELECT 
                COUNT(*) as total_ideas,
                COALESCE(SUM(saving_value_inr), 0) as total_savings,
                COALESCE(SUM(mgi_gross_saving), 0) as mgi_savings,
                COALESCE(SUM(estimated_cost_savings), 0) as est_cost_savings,
                COALESCE(SUM(investment_cr), 0) as total_investment,
                SUM(CASE WHEN status = 'OK' THEN 1 ELSE 0 END) as ok_count,
                SUM(CASE WHEN status = 'TBD' THEN 1 ELSE 0 END) as tbd_count,
                SUM(CASE WHEN status = 'NG' THEN 1 ELSE 0 END) as ng_count
            FROM ideas
            WHERE idea_id IS NOT NULL AND cost_reduction_idea IS NOT NULL
            """
            result = conn.execute(query).fetchone()
        return jsonify({
            'total_ideas': result[0],
            'total_savings': f"INR {result[1]:,.2f}",
            'mgi_estimated_gross_saving': f"INR {result[2]:,.2f}",
            'estimated_cost_savings': f"INR {result[3]:,.2f}",
            'total_investment': f"{result[4]:,.2f} Cr",
            'status_distribution': {
                'OK': result[5],
                'TBD': result[6],
                'NG': result[7]
            }
        })
    except Exception as e:
        logger.error(f"Stats endpoint error: {str(e)}")
        return jsonify({'error': 'Failed to retrieve stats'}), 500

# ------------------- Main -------------------
if __name__ == "__main__":
    try:
        setup_model()
        build_vector_db()
        app.run(debug=False, host="0.0.0.0", port=5000)
    except Exception as e:
        logger.error(f"Application startup error: {str(e)}")
        raise