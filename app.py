import os
import re
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_file, url_for, render_template
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GenerationConfig
import torch
from gtts import gTTS
from pptx import Presentation
from pptx.util import Inches
import markdown
from bs4 import BeautifulSoup
import traceback
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXCEL_PATH = os.path.join(BASE_DIR, "AIML Dummy Ideas Data.xlsx")
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Define Excel fields for structured data extraction
EXCEL_FIELDS = {
    "Cost Reduction Proposal": [
        "Idea Id", "Weight Saving(Kg)", "Saving Value(INR)", 
        "MG Product Scenario", "Competitor Product Scenario",
        "Cost Reduction Idea Proposal", "Reason of cost reduction proposal",
        "Estimated Cost Savings - Breakup", "MGI Estimated Gross saving", "Way Forward"
    ]
}

# Global variables for model and tokenizer
tokenizer = None
model = None

def setup_gpt2_pipeline():
    global tokenizer, model
    try:
        print(f"Loading fine-tuned model from {MODEL_DIR}...")
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
        model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
        model.eval()
        if torch.cuda.is_available():
            model.cuda()
        print("Fine-tuned GPT-2 model loaded successfully.")
        return True
    except Exception as e:
        print(f"Error setting up GPT-2 pipeline: {str(e)}")
        return False

def format_fields(fields_dict):
    formatted = []
    for category, fields in fields_dict.items():
        formatted.append(f"{category}: {', '.join(fields)}")
    return "\n".join(formatted)

def extract_text_from_excel(file_path, max_rows=10):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Excel file not found at {file_path}")
        df = pd.read_excel(file_path)
        df = df.dropna(subset=["Idea Id", "Way Forward"])
        df = df.head(max_rows)
        
        text_data = []
        for _, row in df.iterrows():
            prompt = f"""### Instruction:
You are given a cost-reduction proposal with:
- Idea ID: {row['Idea Id']}
- Weight Saving: {row['Weight Saving(Kg)']} kg
- Saving Value: INR {row['Saving Value(INR)']}
- MG Scenario: {row['MG Product Scenario']}
- Competitor Scenario: {row['Competitor Product Scenario']}
- Proposal: {row['Cost Reduction Idea Proposal']}

Please provide a recommendation for this idea.

### Response:
{row['Way Forward']}"""
            text_data.append(prompt)
            
        text_data.append(f"\nRAG Context: Loaded {len(df)} records from {os.path.basename(file_path)}")
        
        return "\n\n".join(text_data)
    except Exception as e:
        print(f"Error extracting text from Excel: {str(e)}")
        return f"Error extracting Excel data: {str(e)}"

def get_exact_idea_data(idea_id):
    try:
        df = pd.read_excel(EXCEL_PATH, header=1)
        df = df.replace({np.nan: "N/A"})
        idea_row = df[df['Idea Id'].astype(str) == str(idea_id)]
        if not idea_row.empty:
            return idea_row.iloc[0].to_dict()
        return None
    except Exception as e:
        print(f"Error fetching idea data for ID {idea_id}: {str(e)}")
        return None

def generate_table_output(prompts):
    table_data = []
    for prompt in prompts:
        fields = {}
        for line in prompt.split("\n"):
            if ": " in line and not line.startswith("###"):
                key, value = line.split(": ", 1)
                key = key.strip("- ").strip()
                fields[key] = value

        row = {
            "Idea Id": fields.get("Idea ID", "N/A"),
            "Weight Saving(Kg)": fields.get("Weight Saving", "N/A").replace(" kg", ""),
            "Saving Value(INR)": fields.get("Saving Value", "N/A").replace("INR ", ""),
            "MG Product Scenario": fields.get("MG Product Scenario", "N/A"),
            "Competitor Product Scenario": fields.get("Competitor Product Scenario", "N/A"),
            "Cost Reduction Idea Proposal": fields.get("Cost Reduction Idea", "N/A"),
            "Reason of cost reduction proposal": fields.get("Reason of cost reduction proposal", "N/A"),
            "Estimated Cost Savings - Breakup": fields.get("Estimated Cost Savings", "N/A"),
            "MGI Estimated Gross saving": fields.get("MGI Estimated Gross Saving", "N/A"),
            "Way Forward": fields.get("Way Forward", "N/A")
        }
        table_data.append(row)
    
    return pd.DataFrame(table_data)

def generate_gpt2_response(query, context):
    try:
        prompt = f"""### Instruction:
Analyze this cost reduction query with RAG from our dataset: {query}
Context: {context}

Provide response with:
1. Key insights from similar historical proposals
2. Current status analysis
3. Financial impact projections
4. Risk assessment

### Response:"""
        
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, 
                         truncation=True, padding="max_length")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        generation_config = GenerationConfig(
            max_new_tokens=200,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.9,
            temperature=0.3
        )
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=generation_config
            )
        
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract generated response part
        generated_response = full_response.split("### Response:")[-1].strip()
        
        # Create table from response
        df = generate_table_output([full_response])
        html_table = df.to_html(index=False, classes="data-table")
        
        return f"## Cost Reduction Analysis\n{html_table}\n\n**RAG Sources**: Excel Database, Engineering Docs"
        
    except Exception as e:
        print(f"Error in generation: {str(e)}")
        return f"Error generating response: {str(e)}"
    
def generate_audio_response(text):
    try:
        plain_text = re.sub(r'<[^>]+>', '', text)
        plain_text = re.sub(r'\s+', ' ', plain_text).strip()
        if not plain_text:
            return None
        tts = gTTS(text=plain_text, lang='en')
        audio_path = os.path.join(BASE_DIR, f"static/audio_{id(text)}.mp3")
        tts.save(audio_path)
        return url_for('static', filename=f"audio_{id(text)}.mp3")
    except Exception as e:
        print(f"Error generating audio: {str(e)}")
        return None

def generate_csv_from_response(response):
    try:
        table_start = response.find("<table>")
        table_end = response.find("</table>")
        if table_start == -1 or table_end == -1:
            return None
        table_html = response[table_start:table_end + 8]
        soup = BeautifulSoup(table_html, 'html.parser')
        table = soup.find('table')
        headers = [th.get_text(strip=True) for th in table.find('thead').find_all('th')]
        rows = []
        for tr in table.find('tbody').find_all('tr'):
            row = [td.get_text(strip=True) for td in tr.find_all('td')]
            rows.append(row)
        df = pd.DataFrame(rows, columns=headers)
        csv_path = os.path.join(BASE_DIR, f"static/table_{id(response)}.csv")
        df.to_csv(csv_path, index=False)
        return f"table_{id(response)}.csv"
    except Exception as e:
        print(f"Error generating CSV: {str(e)}")
        return None

def generate_ppt_from_response(response, idea_id=None):
    try:
        prs = Presentation()
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        title = slide.shapes.title
        title.text = f"VAVE-AI Response {'for Idea ID ' + str(idea_id) if idea_id else ''}"
        
        body_shape = slide.placeholders[1]
        tf = body_shape.text_frame
        html_text = markdown.markdown(response)
        soup = BeautifulSoup(html_text, 'html.parser')
        
        for element in soup.children:
            if element.name == 'h1':
                p = tf.add_paragraph()
                p.text = element.get_text()
                p.level = 0
            elif element.name == 'h2':
                p = tf.add_paragraph()
                p.text = element.get_text()
                p.level = 1
            elif element.name == 'table':
                rows = len(element.find_all('tr'))
                cols = len(element.find('thead').find('tr').find_all('th'))
                table = slide.shapes.add_table(rows, cols, Inches(1), Inches(2), Inches(8), Inches(3)).table
                for i, th in enumerate(element.find('thead').find('tr').find_all('th')):
                    table.cell(0, i).text = th.get_text(strip=True)
                for i, tr in enumerate(element.find('tbody').find_all('tr'), 1):
                    for j, td in enumerate(tr.find_all('td')):
                        table.cell(i, j).text = td.get_text(strip=True)
            else:
                p = tf.add_paragraph()
                p.text = element.get_text(strip=True) if element.string else ""
                p.level = 2 if element.name == 'h3' else 3
        
        ppt_path = os.path.join(BASE_DIR, f"static/response_{id(response)}.pptx")
        prs.save(ppt_path)
        return f"response_{id(response)}.pptx"
    except Exception as e:
        print(f"Error generating PPT: {str(e)}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stats')
def stats():
    try:
        if not os.path.exists(EXCEL_PATH):
            return jsonify({
                'total_ideas': "N/A",
                'total_savings': "N/A",
                'status_distribution': {'OK': "N/A", 'TBD': "N/A", 'NG': "N/A"}
            })
        df = pd.read_excel(EXCEL_PATH, header=1)
        total_ideas = len(df)
        total_savings = df['Saving Value(INR)'].replace("N/A", 0).astype(float).sum() if 'Saving Value(INR)' in df.columns else "N/A"
        status_distribution = df['Status'].value_counts().to_dict() if 'Status' in df.columns else {'OK': 0, 'TBD': 0, 'NG': 0}
        return jsonify({
            'total_ideas': total_ideas,
            'total_savings': f"{total_savings:,.2f} INR" if isinstance(total_savings, (int, float)) else "N/A",
            'status_distribution': {
                'OK': status_distribution.get('OK', 0),
                'TBD': status_distribution.get('TBD', 0),
                'NG': status_distribution.get('NG', 0)
            }
        })
    except Exception as e:
        print(f"Error fetching stats: {str(e)}")
        return jsonify({
            'total_ideas': "N/A",
            'total_savings': "N/A",
            'status_distribution': {'OK': "N/A", 'TBD': "N/A", 'NG': "N/A"}
        })

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('input', '').strip()
        
        if not message:
            return jsonify({'error': 'Empty query'}), 400
            
        context = extract_text_from_excel(EXCEL_PATH)
        response = generate_gpt2_response(message, context)
        html_response = markdown.markdown(response, extensions=['tables'])
        
        # Generate additional outputs
        audio_url = generate_audio_response(html_response)
        csv_url = generate_csv_from_response(html_response)
        ppt_url = generate_ppt_from_response(html_response)
        
        return jsonify({
            'response': html_response,
            'format': 'table',
            'audio': url_for('download_audio', filename=audio_url.split('/')[-1]) if audio_url else None,
            'csv_url': url_for('download_csv', path=csv_url) if csv_url else None,
            'ppt_url': url_for('download_ppt', path=ppt_url) if ppt_url else None
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'trace': traceback.format_exc()
        }), 500

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'})
        if file and file.filename.endswith('.xlsx'):
            file_path = EXCEL_PATH
            file.save(file_path)
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Invalid file format. Please upload an Excel file (.xlsx)'})
    except Exception as e:
        print(f"Error uploading file: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})
    
@app.route('/download_audio/<path:path>')
def download_audio(path):
    try:
        file_path = os.path.join(BASE_DIR, "static", path)
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        print(f"Error downloading audio: {str(e)}")
        return "File not found", 404

@app.route('/download_csv/<path:path>')
def download_csv(path):
    try:
        file_path = os.path.join(BASE_DIR, "static", path)
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        print(f"Error downloading CSV: {str(e)}")
        return "File not found", 404

@app.route('/download_ppt/<path:path>')
def download_ppt(path):
    try:
        file_path = os.path.join(BASE_DIR, "static", path)
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        print(f"Error downloading PPT: {str(e)}")
        return "File not found", 404

if __name__ == "__main__":
    setup_gpt2_pipeline()
    app.run(debug=False, host='127.0.0.1', port=5000)