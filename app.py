import os
import re
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_file, url_for, render_template
from transformers import GPT2Tokenizer, BlipProcessor, BlipForConditionalGeneration
import torch
from gtts import gTTS
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
import markdown
from bs4 import BeautifulSoup
from flask_cors import CORS
import logging
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer, util
import faiss
import sqlite3
from pathlib import Path
from PIL import Image
from datetime import datetime
import shutil
import threading
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from werkzeug.utils import secure_filename
import tempfile
import uuid

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logger.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
BASE_DIR = Path(__file__).parent.resolve()
DB_PATH = BASE_DIR / "cost_reduction.db"
MODEL_DIR = BASE_DIR / "model"
TORCHSCRIPT_MODEL_PATH = MODEL_DIR / "gpt2.pt"
FAISS_INDEX_PATH = MODEL_DIR / "faiss_index.bin"
STATIC_DIR = BASE_DIR / "static"
VLM_MODEL_PATH = MODEL_DIR / "fine_tuned_blip_model"
TEMP_DIR = BASE_DIR / "temp"
IMAGE_DIRS = {
    "proposal": BASE_DIR / "images" / "proposal",
    "mg": BASE_DIR / "images" / "mg",
    "competitor": BASE_DIR / "images" / "competitor"
}
DATA_PATH = BASE_DIR / "AIML Dummy Ideas Data.xlsx"
UPLOAD_FOLDER = TEMP_DIR / "uploads"
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_DIR
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_SIZE

# Create directories
TEMP_DIR.mkdir(exist_ok=True)
UPLOAD_FOLDER.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
for dir_path in IMAGE_DIRS.values():
    dir_path.mkdir(parents=True, exist_ok=True)

STATIC_IMAGE_DIR = STATIC_DIR / "images"
STATIC_IMAGE_DIR.mkdir(exist_ok=True)
for key, dir_path in IMAGE_DIRS.items():
    static_image_subdir = STATIC_IMAGE_DIR / key
    static_image_subdir.mkdir(exist_ok=True)
    if not any(static_image_subdir.glob("*")) and any(dir_path.glob("*")):
        try:
            shutil.copytree(dir_path, static_image_subdir, dirs_exist_ok=True)
            logger.info(f"Copied images from {dir_path} to {static_image_subdir}")
        except Exception as e:
            logger.error(f"Failed to copy images from {dir_path}: {str(e)}")

# Global variables
embedding_model = None
faiss_index = None
idea_texts = []
idea_rows = []
tokenizer = None
model = None
vlm_processor = None
vlm_model = None
proposal_embeddings = None
valid_df = None
vlm_initialized = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'
executor = ThreadPoolExecutor(max_workers=4)
vlm_init_lock = threading.Lock()
db_lock = threading.Lock()

# FIXED: Safe float conversion function
def safe_float_convert(value, default=0.0):
    """Safely convert value to float, handling various edge cases including backslashes"""
    try:
        if value is None or pd.isna(value):
            return default
        
        # Convert to string and clean
        str_value = str(value).strip()
        
        # Handle empty strings
        if not str_value or str_value.lower() in ['', 'n/a', 'na', 'null', 'none']:
            return default
            
        # Handle backslashes and other invalid characters
        if str_value in ['\\', '\\\\', '/', '#N/A', '#VALUE!', '#ERROR!', '#DIV/0!']:
            logger.warning(f"Invalid numeric value encountered: '{str_value}', using default: {default}")
            return default
            
        # Remove commas and currency symbols but preserve decimal points and negative signs
        str_value = re.sub(r'[^\d\.\-]', '', str_value)
        
        # Handle multiple decimal points or other edge cases
        if str_value.count('.') > 1 or str_value.count('-') > 1:
            return default
            
        # Try to convert to float
        return float(str_value) if str_value and str_value not in ['.', '-', '-.'] else default
        
    except (ValueError, TypeError) as e:
        logger.warning(f"Float conversion error for value '{value}': {e}, using default: {default}")
        return default

# Database Setup
def init_db():
    """Initialize SQLite database and create or migrate ideas table."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            # Define expected table schema based on Excel columns
            expected_columns = [
                'id', 'idea_id', 'cost_reduction_idea', 'reason', 'mgi_gross_saving',
                'estimated_cost_savings', 'saving_value_inr', 'weight_saving', 'group_id', 'status',
                'way_forward', 'dept', 'target_date', 'kd_lc', 'wtd_avg', 'est_impl_date',
                'investment_cr', 'mgi_carline', 'benchmarking_carline', 'mg_product_scenario',
                'competitor_product_scenario', 'purpose_mg_product', 'purpose_competitor_product',
                'impact_other_systems', 'client_statement', 'cae_required', 'homologation_required',
                'styling_change', 'part_level_testing', 'assembly_trials', 'cad_drawing_update',
                'ecn_required', 'part_production_trials', 'vehicle_level_testing', 'idea_generated_by',
                'new_tool_required', 'new_tool_cost', 'tool_modification_required', 'tool_modification_cost',
                'variants', 'current_status', 'resp', 'mix', 'volume', 'purchase_proposal',
                'interest', 'payback_months', 'mgi_pe_feasibility', 'homeroom_approval', 'pp_approval',
                'supplier_feasibility', 'financial_feasibility', 'proposal_image_filename',
                'mg_vehicle_image', 'competitor_vehicle_image', 'user_input', 'response_text',
                'image_path', 'created_at'
            ]

            column_definitions = [
                'id INTEGER PRIMARY KEY AUTOINCREMENT',
                'idea_id TEXT', 'cost_reduction_idea TEXT', 'reason TEXT', 'mgi_gross_saving REAL',
                'estimated_cost_savings TEXT', 'saving_value_inr REAL', 'weight_saving REAL',
                'group_id TEXT', 'status TEXT', 'way_forward TEXT', 'dept TEXT', 'target_date TEXT',
                'kd_lc TEXT', 'wtd_avg REAL', 'est_impl_date TEXT', 'investment_cr REAL',
                'mgi_carline TEXT', 'benchmarking_carline TEXT', 'mg_product_scenario TEXT',
                'competitor_product_scenario TEXT', 'purpose_mg_product TEXT', 'purpose_competitor_product TEXT',
                'impact_other_systems TEXT', 'client_statement TEXT', 'cae_required TEXT',
                'homologation_required TEXT', 'styling_change TEXT', 'part_level_testing TEXT',
                'assembly_trials TEXT', 'cad_drawing_update TEXT', 'ecn_required TEXT',
                'part_production_trials TEXT', 'vehicle_level_testing TEXT', 'idea_generated_by TEXT',
                'new_tool_required TEXT', 'new_tool_cost REAL', 'tool_modification_required TEXT',
                'tool_modification_cost REAL', 'variants TEXT', 'current_status TEXT', 'resp TEXT',
                'mix TEXT', 'volume TEXT', 'purchase_proposal TEXT', 'interest TEXT',
                'payback_months TEXT', 'mgi_pe_feasibility TEXT', 'homeroom_approval TEXT',
                'pp_approval TEXT', 'supplier_feasibility TEXT', 'financial_feasibility TEXT',
                'proposal_image_filename TEXT', 'mg_vehicle_image TEXT', 'competitor_vehicle_image TEXT',
                'user_input TEXT', 'response_text TEXT', 'image_path TEXT',
                'created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
            ]

            # Check if ideas table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ideas'")
            table_exists = cursor.fetchone()

            if not table_exists:
                # Create new table
                cursor.execute(f"""
                    CREATE TABLE ideas (
                        {', '.join(column_definitions)}
                    )
                """)
                conn.commit()
                logger.info("Created new ideas table with correct schema")
                return

            # Verify existing table schema
            cursor.execute("PRAGMA table_info(ideas)")
            columns = [col[1] for col in cursor.fetchall()]

            if set(columns) == set(expected_columns):
                logger.info("Existing ideas table schema matches expected schema")
                return

            logger.warning(f"Schema mismatch: expected {expected_columns}, found {columns}")

            # Create new table with correct schema
            cursor.execute("DROP TABLE IF EXISTS ideas_new")
            cursor.execute(f"""
                CREATE TABLE ideas_new (
                    {', '.join(column_definitions)}
                )
            """)

            # Get common columns for data migration
            common_columns = [col for col in columns if col in expected_columns and col != 'id']

            if common_columns:
                # Copy data from old table to new table
                cursor.execute(f"""
                    INSERT INTO ideas_new ({', '.join(common_columns)})
                    SELECT {', '.join(common_columns)} FROM ideas
                """)
                conn.commit()
                logger.info(f"Migrated data for columns: {common_columns}")

            # Drop old table and rename new table
            cursor.execute("DROP TABLE ideas")
            cursor.execute("ALTER TABLE ideas_new RENAME TO ideas")
            conn.commit()
            logger.info("Successfully migrated ideas table to new schema")

            # Re-verify schema
            cursor.execute("PRAGMA table_info(ideas)")
            columns = [col[1] for col in cursor.fetchall()]
            if set(columns) != set(expected_columns):
                raise ValueError(f"Database schema migration failed: expected {expected_columns}, found {columns}")

            logger.info(f"Database initialized at {DB_PATH} with correct schema")

    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

# Model and Vector DB Setup
def setup_model():
    """Load GPT-2 model and tokenizer."""
    try:
        global tokenizer, model
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
        model = torch.jit.load(TORCHSCRIPT_MODEL_PATH, map_location=device)
        model.eval()
        logger.info(f"TorchScript GPT-2 model loaded from {TORCHSCRIPT_MODEL_PATH} on {device}")
    except Exception as e:
        logger.error(f"Error loading GPT-2 model: {str(e)}")
        raise

def normalize_text(text):
    """Normalize text for better similarity matching."""
    try:
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [t for t in tokens if t not in stop_words]
        return ' '.join(tokens)
    except Exception as e:
        logger.error(f"Error normalizing text: {str(e)}")
        return str(text).lower()

def setup_vlm():
    """Load BLIP VLM and process Excel data."""
    global vlm_processor, vlm_model, embedding_model, proposal_embeddings, valid_df, vlm_initialized

    with vlm_init_lock:
        if vlm_initialized:
            logger.info("VLM already initialized, skipping setup.")
            return

        try:
            vlm_processor = BlipProcessor.from_pretrained(VLM_MODEL_PATH, local_files_only=True)
            vlm_model = BlipForConditionalGeneration.from_pretrained(VLM_MODEL_PATH, local_files_only=True).to(device)
            vlm_model.eval()
            logger.info(f"VLM model (BLIP) loaded from {VLM_MODEL_PATH} on {device}")

            embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

            df = pd.read_excel(DATA_PATH)
            logger.info(f"Loaded Excel file with {len(df)} rows")

            # Map Excel column names to database column names
            column_mapping = {
                'Idea Id': 'idea_id',
                'Cost Reduction Idea Proposal': 'cost_reduction_idea',
                'Reason of cost reduction proposal': 'reason',
                'MGI Estimated Gross saving': 'mgi_gross_saving',
                'Estimated Cost Savings - Breakup': 'estimated_cost_savings',
                'Saving Value(INR)': 'saving_value_inr',
                'Weight Saving(Kg)': 'weight_saving',
                'Group ID': 'group_id',
                'Status (OK/TBD/NG)': 'status',
                'Way Forward': 'way_forward',
                'Dept': 'dept',
                'Target Date': 'target_date',
                'KD/LC': 'kd_lc',
                'Wtd. Avg.': 'wtd_avg',
                'Est. Impl. Date': 'est_impl_date',
                'Investment (Cr)': 'investment_cr',
                'MGI carline': 'mgi_carline',
                'Banchmarking carline': 'benchmarking_carline',
                'MG Product Scenario': 'mg_product_scenario',
                'Competitor Product Scenario': 'competitor_product_scenario',
                'Purpose on MG product': 'purpose_mg_product',
                'Purpose on Competitor Product': 'purpose_competitor_product',
                'Impact on Other Systems': 'impact_other_systems',
                'Client Proposal/Rejection Statement': 'client_statement',
                'CAE Required (No/Yes)': 'cae_required',
                'Homologation Required (No/Yes)': 'homologation_required',
                'Styling Change (No/Yes)': 'styling_change',
                'Part Level Testing (No/Yes)': 'part_level_testing',
                'Assembly Trials (No/Yes)': 'assembly_trials',
                'CAD Drawing Update (No/Yes)': 'cad_drawing_update',
                'ECN Required (No/Yes)': 'ecn_required',
                'Part Production Trials (No/Yes)': 'part_production_trials',
                'Vehicle Level Testing (No/Yes)': 'vehicle_level_testing',
                'Idea Generated By': 'idea_generated_by',
                'New Tool Required (No/Yes)': 'new_tool_required',
                'New Tool Cost': 'new_tool_cost',
                'Tool Modification Required (No/Yes)': 'tool_modification_required',
                'Tool Modification Cost': 'tool_modification_cost',
                'Variants': 'variants',
                'Current Status': 'current_status',
                'Resp': 'resp',
                'Mix': 'mix',
                'Volume': 'volume',
                'Purchase Praposal': 'purchase_proposal',
                'Interest': 'interest',
                'Payback months': 'payback_months',
                'MGI PE Feasibilty': 'mgi_pe_feasibility',
                'Homeroom Approval': 'homeroom_approval',
                'PP Approval': 'pp_approval',
                'Supplier feasibility': 'supplier_feasibility',
                'Financial Feasibility': 'financial_feasibility',
                'Proposal Image': 'proposal_image_filename',
                'MG Vehicle Image': 'mg_vehicle_image',
                'Competitor Vehicle Image': 'competitor_vehicle_image'
            }

            df.rename(columns=column_mapping, inplace=True)
            df['normalized_cost_reduction_idea'] = df['cost_reduction_idea'].apply(normalize_text)
            df = df[df['cost_reduction_idea'].notnull()].reset_index(drop=True)

            image_files = sorted([f.name for f in IMAGE_DIRS["proposal"].glob("*.jpg")])
            df['proposal_image_filename'] = image_files[:len(df)]

            def file_exists(row):
                fname = row['proposal_image_filename']
                exists = fname in image_files
                if not exists:
                    logger.warning(f"Missing proposal image file: {fname}")
                return exists

            global valid_df
            valid_df = df[df.apply(file_exists, axis=1)].reset_index(drop=True)

            if valid_df.empty:
                logger.error("No valid data found after filtering")
                vlm_initialized = False
                return

            proposal_texts = valid_df['cost_reduction_idea'].apply(str).tolist()
            global proposal_embeddings
            proposal_embeddings = embedding_model.encode(proposal_texts, convert_to_tensor=True, device=device)
            logger.info(f"Generated embeddings for {len(proposal_texts)} proposals")

            # Populate database with Excel data
            with db_lock:
                with sqlite3.connect(DB_PATH) as conn:
                    cursor = conn.cursor()
                    for _, row in valid_df.iterrows():
                        cursor.execute("""
                            INSERT OR IGNORE INTO ideas (
                                idea_id, cost_reduction_idea, reason, mgi_gross_saving,
                                estimated_cost_savings, saving_value_inr, weight_saving,
                                group_id, status, way_forward, dept, target_date, kd_lc,
                                wtd_avg, est_impl_date, investment_cr, mgi_carline,
                                benchmarking_carline, mg_product_scenario, competitor_product_scenario,
                                purpose_mg_product, purpose_competitor_product, impact_other_systems,
                                client_statement, cae_required, homologation_required, styling_change,
                                part_level_testing, assembly_trials, cad_drawing_update, ecn_required,
                                part_production_trials, vehicle_level_testing, idea_generated_by,
                                new_tool_required, new_tool_cost, tool_modification_required,
                                tool_modification_cost, variants, current_status, resp, mix,
                                volume, purchase_proposal, interest, payback_months, mgi_pe_feasibility,
                                homeroom_approval, pp_approval, supplier_feasibility, financial_feasibility,
                                proposal_image_filename, mg_vehicle_image, competitor_vehicle_image
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            str(row.get('idea_id', 'N/A')),
                            str(row.get('cost_reduction_idea', 'N/A')),
                            str(row.get('reason', 'N/A')),
                            safe_float_convert(row.get('mgi_gross_saving')),  # FIXED: Using safe conversion
                            str(row.get('estimated_cost_savings', 'N/A')),
                            safe_float_convert(row.get('saving_value_inr')),  # FIXED: Using safe conversion
                            safe_float_convert(row.get('weight_saving')),     # FIXED: Using safe conversion
                            str(row.get('group_id', 'N/A')),
                            str(row.get('status', 'N/A')),
                            str(row.get('way_forward', 'N/A')),
                            str(row.get('dept', 'N/A')),
                            str(row.get('target_date', 'N/A')),
                            str(row.get('kd_lc', 'N/A')),
                            safe_float_convert(row.get('wtd_avg')),           # FIXED: Using safe conversion
                            str(row.get('est_impl_date', 'N/A')),
                            safe_float_convert(row.get('investment_cr')),     # FIXED: Using safe conversion
                            str(row.get('mgi_carline', 'N/A')),
                            str(row.get('benchmarking_carline', 'N/A')),
                            str(row.get('mg_product_scenario', 'N/A')),
                            str(row.get('competitor_product_scenario', 'N/A')),
                            str(row.get('purpose_mg_product', 'N/A')),
                            str(row.get('purpose_competitor_product', 'N/A')),
                            str(row.get('impact_other_systems', 'N/A')),
                            str(row.get('client_statement', 'N/A')),
                            str(row.get('cae_required', 'N/A')),
                            str(row.get('homologation_required', 'N/A')),
                            str(row.get('styling_change', 'N/A')),
                            str(row.get('part_level_testing', 'N/A')),
                            str(row.get('assembly_trials', 'N/A')),
                            str(row.get('cad_drawing_update', 'N/A')),
                            str(row.get('ecn_required', 'N/A')),
                            str(row.get('part_production_trials', 'N/A')),
                            str(row.get('vehicle_level_testing', 'N/A')),
                            str(row.get('idea_generated_by', 'N/A')),
                            str(row.get('new_tool_required', 'N/A')),
                            safe_float_convert(row.get('new_tool_cost')),     # FIXED: Using safe conversion
                            str(row.get('tool_modification_required', 'N/A')),
                            safe_float_convert(row.get('tool_modification_cost')),  # FIXED: Using safe conversion
                            str(row.get('variants', 'N/A')),
                            str(row.get('current_status', 'N/A')),
                            str(row.get('resp', 'N/A')),
                            str(row.get('mix', 'N/A')),
                            str(row.get('volume', 'N/A')),
                            str(row.get('purchase_proposal', 'N/A')),
                            str(row.get('interest', 'N/A')),
                            str(row.get('payback_months', 'N/A')),
                            str(row.get('mgi_pe_feasibility', 'N/A')),
                            str(row.get('homeroom_approval', 'N/A')),
                            str(row.get('pp_approval', 'N/A')),
                            str(row.get('supplier_feasibility', 'N/A')),
                            str(row.get('financial_feasibility', 'N/A')),
                            row.get('proposal_image_filename'),
                            row.get('mg_vehicle_image'),
                            row.get('competitor_vehicle_image')
                        ))
                    conn.commit()
                    logger.info(f"Populated database with {len(valid_df)} ideas from Excel")

            vlm_initialized = True
            logger.info(f"VLM setup complete with {len(valid_df)} valid entries")

        except Exception as e:
            logger.error(f"Error setting up VLM: {str(e)}")
            vlm_processor = None
            vlm_model = None
            valid_df = None
            proposal_embeddings = None
            vlm_initialized = False

def build_vector_db():
    """Build FAISS index from SQLite database."""
    global embedding_model, faiss_index, idea_texts, idea_rows

    try:
        logger.info("Building vector database...")

        if embedding_model is None:
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
            logger.info("SentenceTransformer model initialized successfully")

        faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
        logger.info(f"FAISS index loaded from {FAISS_INDEX_PATH}")

        with db_lock:
            with sqlite3.connect(DB_PATH) as conn:
                # Verify table exists and has required columns
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(ideas)")
                columns = [col[1] for col in cursor.fetchall()]

                required_columns = [
                    'idea_id', 'cost_reduction_idea', 'reason', 'mgi_gross_saving',
                    'estimated_cost_savings', 'saving_value_inr', 'weight_saving', 'group_id', 'status',
                    'way_forward', 'dept', 'target_date', 'kd_lc', 'wtd_avg', 'est_impl_date',
                    'investment_cr', 'mgi_carline', 'benchmarking_carline', 'mg_product_scenario',
                    'competitor_product_scenario', 'purpose_mg_product', 'purpose_competitor_product',
                    'impact_other_systems', 'client_statement', 'cae_required', 'homologation_required',
                    'styling_change', 'part_level_testing', 'assembly_trials', 'cad_drawing_update',
                    'ecn_required', 'part_production_trials', 'vehicle_level_testing', 'idea_generated_by',
                    'new_tool_required', 'new_tool_cost', 'tool_modification_required', 'tool_modification_cost',
                    'variants', 'current_status', 'resp', 'mix', 'volume', 'purchase_proposal',
                    'interest', 'payback_months', 'mgi_pe_feasibility', 'homeroom_approval', 'pp_approval',
                    'supplier_feasibility', 'financial_feasibility', 'proposal_image_filename',
                    'mg_vehicle_image', 'competitor_vehicle_image'
                ]

                if not all(col in columns for col in required_columns):
                    missing = [col for col in required_columns if col not in columns]
                    logger.error(f"Missing columns in ideas table: {missing}")
                    raise ValueError(f"Missing columns in ideas table: {missing}")

                query = f"""
                    SELECT {', '.join(required_columns)}
                    FROM ideas
                    WHERE idea_id IS NOT NULL AND cost_reduction_idea IS NOT NULL
                """
                df = pd.read_sql_query(query, conn)
                logger.info(f"Retrieved {len(df)} ideas from SQLite database")

                if df.empty:
                    logger.warning("No valid ideas found in the database")
                    # Populate with Excel data as fallback
                    setup_vlm()
                    with sqlite3.connect(DB_PATH) as conn:
                        df = pd.read_sql_query(query, conn)
                        logger.info(f"Retrieved {len(df)} ideas after fallback population")

                    if df.empty:
                        raise ValueError("No valid ideas found in the database after fallback")

                idea_texts.clear()
                idea_rows.clear()

                for _, row in df.iterrows():
                    text = ' '.join(str(row.get(col, '')) for col in df.columns if pd.notna(row.get(col)))
                    if text.strip():
                        idea_texts.append(text)
                        idea_rows.append(row)

                logger.info(f"Vector database built with {len(idea_texts)} ideas")

    except Exception as e:
        logger.error(f"Error building vector DB: {str(e)}")
        raise

# Query Parsing and Filtering
def parse_query_filters(query):
    """Parse query to extract filters."""
    filters = {}
    query_lower = query.lower()

    # Numeric filters
    for field in ['saving value', 'mgi gross saving', 'estimated cost savings', 'investment', 'wtd. avg.', 'new tool cost', 'tool modification cost']:
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

    # Status filters
    for status in ['ok', 'tbd', 'ng']:
        if f'status {status}' in query_lower:
            filters['status'] = status.upper()

    # Department filters
    for dept in ['engineering', 'manufacturing', 'procurement', 'design']:
        if dept in query_lower:
            filters['dept'] = dept

    # KD/LC filters
    for kd_lc in ['kd', 'lc']:
        if kd_lc in query_lower:
            filters['kd_lc'] = kd_lc.upper()

    # System filters
    for system in ['brake', 'suspension', 'engine', 'transmission', 'body', 'chassis', 'door', 'paint']:
        if system in query_lower:
            filters['system'] = system

    # Image display filters
    filters['show_proposal'] = 'proposal' in query_lower or 'all' in query_lower
    filters['show_mg'] = 'mg' in query_lower or 'all' in query_lower
    filters['show_competitor'] = 'competitor' in query_lower or 'all' in query_lower

    return filters

def filter_ideas(rows, filters):
    """Filter ideas based on parsed filters."""
    filtered = []

    for row in rows:
        # Numeric filters with safe conversion
        if 'saving value' in filters:
            op, val = filters['saving value']
            saving = safe_float_convert(row.get('saving_value_inr'))  # FIXED: Using safe conversion
            if op == 'greater' and saving <= val:
                continue
            if op == 'less' and saving >= val:
                continue

        if 'mgi gross saving' in filters:
            op, val = filters['mgi gross saving']
            saving = safe_float_convert(row.get('mgi_gross_saving'))  # FIXED: Using safe conversion
            if op == 'greater' and saving <= val:
                continue
            if op == 'less' and saving >= val:
                continue

        if 'estimated cost savings' in filters:
            op, val = filters['estimated cost savings']
            saving = safe_float_convert(row.get('estimated_cost_savings'))  # FIXED: Using safe conversion
            if op == 'greater' and saving <= val:
                continue
            if op == 'less' and saving >= val:
                continue

        if 'investment' in filters:
            op, val = filters['investment']
            investment = safe_float_convert(row.get('investment_cr'))  # FIXED: Using safe conversion
            if op == 'greater' and investment <= val:
                continue
            if op == 'less' and investment >= val:
                continue

        if 'wtd. avg.' in filters:
            op, val = filters['wtd. avg.']
            wtd_avg = safe_float_convert(row.get('wtd_avg'))  # FIXED: Using safe conversion
            if op == 'greater' and wtd_avg <= val:
                continue
            if op == 'less' and wtd_avg >= val:
                continue

        if 'new tool cost' in filters:
            op, val = filters['new tool cost']
            cost = safe_float_convert(row.get('new_tool_cost'))  # FIXED: Using safe conversion
            if op == 'greater' and cost <= val:
                continue
            if op == 'less' and cost >= val:
                continue

        if 'tool modification cost' in filters:
            op, val = filters['tool modification cost']
            cost = safe_float_convert(row.get('tool_modification_cost'))  # FIXED: Using safe conversion
            if op == 'greater' and cost <= val:
                continue
            if op == 'less' and cost >= val:
                continue

        if 'weight_saving' in filters:
            op, val = filters['weight_saving']
            weight = safe_float_convert(row.get('weight_saving'))  # FIXED: Using safe conversion
            if op == 'greater' and weight <= val:
                continue
            if op == 'less' and weight >= val:
                continue
            if op == 'equal' and weight != val:
                continue

        # Text filters
        if 'status' in filters:
            status = filters['status']
            if str(row.get('status', '')).upper() != status:
                continue

        if 'dept' in filters:
            dept = filters['dept']
            if str(row.get('dept', '')).lower() != dept:
                continue

        if 'kd_lc' in filters:
            kd_lc = filters['kd_lc']
            if str(row.get('kd_lc', '')).upper() != kd_lc:
                continue

        if 'system' in filters:
            system = filters['system']
            proposal = str(row.get('cost_reduction_idea', '')).lower()
            if system not in proposal:
                continue

        filtered.append(row)

    return filtered

def retrieve_context(query, top_k=10):
    """Retrieve relevant ideas from FAISS index."""
    if embedding_model is None or faiss_index is None:
        logger.warning("Vector DB not initialized")
        return [], []

    try:
        filters = parse_query_filters(query)
        query_emb = embedding_model.encode([query], convert_to_numpy=True, device=device)
        D, I = faiss_index.search(query_emb, top_k)

        selected_rows = [idea_rows[i] for i in I[0] if i < len(idea_rows)]
        filtered_rows = filter_ideas(selected_rows, filters)

        table_data = []
        contexts = []

        for row in filtered_rows:
            image_filename = row.get('proposal_image_filename')
            image_path = image_filename if image_filename else None

            if image_path:
                src_path = IMAGE_DIRS["proposal"] / image_path
                static_path = STATIC_IMAGE_DIR / "proposal" / image_path
                if src_path.exists() and not static_path.exists():
                    try:
                        shutil.copy(src_path, static_path)
                        logger.info(f"Copied image {image_path} to {static_path}")
                    except Exception as e:
                        logger.error(f"Failed to copy image {image_path}: {str(e)}")
                        image_path = None

                image_path = url_for('static', filename=f'images/proposal/{image_path}') if image_path else "N/A"

            idea_data = {
                "Idea Id": str(row.get('idea_id', 'N/A')),
                "Cost Reduction Idea": str(row.get('cost_reduction_idea', 'N/A')),
                "Proposal Image Path": image_path,
                "MGI Gross Saving": f"INR {safe_float_convert(row.get('mgi_gross_saving')):,}",  # FIXED: Using safe conversion
                "Way Forward": str(row.get('way_forward', 'N/A')),
                "Estimated Cost Savings": str(row.get('estimated_cost_savings', 'N/A')),
                "MGI Carline": str(row.get('mgi_carline', 'N/A')),
                "Saving Value (INR)": f"INR {safe_float_convert(row.get('saving_value_inr')):,}",  # FIXED: Using safe conversion
                "Weight Saving (Kg)": f"{safe_float_convert(row.get('weight_saving')):.2f}",  # FIXED: Using safe conversion
                "Status": str(row.get('status', 'N/A')),
                "Dept": str(row.get('dept', 'N/A')),
                "KD/LC": str(row.get('kd_lc', 'N/A')),
                # Additional fields for PPT
                "Idea ID": str(row.get('idea_id', 'N/A')),
                "Proposal": str(row.get('cost_reduction_idea', 'N/A')),
                "Imp": str(row.get('way_forward', 'N/A')),
                "Saving Value": f"INR {safe_float_convert(row.get('saving_value_inr')):,}",
                "Responsibility": str(row.get('resp', 'N/A')),
                "Date": str(row.get('target_date', 'N/A'))
            }

            table_data.append(idea_data)
            contexts.append("\n".join([f"- {k}: {v}" for k, v in idea_data.items()]))

        return table_data, "\n\n".join(contexts)

    except Exception as e:
        logger.error(f"Error in retrieve_context: {str(e)}")
        return [], []

def infer_vlm(image, query, top_k=10):
    """Run VLM inference on an image and query."""
    if not vlm_initialized:
        logger.warning("VLM not initialized")
        return "VLM not initialized", None, [], [], {}

    try:
        image = image.convert("RGB")
        inputs = vlm_processor(images=image, text=query, return_tensors="pt").to(device)

        with torch.no_grad():
            output = vlm_model.generate(**inputs, max_new_tokens=50)

        generated_text = vlm_processor.decode(output[0], skip_special_tokens=True)
        logger.info(f"BLIP generated text: {generated_text}")

        query_embedding = embedding_model.encode(generated_text, convert_to_tensor=True, device=device)
        scores = util.pytorch_cos_sim(query_embedding, proposal_embeddings)[0]

        top_k = min(top_k, len(valid_df))
        top_scores, top_indices = torch.topk(scores, k=top_k)

        result_rows = valid_df.iloc[top_indices.tolist()].copy()
        logger.info(f"Retrieved {len(result_rows)} rows from valid_df")

        def get_proposal_image_path(x):
            if pd.isna(x):
                logger.warning("Proposal image filename is NaN")
                return "N/A"

            path = IMAGE_DIRS["proposal"] / x
            static_path = STATIC_IMAGE_DIR / "proposal" / x

            if path.exists():
                if not static_path.exists():
                    try:
                        shutil.copy(path, static_path)
                        logger.info(f"Copied image {x} to {static_path}")
                    except Exception as e:
                        logger.error(f"Failed to copy image {x}: {str(e)}")
                return url_for('static', filename=f'images/proposal/{x}')

            logger.warning(f"Proposal image missing: {path}")
            return "N/A"

        result_rows['Proposal Image Path'] = result_rows['proposal_image_filename'].apply(get_proposal_image_path)

        table_data = []
        for _, row in result_rows.iterrows():
            idea_data = {
                "Idea Id": str(row.get('idea_id', 'N/A')),
                "Cost Reduction Idea": str(row.get('cost_reduction_idea', 'N/A')),
                "Proposal Image Path": row.get('Proposal Image Path', 'N/A'),
                "MGI Gross Saving": f"INR {safe_float_convert(row.get('mgi_gross_saving')):,}",  # FIXED: Using safe conversion
                "Way Forward": str(row.get('way_forward', 'N/A')),
                "Estimated Cost Savings": str(row.get('estimated_cost_savings', 'N/A')),
                "MGI Carline": str(row.get('mgi_carline', 'N/A')),
                "Saving Value (INR)": f"INR {safe_float_convert(row.get('saving_value_inr')):,}",  # FIXED: Using safe conversion
                "Weight Saving (Kg)": f"{safe_float_convert(row.get('weight_saving')):.2f}",  # FIXED: Using safe conversion
                "Status": str(row.get('status', 'N/A')),
                "Dept": str(row.get('dept', 'N/A')),
                "KD/LC": str(row.get('kd_lc', 'N/A')),
                # Additional fields for PPT
                "Idea ID": str(row.get('idea_id', 'N/A')),
                "Proposal": str(row.get('cost_reduction_idea', 'N/A')),
                "Imp": str(row.get('way_forward', 'N/A')),
                "Saving Value": f"INR {safe_float_convert(row.get('saving_value_inr')):,}",
                "Responsibility": str(row.get('resp', 'N/A')),
                "Date": str(row.get('target_date', 'N/A'))
            }
            table_data.append(idea_data)

        image_urls = {
            'proposal': [url for url in result_rows['Proposal Image Path'].dropna().tolist() if url != "N/A"],
            'mg': [url_for('static', filename=f'images/mg/{x}') for x in result_rows['mg_vehicle_image'].dropna() if (IMAGE_DIRS['mg'] / x).exists()],
            'competitor': [url_for('static', filename=f'images/competitor/{x}') for x in result_rows['competitor_vehicle_image'].dropna() if (IMAGE_DIRS['competitor'] / x).exists()]
        }

        return None, generated_text, table_data, top_scores.tolist(), image_urls

    except Exception as e:
        logger.error(f"Error in infer_vlm: {str(e)}")
        return str(e), None, [], [], {}

def generate_response(user_query, image_path=None):
    """Generate response using VLM or LLM based on input."""
    try:
        if image_path:
            # VLM processing
            if not vlm_initialized:
                setup_vlm()
                if not vlm_initialized:
                    logger.warning("VLM initialization failed, falling back to LLM")
                    table_data, response_text = retrieve_context(user_query)
                    csv_filename = generate_csv_from_table(table_data)
                    ppt_filename = generate_ppt_from_response(response_text, table_data)
                    return {
                        "response_text": response_text or "No relevant cost reduction ideas found",
                        "table_data": table_data,
                        "image_urls": [],
                        "csv_url": url_for('download_csv', filename=csv_filename) if csv_filename else "",
                        "ppt_url": url_for('download_ppt', filename=ppt_filename) if ppt_filename else "",
                        "vlm_warning": "VLM unavailable due to initialization failure"
                    }

            image = Image.open(image_path)
            error, generated_text, table_data, similarity_scores, image_urls = infer_vlm(
                image, user_query, top_k=10
            )

            if error:
                logger.error(f"VLM processing failed: {error}")
                table_data, response_text = retrieve_context(user_query)
                csv_filename = generate_csv_from_table(table_data)
                ppt_filename = generate_ppt_from_response(response_text, table_data)
                return {
                    "response_text": response_text or "No relevant cost reduction ideas found",
                    "table_data": table_data,
                    "image_urls": [],
                    "csv_url": url_for('download_csv', filename=csv_filename) if csv_filename else "",
                    "ppt_url": url_for('download_ppt', filename=ppt_filename) if ppt_filename else "",
                    "vlm_warning": f"VLM processing failed: {error}"
                }

            # Store VLM result in database
            with db_lock:
                with sqlite3.connect(DB_PATH) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO ideas (
                            idea_id, cost_reduction_idea, mgi_gross_saving, way_forward,
                            estimated_cost_savings, mgi_carline, saving_value_inr, weight_saving,
                            status, dept, kd_lc, proposal_image_filename, user_input, response_text, image_path
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        table_data[0].get('Idea Id') if table_data else 'N/A',
                        table_data[0].get('Cost Reduction Idea') if table_data else user_query,
                        safe_float_convert(str(table_data[0].get('MGI Gross Saving', '0')).replace('INR ', '').replace(',', '')) if table_data else 0,  # FIXED: Using safe conversion
                        table_data[0].get('Way Forward') if table_data else 'N/A',
                        table_data[0].get('Estimated Cost Savings') if table_data else 'N/A',
                        table_data[0].get('MGI Carline') if table_data else 'N/A',
                        safe_float_convert(str(table_data[0].get('Saving Value (INR)', '0')).replace('INR ', '').replace(',', '')) if table_data else 0,  # FIXED: Using safe conversion
                        safe_float_convert(table_data[0].get('Weight Saving (Kg)', '0')) if table_data else 0,  # FIXED: Using safe conversion
                        table_data[0].get('Status') if table_data else 'N/A',
                        table_data[0].get('Dept') if table_data else 'N/A',
                        table_data[0].get('KD/LC') if table_data else 'N/A',
                        os.path.basename(image_path) if image_path else None,
                        user_query,
                        generated_text,
                        os.path.basename(image_path) if image_path else None
                    ))
                    conn.commit()

            csv_filename = generate_csv_from_table(table_data)
            ppt_filename = generate_ppt_from_response(generated_text, table_data)
            
            # Log file generation results
            logger.info(f"Generated CSV filename: {csv_filename}")
            logger.info(f"Generated PPT filename: {ppt_filename}")

            return {
                "response_text": generated_text,
                "table_data": table_data,
                "image_urls": image_urls['proposal'],
                "csv_url": url_for('download_csv', filename=csv_filename) if csv_filename else "",
                "ppt_url": url_for('download_ppt', filename=ppt_filename) if ppt_filename else "",
                "vlm_warning": None
            }

        else:
            # LLM processing
            table_data, context_str = retrieve_context(user_query)

            if not table_data:
                logger.warning(f"No relevant ideas found for query: {user_query}")
                csv_filename = generate_csv_from_table(table_data)
                ppt_filename = generate_ppt_from_response("", table_data)
                logger.info(f"No table data - CSV filename: {csv_filename}, PPT filename: {ppt_filename}")
                return {
                    "response_text": f"No relevant cost reduction ideas found for query: {user_query}",
                    "table_data": [],
                    "image_urls": [],
                    "csv_url": url_for('download_csv', filename=csv_filename) if csv_filename else "",
                    "ppt_url": url_for('download_ppt', filename=ppt_filename) if ppt_filename else ""
                }

            prompt = f"""
### Instruction:
You are a cost-optimization expert. Analyse the following cost-reduction query.

Query: {user_query}

### Context:
{context_str}

### Guidelines:
1. Analyse only the cost-reduction ideas relevant to the query.
2. Compare the proposed solutions, focusing on feasibility and impact.
3. Highlight potential savings and investment requirements.
4. Consider department, status, and implementation dates.
5. Provide actionable recommendations.

### Response:
"""

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                generated = inputs['input_ids']
                attn = inputs['attention_mask']
                
                # FIXED: Add timeout and better loop control to prevent infinite loops
                max_tokens = 512
                timeout_count = 0
                max_timeout = 50  # Maximum iterations without progress
                
                for i in range(max_tokens):
                    try:
                        logits = model(generated, attn)
                        probs = torch.softmax(logits[:, -1, :], dim=-1)
                        next_token = torch.multinomial(probs, 1)
                        
                        # Check if we got a valid token
                        if next_token.item() == tokenizer.eos_token_id:
                            logger.info(f"Generation completed at token {i} with EOS token")
                            break
                            
                        generated = torch.cat([generated, next_token], dim=-1)
                        attn = torch.cat([attn, torch.ones_like(next_token)], dim=-1)
                        
                        # Reset timeout counter on successful token generation
                        timeout_count = 0
                        
                        # Add additional stopping conditions
                        if generated.size(1) > 2048:  # Prevent extremely long sequences
                            logger.warning("Generation stopped due to maximum sequence length")
                            break
                            
                    except Exception as e:
                        logger.error(f"Error during token generation at step {i}: {e}")
                        timeout_count += 1
                        if timeout_count > max_timeout:
                            logger.error("Generation stopped due to repeated errors")
                            break
                        continue

            response = tokenizer.decode(generated[0], skip_special_tokens=True)
            response = response.split("### Response:")[-1].strip()

            # Store LLM result in database
            with db_lock:
                with sqlite3.connect(DB_PATH) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO ideas (
                            idea_id, cost_reduction_idea, mgi_gross_saving, way_forward,
                            estimated_cost_savings, mgi_carline, saving_value_inr, weight_saving,
                            status, dept, kd_lc, user_input, response_text
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        table_data[0].get('Idea Id') if table_data else 'N/A',
                        table_data[0].get('Cost Reduction Idea') if table_data else user_query,
                        safe_float_convert(str(table_data[0].get('MGI Gross Saving', '0')).replace('INR ', '').replace(',', '')) if table_data else 0,  # FIXED: Using safe conversion
                        table_data[0].get('Way Forward') if table_data else 'N/A',
                        table_data[0].get('Estimated Cost Savings') if table_data else 'N/A',
                        table_data[0].get('MGI Carline') if table_data else 'N/A',
                        safe_float_convert(str(table_data[0].get('Saving Value (INR)', '0')).replace('INR ', '').replace(',', '')) if table_data else 0,  # FIXED: Using safe conversion
                        safe_float_convert(table_data[0].get('Weight Saving (Kg)', '0')) if table_data else 0,  # FIXED: Using safe conversion
                        table_data[0].get('Status') if table_data else 'N/A',
                        table_data[0].get('Dept') if table_data else 'N/A',
                        table_data[0].get('KD/LC') if table_data else 'N/A',
                        user_query,
                        response
                    ))
                    conn.commit()

            csv_filename = generate_csv_from_table(table_data)
            ppt_filename = generate_ppt_from_response(response, table_data)
            
            # Log file generation results
            logger.info(f"Generated CSV filename: {csv_filename}")
            logger.info(f"Generated PPT filename: {ppt_filename}")

            return {
                    "response_text": response,
                    "table_data": table_data,   # <-- This MUST be provided for the table to render
                    "image_urls": [row.get('Proposal Image Path', 'N/A') for row in table_data if row.get('Proposal Image Path') != 'N/A'],
                    "csv_url": url_for('download_csv', filename=csv_filename) if csv_filename else "",
                    "ppt_url": url_for('download_ppt', filename=ppt_filename) if ppt_filename else ""
                }
        
    except Exception as e:
        logger.error(f"Error in generate_response: {str(e)}")
        return {
            "response_text": f"Error processing query: {str(e)}",
            "table_data": [],
            "image_urls": [],
            "csv_url": "",
            "ppt_url": "",
            "vlm_warning": str(e) if image_path else None
        }

def generate_html_table(table_data):
    """Generate HTML table from table_data."""
    if not table_data:
        return "<p>No relevant ideas found.</p>"

    headers = [
        'Idea Id', 'Cost Reduction Idea', 'Proposal Image', 'MGI Gross Saving',
        'Way Forward', 'Estimated Cost Savings', 'MGI Carline', 'Saving Value (INR)',
        'Weight Saving (Kg)', 'Status', 'Dept', 'KD/LC'
    ]

    html = ['<table class="table table-striped table-hover">']
    html.append('<thead class="table-dark">')
    html.append('<tr>')
    html.extend([f'<th>{h}</th>' for h in headers])
    html.append('</tr>')
    html.append('</thead>')
    html.append('<tbody>')

    for row in table_data:
        html.append('<tr>')
        for key in headers:
            if key == 'Proposal Image':
                img = row.get('Proposal Image Path', 'N/A')
                html.append(f'<td><img src="{img}" alt="Proposal" style="width:50px;height:50px;"></td>' if img != 'N/A' else '<td>N/A</td>')
            else:
                html.append(f'<td>{row.get(key, "N/A")}</td>')
        html.append('</tr>')

    html.append('</tbody>')
    html.append('</table>')
    return ''.join(html)

def generate_csv_from_table(table_data):
    """Generate CSV file from table data."""
    if not table_data:
        logger.warning("No table data provided for CSV generation")
        return None

    try:
        import io
        import csv

        output = io.StringIO()
        
        # Use predefined headers to ensure consistent order
        headers = [
            'Idea Id', 'Cost Reduction Idea', 'Proposal Image Path', 'MGI Gross Saving',
            'Way Forward', 'Estimated Cost Savings', 'MGI Carline', 'Saving Value (INR)',
            'Weight Saving (Kg)', 'Status', 'Dept', 'KD/LC', 'Idea ID', 'Proposal', 
            'Imp', 'Saving Value', 'Responsibility', 'Date'
        ] if table_data else []
        
        if not headers:
            logger.warning("No headers found in table data")
            return None
        
        writer = csv.DictWriter(output, fieldnames=headers)
        writer.writeheader()
        
        for row in table_data:
            # Clean the data for CSV
            clean_row = {}
            for key, value in row.items():
                if key == 'Proposal Image Path':
                    # Keep only the filename for CSV
                    clean_row[key] = str(value).split('/')[-1] if value != 'N/A' else 'N/A'
                else:
                    clean_row[key] = str(value)
            writer.writerow(clean_row)

        csv_content = output.getvalue()
        output.close()

        # Save to upload folder (same as download routes expect)
        filename = f"cost_reduction_ideas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = UPLOAD_FOLDER / filename
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            f.write(csv_content)
        
        logger.info(f"CSV file generated: {filepath} with {len(table_data)} rows")
        return filename

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

        # Helper: Add footer at bottom right
        def add_footer(slide, text="Generated by VAVE-AI"):
            left = Inches(6.5)
            top = Inches(6.8)
            width = Inches(3.0)
            height = Inches(0.3)
            txBox = slide.shapes.add_textbox(left, top, width, height)
            tf = txBox.text_frame
            p = tf.paragraphs[0]
            p.text = text
            p.font.size = Pt(10)
            p.font.color.rgb = RGBColor(150, 150, 150)
            p.alignment = PP_ALIGN.RIGHT

        # Slide 1: Title Slide with Logo and Custom Text
        title_slide = prs.slides.add_slide(prs.slide_layouts[6])  # Blank layout
        # Add logo
        logo_path = os.path.join(STATIC_DIR, "images", "mg.png")
        if os.path.exists(logo_path):
            try:
                title_slide.shapes.add_picture(logo_path, Inches(0.5), Inches(0.5), width=Inches(2.0))
            except Exception as e:
                logger.warning(f"Failed to add logo: {str(e)}")
        else:
            logger.warning(f"Logo file not found at {logo_path}")

        # Add text box for title content
        left, top, width, height = Inches(2.7), Inches(1.0), Inches(7.0), Inches(3.0)
        txBox = title_slide.shapes.add_textbox(left, top, width, height)
        tf = txBox.text_frame
        tf.word_wrap = True

        # JSW MG MOTOR INDIA
        p = tf.add_paragraph()
        p.text = "JSW MG MOTOR INDIA"
        p.font.size = Pt(36)
        p.font.bold = True
        p.font.color.rgb = secondary_color
        p.font.name = font_name

        # Cost Reduction Idea Analysis Presentation
        p = tf.add_paragraph()
        p.text = "Cost Reduction Idea Analysis Presentation"
        p.font.size = Pt(28)
        p.font.bold = True
        p.font.color.rgb = RGBColor(0, 0, 0)
        p.font.name = font_name

        # Presentation generated by VAVE AI
        p = tf.add_paragraph()
        p.text = "Presentation generated by VAVE AI"
        p.font.size = Pt(16)
        p.font.name = font_name

        add_footer(title_slide)

        # Slide 2: Ideas Summary Slide with Table 
        ideas_per_slide = 5
        for slide_idx, start_idx in enumerate(range(0, len(table_data), ideas_per_slide)):
            summary_slide = prs.slides.add_slide(prs.slide_layouts[6])  # Blank slide
            title_shape = summary_slide.shapes.add_textbox(Inches(0.5), Inches(0.2), Inches(9.0), Inches(0.4))
            title_frame = title_shape.text_frame
            p = title_frame.paragraphs[0]
            p.text = f"Ideas Summary (Part {slide_idx + 1})"
            p.font.size = Pt(10)
            p.font.bold = True
            p.font.name = font_name

            # Prepare table layout
            num_rows = min(len(table_data) - start_idx, ideas_per_slide) + 1  # Data rows + header
            cols = 8
            left, top, width, height = Inches(0.2), Inches(0.7), Inches(9.6), Inches(6.5)  # Adjusted for fit
            table = summary_slide.shapes.add_table(num_rows, cols, left, top, width, height).table

            # Adjust column widths: optimized for content
            column_widths = [0.8, 2.5, 0.6, 1.0, 0.8, 1.8, 1.0, 1.1]  # Total ~9.6 inches
            for i, w in enumerate(column_widths):
                table.columns[i].width = Inches(w)

            # Headers
            headers = ["Idea ID", "Proposal", "Imp", "Saving Value", "Status", "Way Forward", "Responsibility", "Date"]
            for col_idx, header in enumerate(headers):
                cell = table.cell(0, col_idx)
                cell.text = header
                for para in cell.text_frame.paragraphs:
                    para.font.size = Pt(10)
                    para.font.bold = True
                    para.font.name = font_name
                cell.text_frame.word_wrap = True

            # Add data rows
            for row_idx, idea in enumerate(table_data[start_idx:start_idx + ideas_per_slide], start=1):
                row_values = [
                    idea.get("Idea ID", ""),
                    idea.get("Proposal", ""),
                    idea.get("Imp", ""),
                    idea.get("Saving Value", ""),
                    idea.get("Status", ""),
                    idea.get("Way Forward", ""),
                    idea.get("Responsibility", ""),
                    idea.get("Date", "")
                ]
                for col_idx, value in enumerate(row_values):
                    cell = table.cell(row_idx, col_idx)
                    cell.text = str(value)
                    for para in cell.text_frame.paragraphs:
                        para.font.size = Pt(10)
                        para.font.name = font_name
                    cell.text_frame.word_wrap = True

                # Set row height for compact fit
                table.rows[row_idx].height = Inches(0.35)

            add_footer(summary_slide, "Generated by VAVE-AI")
            
        # Slide 3: Chart Analysis Discussion with Bar Chart
        chart_slide = prs.slides.add_slide(prs.slide_layouts[5])
        title = chart_slide.shapes.title
        title.text = "Chart Analysis Discussion"
        for para in title.text_frame.paragraphs:
            para.font.size = Pt(32)
            para.font.bold = True
            para.font.name = font_name

        # Create bar chart data
        chart_data = CategoryChartData()
        chart_data.categories = [idea["Idea ID"] for idea in table_data]
        savings = [float(str(idea["Saving Value"]).replace("INR ", "").replace(",", "")) if 'INR' in str(idea["Saving Value"]) else 0 for idea in table_data]
        chart_data.add_series("Savings (INR)", savings)

        # Add bar chart
        x, y, cx, cy = Inches(0.5), Inches(1.5), Inches(9.0), Inches(4.5)
        chart = chart_slide.shapes.add_chart(
            XL_CHART_TYPE.COLUMN_CLUSTERED, x, y, cx, cy, chart_data
        ).chart
        chart.has_legend = True
        chart.legend.position = XL_LEGEND_POSITION.BOTTOM
        chart.legend.include_in_layout = False
        chart.chart_title.text_frame.text = "Cost Savings by Idea"
        for para in chart.chart_title.text_frame.paragraphs:
            para.font.size = Pt(16)
            para.font.name = font_name

        add_footer(chart_slide)

        # Slide 4: Pie Chart Slide
        pie_slide = prs.slides.add_slide(prs.slide_layouts[5])
        title = pie_slide.shapes.title
        title.text = "Pie chart for ideas status"
        for para in title.text_frame.paragraphs:
            para.font.size = Pt(32)
            para.font.bold = True
            para.font.name = font_name

        # Create pie chart data
        status_counts = {"Implemented": 0, "Status OK": 0, "Status TBD": 0, "Status NG": 0}
        for idea in table_data:
            status = idea.get("Status", "TBD")
            if status == "OK":
                status_counts["Status OK"] += 1
            elif status == "TBD":
                status_counts["Status TBD"] += 1
            elif status == "NG":
                status_counts["Status NG"] += 1
            elif status == "Implemented":
                status_counts["Implemented"] += 1

        chart_data = CategoryChartData()
        chart_data.categories = ["Implemented", "Status OK", "Status TBD", "Status NG"]
        chart_data.add_series("Status", [status_counts["Implemented"], status_counts["Status OK"], status_counts["Status TBD"], status_counts["Status NG"]])

        # Add pie chart
        x, y, cx, cy = Inches(0.5), Inches(1.5), Inches(4.5), Inches(4.5)
        chart = pie_slide.shapes.add_chart(
            XL_CHART_TYPE.PIE, x, y, cx, cy, chart_data
        ).chart
        chart.has_legend = True
        chart.legend.position = XL_LEGEND_POSITION.RIGHT
        chart.legend.include_in_layout = False
        chart.chart_title.text_frame.text = "Ideas Status Distribution"
        for para in chart.chart_title.text_frame.paragraphs:
            para.font.size = Pt(16)
            para.font.name = font_name

        # Customize pie chart colors
        chart.plots[0].series[0].points[0].format.fill.solid()  # Implemented (blue)
        chart.plots[0].series[0].points[0].format.fill.fore_color.rgb = RGBColor(0, 0, 255)
        chart.plots[0].series[0].points[1].format.fill.solid()  # Status OK (red)
        chart.plots[0].series[0].points[1].format.fill.fore_color.rgb = RGBColor(255, 0, 0)
        chart.plots[0].series[0].points[2].format.fill.solid()  # Status TBD (green)
        chart.plots[0].series[0].points[2].format.fill.fore_color.rgb = RGBColor(0, 128, 0)
        chart.plots[0].series[0].points[3].format.fill.solid()  # Status NG (purple)
        chart.plots[0].series[0].points[3].format.fill.fore_color.rgb = RGBColor(128, 0, 128)

        # Add summary text on the right side
        left, top, width, height = Inches(5.5), Inches(1.5), Inches(4.0), Inches(4.5)
        txBox = pie_slide.shapes.add_textbox(left, top, width, height)
        tf = txBox.text_frame
        tf.word_wrap = True
        total_savings = sum(float(str(idea["Saving Value"]).replace("INR ", "").replace(",", "")) if 'INR' in str(idea["Saving Value"]) else 0 for idea in table_data)
        summary_text = [
            f"• Total Ideas: {len(table_data)}",
            f"• Total Potential Savings: INR {total_savings:,.2f}",
            f"• Implemented Ideas: {status_counts['Implemented']}",
            f"• Status OK: {status_counts['Status OK']}",
            f"• Status TBD: {status_counts['Status TBD']}",
            f"• Status NG: {status_counts['Status NG']}",
            "• The provided cost reduction ideas focus on process deletions and material replacements to reduce manufacturing costs.",
            "• Key proposals include removing paint coatings from brackets and mounts, replacing PC-ABS with ABS, and switching to galvanized steel for window regulators.",
            "• These changes aim to maintain functionality while reducing costs and, in some cases, weight."
        ]
        for line in summary_text:
            p = tf.add_paragraph()
            p.text = line
            for para in tf.paragraphs:
                para.font.size = Pt(10)
                para.font.name = font_name
        add_footer(pie_slide)

        # Slide 5: Summary Slide with Bullet Points
        summary_slide = prs.slides.add_slide(prs.slide_layouts[1])
        title = summary_slide.shapes.title
        title.text = "Summary section"
        for para in title.text_frame.paragraphs:
            para.font.size = Pt(32)
            para.font.bold = True
            para.font.name = font_name
        body_shape = summary_slide.placeholders[1]
        tf = body_shape.text_frame
        tf.word_wrap = True
        summary_text = [
            f"Total Ideas: {len(table_data)} ideas analyzed.",
            f"Total Savings: INR {total_savings:,.2f} in potential cost reductions.",
            "Key Proposals: Remove paint coatings from mounts and brackets, replace PC-ABS with ABS, and use galvanized steel for window regulators.",
            "Status: All ideas currently TBD, pending further evaluation.",
            "Focus: Cost reduction through process deletion and material substitution while maintaining functionality."
        ]
        for line in summary_text:
            p = tf.add_paragraph()
            p.text = line
            p.level = 0
        for para in tf.paragraphs:
            para.font.size = Pt(18)
            para.font.name = font_name
        add_footer(summary_slide)

        ppt_filename = f"response_{id(response_text)}.pptx"
        ppt_path = os.path.join(UPLOAD_FOLDER, ppt_filename)
        prs.save(ppt_path)
        logger.info(f"PPT file generated: {ppt_filename}")
        return ppt_filename
    except Exception as e:
        logger.error(f"Error generating PPT: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# Flask Routes
@app.route('/')
def index():
    """Render the main chat interface."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_query = request.form.get('message', '').strip()
        if not user_query:
            logger.warning("Received empty or missing 'message' in POST request")
            return jsonify({
                "success": True,
                "response_text": "Please enter a query to proceed.",
                "table_data": [],
                "image_urls": [],
                "csv_url": "",
                "ppt_url": ""
            }), 200

        # Check if an image was uploaded
        image_path = None
        if 'image' in request.files:
            file = request.files['image']
            if file and allowed_file(file.filename):
                filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
                image_path = UPLOAD_FOLDER / filename
                file.save(str(image_path))
                logger.info(f"Image uploaded: {image_path}")

        # Generate response
        result = generate_response(user_query, str(image_path) if image_path else None)

        # Log table_data contents for debugging
        logger.info(f"Returning table_data with {len(result.get('table_data', []))} rows")
        if not result.get('table_data'):
            logger.warning("No table data generated for query: %s", user_query)

        # Clean up uploaded image
        if image_path and image_path.exists():
            try:
                os.remove(image_path)
                logger.info(f"Cleaned up uploaded image: {image_path}")
            except Exception as e:
                logger.error(f"Failed to clean up image {image_path}: {str(e)}")

        # Return JSON response without the redundant 'table' field
        response_data = {
            "success": True,
            "response_text": result.get('response_text', 'No response generated'),
            "table_data": result.get('table_data', []),
            "image_urls": result.get('image_urls', []),
            "csv_url": result.get('csv_url', ''),
            "ppt_url": result.get('ppt_url', ''),
            "warning": result.get('vlm_warning')
        }
        
        # Log the response data for debugging
        logger.info(f"Response data - CSV URL: {response_data['csv_url']}, PPT URL: {response_data['ppt_url']}")
        logger.info(f"Table data length: {len(response_data['table_data'])}")
        
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"An error occurred: {str(e)}",
            "table_data": [],
            "image_urls": [],
            "csv_url": "",
            "ppt_url": ""
        }), 500

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Handle image upload."""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
            filepath = UPLOAD_FOLDER / filename
            file.save(str(filepath))
            
            return jsonify({
                "message": "File uploaded successfully",
                "filename": filename,
                "filepath": str(filepath)
            })
        
        return jsonify({"error": "Invalid file type"}), 400

    except Exception as e:
        logger.error(f"Error in upload_image: {str(e)}")
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@app.route('/download_csv/<filename>')
def download_csv(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            logger.info(f"Serving CSV file: {file_path}")
            return send_file(file_path, as_attachment=True)
        else:
            logger.error(f"CSV file not found: {file_path}")
            return jsonify({"error": "File not found"}), 404
    except Exception as e:
        logger.error(f"Error serving CSV file {filename}: {str(e)}")
        return jsonify({"error": f"Download failed: {str(e)}"}), 500

@app.route('/download_ppt/<filename>')
def download_ppt(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            logger.info(f"Serving PPT file: {file_path}")
            return send_file(file_path, as_attachment=True)
        else:
            logger.error(f"PPT file not found: {file_path}")
            return jsonify({"error": "File not found"}), 404
    except Exception as e:
        logger.error(f"Error serving PPT file {filename}: {str(e)}")
        return jsonify({"error": f"Download failed: {str(e)}"}), 500
    
@app.route('/stats')
def stats():
    """Get application statistics."""
    try:
        with db_lock:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                
                # Count total ideas
                cursor.execute("SELECT COUNT(*) FROM ideas")
                total_ideas = cursor.fetchone()[0]
                
                # Count by status
                cursor.execute("SELECT status, COUNT(*) FROM ideas GROUP BY status")
                status_counts = dict(cursor.fetchall())
                
                # Count by department
                cursor.execute("SELECT dept, COUNT(*) FROM ideas GROUP BY dept")
                dept_counts = dict(cursor.fetchall())
                
                return jsonify({
                    "total_ideas": total_ideas,
                    "status_distribution": status_counts,
                    "department_distribution": dept_counts,
                    "vlm_initialized": vlm_initialized,
                    "vector_db_size": len(idea_texts)
                })
                
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return jsonify({"error": f"Failed to get stats: {str(e)}"}), 500

@app.route('/history')
def history():
    """Get recent query history."""
    try:
        with db_lock:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT user_input, response_text, created_at 
                    FROM ideas 
                    WHERE user_input IS NOT NULL 
                    ORDER BY created_at DESC 
                    LIMIT 10
                """)
                history_data = cursor.fetchall()
                
                return jsonify({
                    "history": [
                        {
                            "query": row[0],
                            "response": row[1][:200] + "..." if len(row[1]) > 200 else row[1],
                            "timestamp": row[2]
                        }
                        for row in history_data
                    ]
                })
                
    except Exception as e:
        logger.error(f"Error getting history: {str(e)}")
        return jsonify({"error": f"Failed to get history: {str(e)}"}), 500

def cleanup_temp_files():
    """Clean up old temporary files."""
    try:
        import time
        current_time = time.time()
        
        # Only clean up files in TEMP_DIR, not UPLOAD_FOLDER
        for file_path in TEMP_DIR.glob("*"):
            if file_path.is_file() and file_path.parent == TEMP_DIR:
                # Delete files older than 1 hour
                if current_time - file_path.stat().st_mtime > 3600:
                    file_path.unlink()
                    logger.info(f"Cleaned up old temp file: {file_path}")
                    
    except Exception as e:
        logger.error(f"Error cleaning up temp files: {str(e)}")

# Initialize the application
if __name__ == '__main__':
    try:
        init_db()
        setup_model()
        build_vector_db()
        
        # Clean up old temp files on startup
        cleanup_temp_files()
        
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise