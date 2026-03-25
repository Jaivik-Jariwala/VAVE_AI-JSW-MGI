import os
import re
import pandas as pd
import psycopg2
from pathlib import Path
import logging
from dotenv import load_dotenv
import mimetypes

# --- Configuration ---
load_dotenv()

BASE_DIR = Path(__file__).parent.resolve()
DATA_PATH = BASE_DIR / "20260219_AIML_Data_Hector Plus 1.5 Gas CVT_Idea_Consolidated_W Image.xlsx_v1.xlsx"

IMAGE_DIRS = {
    "proposal": BASE_DIR / "static" / "images" / "proposal",
    "mg": BASE_DIR / "static" / "images" / "mg",
    "competitor": BASE_DIR / "static" / "images" / "competitor"
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_db_connection():
    """Creates a new connection to the PostgreSQL database."""
    try:
        # Check for both DB_PASS and DB_PASSWORD
        password = os.getenv('DB_PASS') or os.getenv('DB_PASSWORD')
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432'),
            dbname=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=password
        )
        return conn
    except Exception as e:
        logger.error(f"FATAL: Could not connect to database: {e}")
        raise

def init_db():
    """Initialize PostgreSQL database and create ALL required tables."""
    logger.info(f"Initializing database schema...")

    # 1. Main Ideas Table
    ideas_table_sql = """
    CREATE TABLE IF NOT EXISTS ideas (
        id SERIAL PRIMARY KEY,
        idea_id TEXT UNIQUE,
        cost_reduction_idea TEXT, reason TEXT, mgi_gross_saving REAL,
        estimated_cost_savings TEXT, saving_value_inr REAL, weight_saving REAL,
        group_id TEXT, status TEXT, way_forward TEXT, dept TEXT, target_date TEXT,
        kd_lc TEXT, wtd_avg REAL, est_impl_date TEXT, investment_cr REAL, capex REAL,
        mgi_carline TEXT, benchmarking_carline TEXT, mg_product_scenario TEXT,
        competitor_product_scenario TEXT, purpose_mg_product TEXT, purpose_competitor_product TEXT,
        impact_other_systems TEXT, client_statement TEXT, cae_required TEXT,
        homologation_required TEXT, styling_change TEXT, part_level_testing TEXT,
        assembly_trials TEXT, cad_drawing_update TEXT, ecn_required TEXT,
        part_production_trials TEXT, vehicle_level_testing TEXT, idea_generated_by TEXT,
        new_tool_required TEXT, new_tool_cost REAL, tool_modification_required TEXT,
        tool_modification_cost REAL, variants TEXT, current_status TEXT, resp TEXT,
        mix TEXT, volume TEXT, purchase_proposal TEXT, interest TEXT,
        payback_months TEXT, mgi_pe_feasibility TEXT, homeroom_approval TEXT,
        pp_approval TEXT, supplier_feasibility TEXT, financial_feasibility TEXT,
        proposal_image_filename TEXT, mg_vehicle_image TEXT, competitor_vehicle_image TEXT,
        user_input TEXT, response_text TEXT, image_path TEXT,
        proposal_image_data BYTEA, proposal_image_mimetype TEXT,
        mg_vehicle_data BYTEA, mg_vehicle_mimetype TEXT,
        competitor_vehicle_data BYTEA, competitor_vehicle_mimetype TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    # 2. Chat History
    chat_history_sql = """
    CREATE TABLE IF NOT EXISTS chat_history (
        id SERIAL PRIMARY KEY,
        username TEXT NOT NULL,
        user_query TEXT,
        response_text TEXT,
        image_path TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    # 3. Data Uploads Tracking
    uploads_sql = """
    CREATE TABLE IF NOT EXISTS data_uploads (
        id SERIAL PRIMARY KEY,
        excel_filename TEXT,
        zip_filename TEXT,
        uploaded_by TEXT,
        status TEXT DEFAULT 'active',
        message TEXT,
        records_loaded INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Execute Creations
        print("Creating ideas table...")
        cursor.execute(ideas_table_sql)
        try:
            cursor.execute("ALTER TABLE ideas ADD COLUMN IF NOT EXISTS capex REAL;")
            conn.commit()
        except Exception:
            pass
        print("Creating chat history table...")
        cursor.execute(chat_history_sql)
        
        print("Creating uploads table...")
        cursor.execute(uploads_sql)

        # FIX: MANUALLY CREATE MISSING LOGGING TABLES
        print("Creating events table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id SERIAL PRIMARY KEY,
                event_type TEXT,
                username TEXT,
                payload JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        print("Creating metrics table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics_timeseries (
                id SERIAL PRIMARY KEY,
                metric_name TEXT,
                metric_value REAL,
                labels JSONB,
                ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        conn.commit()
        logger.info("All database tables created successfully.")

    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        if conn: conn.rollback()
        raise
    finally:
        if conn: conn.close()

def safe_float_convert(value, default=0.0):
    """
    Industry-grade float converter using Regex.
    """
    try:
        if value is None or pd.isna(value):
            return default

        str_value = str(value).strip()
        if not str_value or str_value.lower() in ['', 'n/a', 'na', 'null', 'nan', 'tbd', '-', 'nil']:
            return default

        clean_value = re.sub(r'[^\d.-]', '', str_value)
        if not clean_value or clean_value in {'.', '-'}:
            return default

        return float(clean_value)
    except Exception as e:
        return default

def populate_database():
    conn = None
    logger.info(f"Reading Excel file from: {DATA_PATH}")
    if not DATA_PATH.exists():
        logger.error("Excel file not found.")
        return

    try:
        excel_data = pd.read_excel(DATA_PATH, sheet_name=None)
        # Assuming the data is in the first sheet if not named 'AIML Dummy ideas'
        sheet_name = 'AIML Dummy ideas' if 'AIML Dummy ideas' in excel_data else list(excel_data.keys())[0]
        df_ideas = excel_data[sheet_name]
        
        # Rename columns to match DB
        column_mapping = {
            'Idea Id': 'idea_id',
            'Title': 'cost_reduction_idea', # Using Title as the primary idea name
            'Why?': 'reason', # Mapping 'Why?' to reason
            'Description': 'way_forward', # Initial mapping for way_forward, will append impact later
            'Saving Value(INR)': 'saving_value_inr',
            'Weight Saving(Kg)': 'weight_saving',
            'Group': 'group_id', # It's 'Group' in the new excel, not 'Group ID'
            'Status': 'status', # It's 'Status' not 'Status (OK/TBD/NG)'
            'Impact on Other Systems': 'impact_other_systems',
            'MGI DB Saving': 'mgi_gross_saving',
            'CAPEX': 'capex',
            'Vehicle': 'mgi_carline',
            'Idea Origin': 'dept',
            'System': 'mg_product_scenario',
            'Component': 'purpose_competitor_product',
            'Sub-Component': 'part_level_testing',
            'Material': 'new_tool_required',
            'Process': 'tool_modification_required',
            'Volume': 'volume',
            'Implement Date': 'est_impl_date',
            'Proposal Image': 'proposal_image_filename',
            'Current Scenario Image': 'mg_vehicle_image',
            'Competitor Image': 'competitor_vehicle_image'
        }
        df_ideas.rename(columns=column_mapping, inplace=True)
        
        # Merge 'Why?' and 'Impact on Other Systems' into 'way_forward' for better RAG context
        if 'way_forward' in df_ideas.columns and 'impact_other_systems' in df_ideas.columns and 'reason' in df_ideas.columns:
            df_ideas['way_forward'] = df_ideas.apply(
                lambda row: f"Description: {row.get('way_forward', '')} | Why: {row.get('reason', '')} | Impact: {row.get('impact_other_systems', '')}", 
                axis=1
            )
            
        df_ideas = df_ideas[df_ideas['cost_reduction_idea'].notnull()]

        conn = get_db_connection()
        cursor = conn.cursor()

        for _, row in df_ideas.iterrows():
            try:
                cursor.execute("""
                    INSERT INTO ideas (
                        idea_id, cost_reduction_idea, reason, mgi_gross_saving,
                        estimated_cost_savings, saving_value_inr, weight_saving,
                        group_id, status, way_forward, dept, target_date, kd_lc,
                        wtd_avg, est_impl_date, investment_cr, capex, mgi_carline,
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
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    ) ON CONFLICT (idea_id) DO NOTHING
                """, (
                    str(row.get('idea_id', '')), str(row.get('cost_reduction_idea', '')),
                    str(row.get('reason', '')), safe_float_convert(row.get('mgi_gross_saving')),
                    str(row.get('estimated_cost_savings', '')), safe_float_convert(row.get('saving_value_inr')),
                    safe_float_convert(row.get('weight_saving')), str(row.get('group_id', '')),
                    str(row.get('status', '')), str(row.get('way_forward', '')),
                    str(row.get('dept', '')), str(row.get('target_date', '')),
                    str(row.get('kd_lc', '')), safe_float_convert(row.get('wtd_avg')),
                    str(row.get('est_impl_date', '')), safe_float_convert(row.get('investment_cr')),
                    safe_float_convert(row.get('capex') or row.get('investment_cr')),
                    str(row.get('mgi_carline', '')), str(row.get('benchmarking_carline', '')),
                    str(row.get('mg_product_scenario', '')), str(row.get('competitor_product_scenario', '')),
                    str(row.get('purpose_mg_product', '')), str(row.get('purpose_competitor_product', '')),
                    str(row.get('impact_other_systems', '')), str(row.get('client_statement', '')),
                    str(row.get('cae_required', '')), str(row.get('homologation_required', '')),
                    str(row.get('styling_change', '')), str(row.get('part_level_testing', '')),
                    str(row.get('assembly_trials', '')), str(row.get('cad_drawing_update', '')),
                    str(row.get('ecn_required', '')), str(row.get('part_production_trials', '')),
                    str(row.get('vehicle_level_testing', '')), str(row.get('idea_generated_by', '')),
                    str(row.get('new_tool_required', '')), safe_float_convert(row.get('new_tool_cost')),
                    str(row.get('tool_modification_required', '')), safe_float_convert(row.get('tool_modification_cost')),
                    str(row.get('variants', '')), str(row.get('current_status', '')),
                    str(row.get('resp', '')), str(row.get('mix', '')),
                    str(row.get('volume', '')), str(row.get('purchase_proposal', '')),
                    str(row.get('interest', '')), str(row.get('payback_months', '')),
                    str(row.get('mgi_pe_feasibility', '')), str(row.get('homeroom_approval', '')),
                    str(row.get('pp_approval', '')), str(row.get('supplier_feasibility', '')),
                    str(row.get('financial_feasibility', '')), row.get('proposal_image_filename'),
                    row.get('mg_vehicle_image'), row.get('competitor_vehicle_image')
                ))
            except Exception as row_error:
                logger.warning(f"Skipping row due to error: {row_error}")
                continue
        
        conn.commit()
        logger.info(f"Data populated successfully.")
    
    except Exception as e:
        logger.error(f"Error populating database: {e}")
        if conn: conn.rollback()
    finally:
        if conn: conn.close()

if __name__ == '__main__':
    init_db()
    populate_database()