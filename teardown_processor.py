import os
import pandas as pd
import openpyxl
from PIL import Image as PILImage
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class TeardownProcessor:
    def __init__(self, static_dir="static/teardown"):
        self.static_dir = Path(static_dir)
        self.static_dir.mkdir(parents=True, exist_ok=True)

    def process_file(self, file_path: str, vehicle_name: str):
        """
        Parses the Teardown Excel:
        1. Extracts Floating Images -> Saves to static/teardown/{vehicle_name}/
        2. Reads Data -> Maps Row # to Data + Image
        """
        file_path = str(file_path)
        vehicle_folder = self.static_dir / vehicle_name
        vehicle_folder.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing Teardown File: {vehicle_name}...")

        # 1. LOAD DATA (Skip metadata rows, header is at index 1 i.e. Excel Row 2)
        try:
            df = pd.read_excel(file_path, header=1)
            # Filter rows where 'Part name' is valid
            df = df.dropna(subset=['Part name'])
        except Exception as e:
            logger.error(f"Failed to read excel data: {e}")
            return []

        # 2. LOAD IMAGES (OpenPyXL)
        image_map = {} # Row_Index -> [Image Paths]
        try:
            wb = openpyxl.load_workbook(file_path)
            ws = wb.active
            
            # Iterate through all images in the sheet
            # Note: OpenPyXL images have an 'anchor' which tells us the cell location
            for i, img in enumerate(ws._images):
                try:
                    # anchor.from.row is 0-indexed. 
                    # Dataframe index usually aligns with (Excel Row - Header Offset - 2)
                    # Let's verify mapping: header at row 1 (0-indexed). Data starts row 2.
                    # so img_row 2 maps to df index 0.
                    row_idx = img.anchor._from.row
                    col_idx = img.anchor._from.col
                    
                    # Save Image
                    img_filename = f"{vehicle_name}_row{row_idx}_{i}.png"
                    img_path = vehicle_folder / img_filename
                    
                    # Convert to PIL and Save
                    # OpenPyXL images are already PIL compatible-ish or binary
                    # img.ref is the binary data usually
                    if hasattr(img, 'ref'):
                        # This works for some versions
                         with open(img_path, "wb") as f:
                            f.write(img.ref.read())
                    else:
                        # Fallback for standard OpenPyXL Image object
                        img.image.save(str(img_path))
                        
                    rel_path = f"static/teardown/{vehicle_name}/{img_filename}"
                    
                    # We map based on Excel Row Index
                    if row_idx not in image_map: image_map[row_idx] = []
                    image_map[row_idx].append(rel_path)
                    
                except Exception as e:
                    logger.warning(f"Could not extract image {i}: {e}")

        except Exception as e:
            logger.error(f"Failed to extract images: {e}")

        # 3. MERGE DATA
        parts = []
        # Offset: Header is at Row 1. Data starts at Row 2.
        # OpenPyXL Row 2 = Dataframe Index 0.
        header_row_index = 1 
        
        for idx, row in df.iterrows():
            # Align DF index to OpenPyXL Row Logic
            # Excel Row for this data item = header_row_index + 1 + idx + 1 ??
            # Let's approximate: Header is row 1. Data starts row 2.
            # df.iloc[0] is from Excel Row 2.
            # So excel_row = idx + 2.
            excel_row = idx + 2
            
            # Get images for this row (and maybe neighbor rows if merged?)
            # For strict mapping, use excel_row
            row_images = image_map.get(excel_row, [])
            
            # Also check excel_row - 1 or +1 if alignment is loose? No, stick to strict first.
            if not row_images:
                # Sometimes images are anchored slightly off (e.g. top of cell vs middle)
                # Check fuzzy...
                pass

            part_data = {
                "part_name": str(row.get("Part name", "Unknown")).strip(),
                "part_number": str(row.get("Part No", "")).strip(),
                "material": str(row.get("RM Grades", "N/A")).strip(),
                "weight_g": row.get("Wt.(g)", 0),
                "cost_inr": row.get("Individual Part Cost (IPC)", 0),
                "quantity": row.get("Qty.", 1),
                "images": row_images,
                "vehicle": vehicle_name
            }
            parts.append(part_data)
            
        logger.info(f"Processed {len(parts)} parts for {vehicle_name}")
        return parts

    def compare_vehicles(self, mg_parts, comp_parts):
        """
        Aligns parts by Name and generates comparison context.
        """
        comparisons = []
        
        # Create lookup
        comp_map = {p["part_name"].lower(): p for p in comp_parts}
        
        for mg in mg_parts:
            name = mg["part_name"].lower()
            if not name or name == "nan": continue
            
            # Find best match
            # (Starting with direct match, can add fuzzy later)
            match = comp_map.get(name)
            
            entry = {
                "part_name": mg["part_name"],
                "mg_data": mg,
                "competitor_data": match, # Can be None
                "weight_diff": 0,
                "cost_diff": 0
            }
            
            if match:
                try:
                    entry["weight_diff"] = float(mg["weight_g"]) - float(match["weight_g"])
                    entry["cost_diff"] = float(mg["cost_inr"]) - float(match["cost_inr"])
                except: pass
            
            comparisons.append(entry)
            
        return comparisons
