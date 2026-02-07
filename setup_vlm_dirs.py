"""
Setup script for VLM Engine directory structure.
Creates necessary directories and moves images to static folder.
Generates placeholder images for missing scenarios.
"""
import os
import shutil
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL/Pillow not available. Placeholder images will not be generated.")

# Get base directory (same as app.py)
BASE_DIR = Path(__file__).parent.resolve()

def create_directories():
    """Create all necessary directories for VLM."""
    directories = [
        BASE_DIR / "static" / "images",
        BASE_DIR / "static" / "images" / "mg",
        BASE_DIR / "static" / "images" / "proposal",
        BASE_DIR / "static" / "images" / "competitor",
        BASE_DIR / "static" / "generated",
        BASE_DIR / "static" / "defaults"
    ]
    
    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"[OK] Created/verified directory: {dir_path}")

def move_images_to_static():
    """Move existing images from root/images to static/images."""
    source_dirs = {
        BASE_DIR / "images" / "mg": BASE_DIR / "static" / "images" / "mg",
        BASE_DIR / "images" / "proposal": BASE_DIR / "static" / "images" / "proposal",
        BASE_DIR / "images" / "competitor": BASE_DIR / "static" / "images" / "competitor"
    }
    
    for source, dest in source_dirs.items():
        if source.exists():
            # Move files, not the directory itself
            for file in source.glob("*"):
                if file.is_file():
                    dest_file = dest / file.name
                    if not dest_file.exists():
                        shutil.move(str(file), str(dest_file))
                        print(f"[OK] Moved: {file.name} -> {dest_file}")
                    else:
                        print(f"[SKIP] Skipped (exists): {file.name}")
            print(f"[OK] Processed directory: {source}")
        else:
            print(f"[SKIP] Source directory not found: {source}")

def create_placeholder_image(name, color, text, size=(400, 300)):
    """Create a colored placeholder image with text."""
    img = Image.new('RGB', size, color=color)
    draw = ImageDraw.Draw(img)
    
    # Try to use a nice font, fallback to default
    try:
        # Try to use a system font
        font_size = 24
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            font = ImageFont.load_default()
    
    # Calculate text position (centered)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    position = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)
    
    # Draw text with white color and black outline for visibility
    draw.text(position, text, fill='white', font=font, stroke_width=2, stroke_fill='black')
    
    return img

def generate_placeholder_images():
    """Generate placeholder images for different scenarios."""
    if not PIL_AVAILABLE:
        print("[SKIP] Skipping placeholder generation (PIL not available)")
        return
    
    placeholders = {
        "current_placeholder.jpg": ("#4A90E2", "Current Scenario"),
        "proposal_placeholder.jpg": ("#50C878", "Proposal Scenario"),
        "ai_placeholder.jpg": ("#9B59B6", "AI Generated"),
        "web_placeholder.jpg": ("#E67E22", "Web Sourced")
    }
    
    defaults_dir = BASE_DIR / "static" / "defaults"
    
    for filename, (color, text) in placeholders.items():
        filepath = defaults_dir / filename
        if not filepath.exists():
            img = create_placeholder_image(filename, color, text)
            img.save(filepath, "JPEG", quality=85)
            print(f"[OK] Generated placeholder: {filename}")
        else:
            print(f"[SKIP] Placeholder already exists: {filename}")

def main():
    """Main setup function."""
    print("=" * 60)
    print("VLM Engine Directory Setup")
    print("=" * 60)
    
    print("\n1. Creating directories...")
    create_directories()
    
    print("\n2. Moving images to static folder...")
    move_images_to_static()
    
    print("\n3. Generating placeholder images...")
    generate_placeholder_images()
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print(f"\nDirectory structure:")
    print(f"  static/images/mg/")
    print(f"  static/images/proposal/")
    print(f"  static/images/competitor/")
    print(f"  static/generated/")
    print(f"  static/defaults/")

if __name__ == "__main__":
    main()

