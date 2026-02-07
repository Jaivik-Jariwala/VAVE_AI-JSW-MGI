import os
import time
import requests
import hashlib
import random
import base64  # <--- Added for decoding images
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

# --- CONFIGURATION ---
BASE_DIR = "MG_Hector_Full_Engineering_Dataset"
IMAGES_PER_ITEM = 12  
HEADLESS_MODE = False 

# --- THE MASTER BOM (Same as before) ---
BOM = {
    "1_Engine_Powertrain/Core_Engine": [
        "MG Hector engine block assembly",
        "MG Hector cylinder head gasket",
        "MG Hector piston and rings set",
        "MG Hector connecting rod assembly",
        "MG Hector crankshaft and bearings",
        "MG Hector camshaft intake exhaust",
        "MG Hector engine valves and springs",
        "MG Hector timing chain kit",
        "MG Hector cylinder head valve cover",
        "MG Hector oil pan sump",
        "MG Hector oil pump assembly",
        "MG Hector flywheel assembly",
        "MG Hector engine mounts torque strut"
    ],
    "1_Engine_Powertrain/Air_Fuel": [
        "MG Hector turbocharger assembly wastegate",
        "MG Hector intercooler assembly",
        "MG Hector intercooler hoses",
        "MG Hector air intake manifold",
        "MG Hector throttle body assembly",
        "MG Hector air filter housing box",
        "MG Hector fuel injector rail",
        "MG Hector high pressure fuel pump diesel",
        "MG Hector fuel rail assembly",
        "MG Hector fuel tank cap",
        "MG Hector fuel pump in-tank",
        "MG Hector fuel filter diesel",
        "MG Hector mass air flow sensor MAF"
    ],
    "1_Engine_Powertrain/Exhaust_Cooling": [
        "MG Hector exhaust manifold",
        "MG Hector catalytic converter DPF",
        "MG Hector oxygen sensor lambda",
        "MG Hector exhaust pipe muffler",
        "MG Hector exhaust rubber hanger",
        "MG Hector radiator assembly",
        "MG Hector radiator fan shroud",
        "MG Hector coolant expansion tank",
        "MG Hector water pump gasket",
        "MG Hector thermostat housing",
        "MG Hector radiator hose upper lower",
        "MG Hector heater core hoses"
    ],
    "2_Transmission/Manual_6MT": [
        "MG Hector 6 speed gearbox housing",
        "MG Hector clutch disc pressure plate",
        "MG Hector concentric slave cylinder",
        "MG Hector dual mass flywheel",
        "MG Hector gear shift cable selector",
        "MG Hector gear shift lever assembly"
    ],
    "2_Transmission/Automatic_CVT": [
        "MG Hector CVT transmission assembly",
        "MG Hector transmission oil cooler",
        "MG Hector transmission control module TCM",
        "MG Hector transmission solenoid valve"
    ],
    "2_Transmission/Drive_Components": [
        "MG Hector drive shaft CV axle",
        "MG Hector CV joint inner outer",
        "MG Hector CV boot rubber",
        "MG Hector wheel hub bearing unit"
    ],
    "3_Chassis_Suspension/Front": [
        "MG Hector macpherson strut assembly",
        "MG Hector strut mount bearing",
        "MG Hector lower control arm",
        "MG Hector ball joint suspension",
        "MG Hector stabilizer bar sway bar",
        "MG Hector stabilizer link drop link",
        "MG Hector tie rod end rack end"
    ],
    "3_Chassis_Suspension/Rear": [
        "MG Hector rear torsion beam axle",
        "MG Hector rear coil spring",
        "MG Hector rear shock absorber",
        "MG Hector rear spring seat isolator"
    ],
    "3_Chassis_Suspension/Steering_Brakes": [
        "MG Hector electric steering rack pinion",
        "MG Hector steering column assembly",
        "MG Hector intermediate shaft steering",
        "MG Hector brake master cylinder",
        "MG Hector brake booster vacuum",
        "MG Hector ABS pump module",
        "MG Hector front brake caliper",
        "MG Hector front brake disc rotor",
        "MG Hector rear brake caliper electronic parking",
        "MG Hector rear brake disc rotor",
        "MG Hector electronic parking brake actuator",
        "MG Hector brake pads set",
        "MG Hector brake hose line"
    ],
    "4_Electrical/Lighting": [
        "MG Hector headlamp assembly projector",
        "MG Hector LED DRL floating light",
        "MG Hector front fog lamp",
        "MG Hector tail light connected LED strip",
        "MG Hector high mount stop lamp",
        "MG Hector rear fog lamp reverse light",
        "MG Hector license plate light",
        "MG Hector interior dome light reading"
    ],
    "4_Electrical/Sensors_Modules": [
        "MG Hector engine control unit ECU",
        "MG Hector body control module BCM",
        "MG Hector airbag control module",
        "MG Hector crankshaft position sensor",
        "MG Hector camshaft position sensor",
        "MG Hector ABS wheel speed sensor",
        "MG Hector TPMS sensor valve",
        "MG Hector parking sensor ultrasonic",
        "MG Hector 360 camera grille mirror",
        "MG Hector ADAS radar module bosch"
    ],
    "4_Electrical/Starting_Charging": [
        "MG Hector car battery 12v",
        "MG Hector alternator assembly",
        "MG Hector starter motor",
        "MG Hector fuse box engine cabin",
        "MG Hector 48V hybrid battery lithium"
    ],
    "5_Exterior_Body/Front_Sides": [
        "MG Hector front bumper cover",
        "MG Hector front grille aura hex",
        "MG Hector logo emblem front",
        "MG Hector skid plate front",
        "MG Hector bonnet hood insulation",
        "MG Hector windshield glass front",
        "MG Hector wiper arm blade",
        "MG Hector fender panel",
        "MG Hector door shell assembly",
        "MG Hector ORVM side mirror",
        "MG Hector door handle chrome",
        "MG Hector side cladding trim",
        "MG Hector roof rails",
        "MG Hector pillar trim black tape",
        "MG Hector fuel filler flap"
    ],
    "5_Exterior_Body/Rear": [
        "MG Hector tailgate boot lid powered",
        "MG Hector rear bumper cover",
        "MG Hector rear skid plate diffuser",
        "MG Hector logo emblem rear boot release",
        "MG Hector badging internet inside",
        "MG Hector rear windshield glass",
        "MG Hector rear wiper arm",
        "MG Hector roof spoiler"
    ],
    "6_Interior_Cabin/Dashboard_Seating": [
        "MG Hector dashboard shell soft touch",
        "MG Hector instrument cluster digital",
        "MG Hector 14 inch infotainment screen",
        "MG Hector AC vent assembly",
        "MG Hector glove box assembly",
        "MG Hector steering wheel buttons",
        "MG Hector driver seat power adjust frame",
        "MG Hector rear bench seat split",
        "MG Hector headrest front rear",
        "MG Hector center console armrest",
        "MG Hector door panel card interior",
        "MG Hector roof headliner fabric",
        "MG Hector panoramic sunroof mechanism",
        "MG Hector floor carpet mat",
        "MG Hector parcel tray boot cover"
    ],
    "6_Interior_Cabin/Safety_Misc": [
        "MG Hector seat belt assembly pretensioner",
        "MG Hector airbag driver passenger curtain",
        "MG Hector ISOFIX mount bracket",
        "MG Hector cabin air filter",
        "MG Hector engine oil filter",
        "MG Hector spark plug petrol",
        "MG Hector glow plug diesel",
        "MG Hector serpentine belt",
        "MG Hector jack tool kit",
        "MG Hector spare wheel steel rim",
        "MG Hector warning triangle"
    ]
}

def setup_driver():
    options = Options()
    if HEADLESS_MODE: options.add_argument("--headless")
    options.add_argument("--window-size=1600,900")
    # Using a standard user-agent to look like a real browser
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

def download_images(driver, folder, term):
    save_dir = os.path.join(BASE_DIR, folder, term.replace(" ", "_"))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Updated Query to be more specific
    search_url = f"https://www.google.com/search?q={term} car part&tbm=isch"
    driver.get(search_url)
    
    # Wait for page load
    time.sleep(3) 
    
    # Scroll once to load images
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)
    
    # Get image elements (using the current Google Class for thumbnails)
    thumbnails = driver.find_elements(By.CSS_SELECTOR, "img.Q4LuWd")
    
    # If standard class fails, fallback to generic tag
    if len(thumbnails) == 0:
        print("  -> (Debug) No 'Q4LuWd' class found, trying generic tags...")
        thumbnails = driver.find_elements(By.CSS_SELECTOR, "img")

    count = 0
    print(f"  -> Found {len(thumbnails)} potential images for: {term}")

    for img in thumbnails:
        if count >= IMAGES_PER_ITEM: break
        
        try:
            # We don't click anymore; scraping thumbnails is faster and more reliable 
            # for this specific 'Empty Directory' error.
            
            src = img.get_attribute('src')

            if not src:
                continue

            image_content = None
            
            # CASE 1: Base64 Image (The likely culprit of your previous error)
            if "data:image" in src:
                try:
                    # Split the metadata from the actual data
                    base64_data = src.split(",")[1]
                    image_content = base64.b64decode(base64_data)
                except Exception as e:
                    # print(f"    (Skipping Base64 error: {e})")
                    continue

            # CASE 2: HTTP URL
            elif src.startswith("http"):
                try:
                    response = requests.get(src, timeout=5)
                    if response.status_code == 200:
                        image_content = response.content
                except Exception as e:
                    # print(f"    (Skipping HTTP error: {e})")
                    continue
            
            # Save the image if we got content
            if image_content:
                # Generate unique name
                file_hash = hashlib.md5(src.encode('utf-8') if src else str(time.time()).encode('utf-8')).hexdigest()
                file_path = os.path.join(save_dir, f"{file_hash}.jpg")
                
                with open(file_path, 'wb') as f:
                    f.write(image_content)
                
                count += 1
                if count % 5 == 0: # Print status every 5 images
                    print(f"    -> Saved {count}/{IMAGES_PER_ITEM}")

        except Exception as e:
            # Catch unexpected errors to keep the loop alive
            print(f"    !!! CRITICAL ERROR on image: {e}")
            continue

    if count == 0:
        print(f"    ⚠️ WARNING: Saved 0 images for {term}. Check internet/Google bans.")

def main():
    driver = setup_driver()
    print("--- STARTING MASSIVE BOM COLLECTION (V2 - Base64 Fix) ---")
    
    try:
        for category, items in BOM.items():
            print(f"\n📂 Processing Group: {category}")
            for item in items:
                download_images(driver, category, item)
                # Random sleep to avoid bot detection
                time.sleep(random.uniform(1.5, 3.0))
    finally:
        driver.quit()
        print("\n✅ Collection Complete. Run Step 2 (Analyst) next.")

if __name__ == "__main__":
    main()