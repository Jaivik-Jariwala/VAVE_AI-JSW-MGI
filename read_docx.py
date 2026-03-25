import zipfile
import xml.etree.ElementTree as ET
import sys
import os

def extract_text(p):
    if not os.path.exists(p):
        return f"File not found: {p}"
    try:
        with zipfile.ZipFile(p) as d:
            t = ET.XML(d.read('word/document.xml'))
            return '\n'.join(n.text for n in t.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t') if n.text)
    except Exception as e:
        return str(e)

with open('extracted.txt', 'w', encoding='utf-8') as f:
    f.write('--- LEGAL ---\n')
    f.write(extract_text(sys.argv[1]))
    f.write('\n\n--- PATENT ---\n')
    f.write(extract_text(sys.argv[2]))
