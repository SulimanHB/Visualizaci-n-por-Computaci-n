import os
import xml.etree.ElementTree as ET
from pathlib import Path

# === CONFIGURACIÓN ===
# Ruta donde tienes los XML
XML_DIR = Path(r"C:\Users\hsuli\Desktop\annotations")
# Ruta donde guardarás los TXT convertidos
LABELS_DIR = Path(r"C:\Users\hsuli\Desktop\Matriculas\labels")

# Clase (en este caso solo una: "matrícula" → 0)
CLASS_ID = 0
# =====================

os.makedirs(LABELS_DIR, exist_ok=True)

def convert_bbox(size, box):
    """Convierte coordenadas VOC a formato YOLO normalizado"""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x_center *= dw
    w *= dw
    y_center *= dh
    h *= dh
    return (x_center, y_center, w, h)

def convert_annotation(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    label_file = LABELS_DIR / (xml_path.stem + ".txt")

    with open(label_file, "w") as out:
        for obj in root.iter("object"):
            xmlbox = obj.find("bndbox")
            xmin = int(xmlbox.find("xmin").text)
            xmax = int(xmlbox.find("xmax").text)
            ymin = int(xmlbox.find("ymin").text)
            ymax = int(xmlbox.find("ymax").text)
            b = (xmin, xmax, ymin, ymax)
            bb = convert_bbox((w, h), b)
            out.write(f"{CLASS_ID} {' '.join([f'{a:.6f}' for a in bb])}\n")

    print(f"✅ Convertido: {xml_path.name} → {label_file.name}")

def main():
    xml_files = list(XML_DIR.glob("*.xml"))
    print(f"Encontrados {len(xml_files)} archivos XML.")

    for xml_file in xml_files:
        convert_annotation(xml_file)

    print("\n🎯 Conversión completa. Archivos guardados en:", LABELS_DIR)

if __name__ == "__main__":
    main()
