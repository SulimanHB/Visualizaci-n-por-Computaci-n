import os
import shutil
import random
from pathlib import Path

# === CONFIGURACIÓN ===
# Ruta base donde tienes la carpeta "Matriculas"
RUTA_BASE = Path(r"C:\Users\hsuli\Desktop\Matriculas")

# Rutas origen
IMAGES_DIR = RUTA_BASE / "Images"
LABELS_DIR = RUTA_BASE / "Labels"

# Rutas destino dentro de tu proyecto P4
DEST_ROOT = Path(r"C:\Users\hsuli\Desktop\VC\Visualizaci-n-por-Computaci-n\VC\P4\data")

# Porcentajes de división (deben sumar 1.0)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2
# =====================


def crear_directorio(ruta):
    """Crea un directorio si no existe."""
    os.makedirs(ruta, exist_ok=True)


def limpiar_directorio(ruta):
    """Elimina el contenido previo del directorio (si existe)."""
    if ruta.exists():
        shutil.rmtree(ruta)
    os.makedirs(ruta)


def copiar_archivo(origen, destino):
    """Copia un archivo a destino, creando la carpeta si hace falta."""
    crear_directorio(destino.parent)
    shutil.copy2(origen, destino)


def main():
    # Buscar imágenes válidas
    imagenes = sorted([f for f in IMAGES_DIR.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    pares = []

    for img in imagenes:
        label_name = img.stem + ".txt"
        label_path = LABELS_DIR / label_name
        if label_path.exists():
            pares.append((img, label_path))

    total = len(pares)
    print(f"Total de pares válidos encontrados: {total}")

    if total == 0:
        print("⚠️ No se encontraron pares imagen-label.")
        return

    # Mezclar aleatoriamente
    random.shuffle(pares)

    # Cálculo de divisiones
    n_train = int(total * TRAIN_RATIO)
    n_val = int(total * VAL_RATIO)
    n_test = total - n_train - n_val

    train_pairs = pares[:n_train]
    val_pairs = pares[n_train:n_train + n_val]
    test_pairs = pares[n_train + n_val:]

    # Crear y limpiar rutas destino
    conjuntos = ["train", "val", "test"]
    for c in conjuntos:
        limpiar_directorio(DEST_ROOT / c / "images")
        limpiar_directorio(DEST_ROOT / c / "labels")

    # Copiar archivos
    for img, lbl in train_pairs:
        copiar_archivo(img, DEST_ROOT / "train" / "images" / img.name)
        copiar_archivo(lbl, DEST_ROOT / "train" / "labels" / lbl.name)

    for img, lbl in val_pairs:
        copiar_archivo(img, DEST_ROOT / "val" / "images" / img.name)
        copiar_archivo(lbl, DEST_ROOT / "val" / "labels" / lbl.name)

    for img, lbl in test_pairs:
        copiar_archivo(img, DEST_ROOT / "test" / "images" / img.name)
        copiar_archivo(lbl, DEST_ROOT / "test" / "labels" / lbl.name)

    print("\n✅ División completada:")
    print(f"   Entrenamiento (train): {len(train_pairs)}")
    print(f"   Validación (val):       {len(val_pairs)}")
    print(f"   Prueba (test):          {len(test_pairs)}")
    print("\nCarpetas generadas en:", DEST_ROOT)


if __name__ == "__main__":
    main()
