import os

# Ruta a tu carpeta raíz de datos
base_path = r"C:\Users\hsuli\Desktop\VC\Visualizaci-n-por-Computaci-n\VC\P4\data"

for split in ["train", "val", "test"]:
    labels_dir = os.path.join(base_path, split, "labels")
    if not os.path.exists(labels_dir):
        continue

    for file in os.listdir(labels_dir):
        if file.endswith(".txt"):
            file_path = os.path.join(labels_dir, file)
            new_lines = []
            with open(file_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    # Forzar clase a 0
                    parts[0] = "0"
                    new_lines.append(" ".join(parts) + "\n")

            with open(file_path, "w") as f:
                f.writelines(new_lines)

print("✅ Todas las etiquetas han sido corregidas (clase = 0).")
