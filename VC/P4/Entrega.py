import os
import cv2
import csv
import math
import argparse
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ========== OCR backends ==========

def clean_plate_text(s: str) -> str:
    if s is None:
        return ""
    return "".join([c for c in s if c.isalnum()]).upper()


class EasyOCRBackend:
    """OCR principal basado en EasyOCR (ligero y fiable)."""
    def __init__(self):
        import easyocr
        self.reader = easyocr.Reader(["en"], gpu=False)

    def infer(self, crop):
        out = self.reader.readtext(crop, detail=1, paragraph=False)
        if not out:
            return "", 0.0
        out.sort(key=lambda x: x[2], reverse=True)
        txt = clean_plate_text(out[0][1])
        conf = float(out[0][2])
        return txt, conf


class SmolVLMBackend:
    """OCR alternativo basado en visión-texto (TrOCR, CPU compatible)."""
    def __init__(self):
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        import torch

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = "microsoft/trocr-base-printed"

        print(f"[INFO] Cargando modelo TrOCR ({self.model_id}) en {self.device}...")
        self.processor = TrOCRProcessor.from_pretrained(self.model_id)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_id).to(self.device)
        print("[INFO] TrOCR listo ✅")

    def infer(self, crop):
        from PIL import Image
        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img)

        pixel_values = self.processor(pil, return_tensors="pt").pixel_values.to(self.device)
        with self.torch.no_grad():
            generated_ids = self.model.generate(pixel_values)
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return clean_plate_text(text), 0.0


def build_ocr(name: str):
    name = (name or "easy").lower()
    if name == "easy":
        return EasyOCRBackend()
    if name in ["smolvlm", "trocr"]:
        return SmolVLMBackend()
    raise ValueError(f"OCR desconocido: {name}")


# ===== CONFIG =====
COCO_CLASS_FILTER = [0, 1, 2, 3, 5, 7]
COCO_CLASS_NAMES = {
    0: "persona",
    1: "bicicleta",
    2: "coche",
    3: "moto",
    5: "autobús",
    7: "camión",
}
PLATE_CLASS_NAME = "matricula"
# ==================


def draw_box(img, box, color, label=None):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, required=True, help="Ruta al vídeo o imagen")
    ap.add_argument("--out_dir", type=str, default="salida", help="Directorio de salida")
    ap.add_argument("--device", type=str, default="cpu", help="cpu o 0 (si tu CUDA funcionara)")
    ap.add_argument("--conf_det", type=float, default=0.25, help="conf det general")
    ap.add_argument("--conf_plate", type=float, default=0.25, help="conf detector matrículas")
    ap.add_argument("--ocr", type=str, default="easy", help="easy | smolvlm")
    ap.add_argument("--show", action="store_true", help="Mostrar ventana con preview (si tu OpenCV lo soporta)")
    args = ap.parse_args()

    video_path = Path(args.video)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # === MODELOS ===
    model_det = YOLO("yolo11n.pt")
    model_plate = YOLO(r"runs/detect/train6/weights/best.pt")  # ajusta a tu mejor run
    ocr = build_ocr(args.ocr)

    ext = video_path.suffix.lower()
    is_image = ext in [".jpg", ".jpeg", ".png"]

    # === Entrada ===
    if is_image:
        frame = cv2.imread(str(video_path))
        if frame is None:
            raise RuntimeError(f"No se pudo abrir la imagen: {video_path}")
        width, height = frame.shape[1], frame.shape[0]
        fps = 1
    else:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el vídeo: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # === Salidas ===
    out_video_path = out_dir / f"{video_path.stem}_result.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    out_csv_path = out_dir / f"{video_path.stem}_result.csv"
    csv_fields = [
        "frame","tipo_objeto","confianza","identificador_tracking",
        "x1","y1","x2","y2",
        "matricula_en_su_caso","confianza_m","mx1","my1","mx2","my2","texto_matricula"
    ]
    csv_f = open(out_csv_path, "w", newline="", encoding="utf-8")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(csv_fields)

    unique_ids_by_class = {name: set() for name in COCO_CLASS_NAMES.values()}
    total_matriculas = 0

    # === Tracking/detección ===
    track_source = frame if is_image else str(video_path)
    track_generator = model_det.track(
        source=track_source,
        device=args.device,
        conf=args.conf_det,
        classes=COCO_CLASS_FILTER,
        tracker="botsort.yaml",
        persist=True,
        stream=True,
        verbose=False
    )

    frame_index = -1
    for result in track_generator:
        frame_index += 1
        frame = result.orig_img.copy()

        # Detecciones COCO con IDs
        det_boxes, det_confs, det_cls, det_ids = [], [], [], []
        if result.boxes is not None and len(result.boxes) > 0:
            b = result.boxes
            det_boxes = b.xyxy.cpu().numpy()
            det_confs = b.conf.cpu().numpy() if b.conf is not None else np.zeros((len(det_boxes),))
            det_cls = b.cls.cpu().numpy().astype(int) if b.cls is not None else np.zeros((len(det_boxes),), dtype=int)
            det_ids = b.id.cpu().numpy().astype(int) if b.id is not None else np.full((len(det_boxes),), -1, dtype=int)

        # Detección matrículas
        plate_pred = model_plate.predict(frame, device=args.device, conf=args.conf_plate, verbose=False)[0]

        plate_infos = []
        if plate_pred.boxes is not None and len(plate_pred.boxes) > 0:
            pxys = plate_pred.boxes.xyxy.cpu().numpy()
            pconfs = plate_pred.boxes.conf.cpu().numpy() if plate_pred.boxes.conf is not None else np.zeros((len(pxys),))
            for pbox, pconf in zip(pxys, pconfs):
                x1, y1, x2, y2 = [int(v) for v in pbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width - 1, x2), min(height - 1, y2)
                crop = frame[y1:y2, x1:x2]
                txt, tconf = "", 0.0
                if crop.size > 0:
                    try:
                        t0 = time.time()
                        txt, tconf = ocr.infer(crop)
                        _ = time.time() - t0  # tiempo de inferencia (opcional)
                    except Exception:
                        txt, tconf = "", 0.0
                plate_infos.append({"box": (x1, y1, x2, y2), "conf": float(pconf), "text": txt, "text_conf": tconf})

        matched_plates = {}
        for p in plate_infos:
            px1, py1, px2, py2 = p["box"]
            pcenter = ((px1 + px2) / 2, (py1 + py2) / 2)
            best_id = None
            for i in range(len(det_boxes)):
                box = det_boxes[i]
                tid = int(det_ids[i])
                if tid < 0:
                    continue
                if box[0] <= pcenter[0] <= box[2] and box[1] <= pcenter[1] <= box[3]:
                    best_id = tid
                    break
            if best_id is not None:
                matched_plates[best_id] = p

        # === Evitar duplicados ===
        if "seen_plates" not in locals():
            seen_plates = {}
        for car_id, p in matched_plates.items():
            if car_id not in seen_plates:
                seen_plates[car_id] = p["text"]
                total_matriculas += 1

        # Dibujar objetos COCO
        for i in range(len(det_boxes)):
            box = det_boxes[i]
            conf = float(det_confs[i])
            cid = int(det_cls[i])
            tid = int(det_ids[i])
            name = COCO_CLASS_NAMES.get(cid, f"cls{cid}")
            label = f"{name} {tid if tid>=0 else '-'} {conf:.2f}"
            draw_box(frame, box, (0, 200, 0), label)
            if tid >= 0:
                unique_ids_by_class.setdefault(name, set()).add(tid)

        # Dibujar matrículas + CSV
        for p in plate_infos:
            pbox, pconf, ptext = p["box"], p["conf"], p["text"]
            draw_box(frame, pbox, (0, 0, 255), f"{PLATE_CLASS_NAME} {pconf:.2f} {ptext}")
            csv_w.writerow([
                frame_index, PLATE_CLASS_NAME, f"{pconf:.4f}", -1,
                "", "", "", "",
                PLATE_CLASS_NAME, f"{pconf:.4f}",
                *map(int, pbox), ptext
            ])

        # Overlays de conteo
        y0 = 30
        for cname, idset in unique_ids_by_class.items():
            cv2.putText(frame, f"{cname}: {len(idset)}", (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 200, 255), 2, cv2.LINE_AA)
            y0 += 28
        cv2.putText(frame, f"matriculas_detectadas: {total_matriculas}", (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        writer.write(frame)
        if args.show:
            cv2.imshow("Resultados", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    writer.release()
    csv_f.close()
    try:
        if not is_image:
            cap.release()
        cv2.destroyAllWindows()
    except:
        pass

    print("\n=== Conteo (IDs únicos COCO) ===")
    for k, v in unique_ids_by_class.items():
        print(f"{k}: {len(v)}")
    print(f"matriculas_detectadas: {total_matriculas}")
    print(f"\n✅ Resultado guardado en:\n📹 {out_video_path}\n📄 {out_csv_path}\n(OCR usado: {args.ocr})")


if __name__ == "__main__":
    main()
