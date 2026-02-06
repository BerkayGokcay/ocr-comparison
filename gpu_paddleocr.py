import argparse
import json
from pathlib import Path

import cv2
from paddleocr import PaddleOCR

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(input_path: Path):
    if input_path.is_file():
        if input_path.suffix.lower() in IMAGE_EXTS:
            yield input_path
        else:
            raise ValueError(f"Unsupported ext: {input_path.suffix}")
    elif input_path.is_dir():
        for p in sorted(input_path.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                yield p
    else:
        raise FileNotFoundError(str(input_path))


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def ocr_one(ocr: PaddleOCR, img_path: Path):
    """
    result example:
    result = [
      [
        [ [x1,y1],[x2,y2],[x3,y3],[x4,y4] ], (text, confidence)
      ],
      ...
    ]
    """
    result = ocr.ocr(str(img_path), cls=True)
    lines = []
    if not result or result[0] is None:
        return lines

    for item in result[0]:
        box = item[0]  # 4 points
        text = item[1][0]
        conf = float(item[1][1])
        lines.append({"box": box, "text": text, "confidence": conf})
    return lines


def draw_boxes_opencv(image_bgr, lines, min_conf=0.0):
    """
    Draw polygon boxes + (optional) text/conf using OpenCV only.
    """
    img = image_bgr.copy()

    for l in lines:
        if l["confidence"] < min_conf:
            continue

        pts = l["box"]
        # Ensure int points for OpenCV
        pts_int = [(int(p[0]), int(p[1])) for p in pts]
        poly = cv2.polylines(img, [cv2.UMat(cv2.convexHull(cv2.UMat(pts_int)).get()).get()], True, (0, 255, 0), 2)

        # Put text near first point (keep it short)
        x0, y0 = pts_int[0]
        label = l["text"]
        if len(label) > 40:
            label = label[:40] + "…"
        conf_txt = f"{l['confidence']:.2f}"
        cv2.putText(
            img,
            f"{label} ({conf_txt})",
            (x0, max(0, y0 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return img


def visualize(img_path: Path, lines, out_path: Path, min_conf=0.0):
    image = cv2.imread(str(img_path))
    if image is None:
        raise ValueError(f"Cannot read image: {img_path}")

    vis = draw_boxes_opencv(image, lines, min_conf=min_conf)
    cv2.imwrite(str(out_path), vis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Image file or folder")
    ap.add_argument("-o", "--outdir", default="ocr_out", help="Output folder")
    ap.add_argument("--lang", default="tr")
    ap.add_argument("--use_gpu", type=int, default=1)
    ap.add_argument("--no_cls", action="store_true")
    ap.add_argument("--vis", action="store_true")
    ap.add_argument("--min_conf", type=float, default=0.0)
    args = ap.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    preferred_device = "gpu:0"  # veya "cpu"
    ocr = PaddleOCR(
        lang=args.lang,
        device=preferred_device,   # <-- use_gpu yerine bu
        cls=not args.no_cls,
    )

    merged = []
    for img in iter_images(input_path):
        lines = ocr_one(ocr, img)

        # confidence filtresi JSON için de uygulansın isterseniz:
        if args.min_conf > 0:
            lines = [l for l in lines if l["confidence"] >= args.min_conf]

        payload = {"image": str(img), "results": lines, "lang": args.lang, "use_gpu": bool(args.use_gpu)}
        out_json = outdir / f"{img.stem}.json"
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        if args.vis:
            out_vis = outdir / f"{img.stem}_vis{img.suffix.lower()}"
            try:
                visualize(img, lines, out_vis, min_conf=0.0)
            except Exception as e:
                print(f"[WARN] visualize failed for {img.name}: {e}")

        merged.append(payload)
        print(f"[OK] {img.name} -> {out_json.name} ({len(lines)} lines)")

    (outdir / "merged_results.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[DONE] merged_results.json written")


if __name__ == "__main__":
    main()
