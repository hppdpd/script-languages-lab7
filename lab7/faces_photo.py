import sys
from pathlib import Path

from lab7.detectors import (
    FaceDetector,
    FrontalAlt,
    FrontalAlt2,
    FrontalDefault,
    draw_faces,
    read_image,
    write_image,
)

OUT = Path("output")


def main(path: str) -> None:
    image = read_image(path)
    if image is None:
        raise SystemExit(f"Не вдалося прочитати зображення: {path}")
    OUT.mkdir(exist_ok=True)
    print(f"Зображення: {path}, розмір {image.shape[1]}x{image.shape[0]}")

    # LSP: усі детектори використовуються однаково
    detectors: list[FaceDetector] = [FrontalDefault(), FrontalAlt2(), FrontalAlt()]
    for det in detectors:
        boxes = det.detect(image)
        print(f"{det.name}: знайдено облич - {len(boxes)}; (x, y, w, h): {boxes}")
        write_image(OUT / f"faces_{det.name}.jpg", draw_faces(image, boxes))
    print(f"Результати збережено в {OUT}/faces_*.jpg")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/group.jpg")
