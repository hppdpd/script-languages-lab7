import sys
from pathlib import Path

import cv2

from lab7.detectors import FaceDetector, FrontalAlt2, draw_faces, write_image

OUT = Path("output")


def main(source: str) -> None:
    cap = cv2.VideoCapture(0 if source == "webcam" else source)
    if not cap.isOpened():
        raise SystemExit(f"Не вдалося відкрити джерело: {source}")
    OUT.mkdir(exist_ok=True)

    # LSP: достатньо замінити клас, решта коду не зміниться
    detector: FaceDetector = FrontalAlt2()
    writer: cv2.VideoWriter | None = None
    frames = total_faces = saved = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            boxes = detector.detect(frame)
            total_faces += len(boxes)
            frame = draw_faces(frame, boxes)
            label = f"Faces: {len(boxes)}"
            cv2.putText(
                frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

            if writer is None:
                h, w = frame.shape[:2]
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
                writer = cv2.VideoWriter(
                    str(OUT / "video_out.mp4"), fourcc, fps, (w, h)
                )
            writer.write(frame)

            if boxes and saved < 3 and frames % 30 == 0:  # кадри для звіту
                write_image(OUT / f"video_frame_{saved}.jpg", frame)
                saved += 1

            cv2.imshow("Face detection (q - exit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frames += 1
    finally:  # файл відео коректно закривається, навіть якщо натиснули Ctrl+C
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    print(f"Оброблено кадрів: {frames}, виявлено облич загалом: {total_faces}")
    print(f"Відео збережено: {OUT}/video_out.mp4, кадри: {OUT}/video_frame_*.jpg")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "webcam")
