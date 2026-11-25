import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

Box = tuple[int, int, int, int]  # (x, y, w, h)


class FaceDetector(ABC):
    """Контракт (потрібен для LSP).

    detect(image) приймає зображення (BGR або сіре) і повертає список
    прямокутників (x, y, w, h); якщо облич немає, повертає порожній список.
    Нащадки не змінюють вхід, вихід і не кидають нових винятків.
    """

    name: str

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Box]: ...


MAX_SIDE = 1000  # великі фото зменшуємо: швидше і менше хибних спрацювань


def _load_cascade(xml_file: str) -> cv2.CascadeClassifier:
    haar_dir = cv2.data.haarcascades  # type: ignore[attr-defined]
    src = Path(haar_dir) / xml_file
    if not src.exists():
        raise FileNotFoundError(
            f"Немає {src}. Встановіть opencv-python версії 4.x: "
            'uv add "opencv-python>=4.10,<5"'
        )
    if not str(src).isascii():
        # OpenCV на Windows не читає шляхи з кирилицею: копіюємо файл у temp
        tmp = Path(tempfile.gettempdir()) / xml_file
        shutil.copyfile(src, tmp)
        src = tmp
    cascade = cv2.CascadeClassifier(str(src))
    if cascade.empty():
        raise FileNotFoundError(f"Не вдалося завантажити каскад {src}")
    return cascade


class HaarDetector(FaceDetector):
    def __init__(self, name: str, xml_file: str) -> None:
        self.name = name
        self._cascade = _load_cascade(xml_file)

    def detect(self, image: np.ndarray) -> list[Box]:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        scale = min(1.0, MAX_SIDE / max(gray.shape[:2]))
        if scale < 1.0:
            gray = cv2.resize(
                gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
        gray = cv2.equalizeHist(gray)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        # координати повертаємо в масштабі оригінального зображення
        return [
            (int(x / scale), int(y / scale), int(w / scale), int(h / scale))
            for x, y, w, h in faces
        ]


class FrontalDefault(HaarDetector):
    def __init__(self) -> None:
        super().__init__("frontal_default", "haarcascade_frontalface_default.xml")


class FrontalAlt(HaarDetector):
    def __init__(self) -> None:
        super().__init__("frontal_alt", "haarcascade_frontalface_alt.xml")


class FrontalAlt2(HaarDetector):
    def __init__(self) -> None:
        super().__init__("frontal_alt2", "haarcascade_frontalface_alt2.xml")


def draw_faces(image: np.ndarray, boxes: list[Box]) -> np.ndarray:
    result = image.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return result


def read_image(path: str | Path) -> np.ndarray | None:
    """imread, що працює і зі шляхами з кирилицею."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: str | Path, image: np.ndarray) -> None:
    ok, buf = cv2.imencode(Path(path).suffix or ".jpg", image)
    if not ok:
        raise OSError(f"Не вдалося закодувати {path}")
    buf.tofile(str(path))
