import os
import base64
import io
from typing import List, Optional
from PIL import Image

from core.interfaces.perception_interface import VisualPerceptionProvider
from core.models.computer import ComputerObservation
from core.models.perception import (
    VisualElement,
    BoundingBox,
    ElementType,
    PerceptionSource,
)


BUTTON_KEYWORDS = {
    "submit", "ok", "cancel", "next", "back", "login", "sign in", "signup",
    "save", "close", "apply", "continue", "delete", "edit", "search", "send",
}


class OCRPerceptionProvider(VisualPerceptionProvider):
    """
    Adapter extracting text elements and bounding boxes using OCR.
    Model-neutral and fails safely if OCR libraries are not configured.
    """
    source = PerceptionSource.OCR

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self._reader = None
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import easyocr  # noqa
            self._available = True
        except ImportError:
            self._available = False
        return self._available

    def _get_reader(self):
        if self._reader is None and self.is_available():
            try:
                import easyocr
                self._reader = easyocr.Reader(["en"], gpu=self.use_gpu)
            except Exception:
                self._reader = None
                self._available = False
        return self._reader

    def _load_image(self, observation: ComputerObservation) -> Optional[Image.Image]:
        if observation.screenshot_path and os.path.exists(observation.screenshot_path):
            try:
                return Image.open(observation.screenshot_path)
            except Exception:
                pass

        if observation.screenshot_base64:
            try:
                raw_bytes = base64.b64decode(observation.screenshot_base64)
                return Image.open(io.BytesIO(raw_bytes))
            except Exception:
                pass

        return None

    def perceive(self, observation: ComputerObservation) -> List[VisualElement]:
        """Run OCR on the observation image to detect text and bounding boxes."""
        reader = self._get_reader()
        if not reader:
            return []

        img = self._load_image(observation)
        if img is None:
            return []

        try:
            # EasyOCR expects numpy array or file path
            import numpy as np
            img_np = np.array(img.convert("RGB"))
            raw_results = reader.readtext(img_np)
        except Exception:
            return []

        elements: List[VisualElement] = []
        for idx, (bbox_points, text, conf) in enumerate(raw_results):
            if not text or not text.strip():
                continue

            # bbox_points: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            xs = [int(p[0]) for p in bbox_points]
            ys = [int(p[1]) for p in bbox_points]
            min_x, max_x = max(0, min(xs)), max(xs)
            min_y, max_y = max(0, min(ys)), max(ys)
            width = max(1, max_x - min_x)
            height = max(1, max_y - min_y)

            # Ensure bounds do not exceed screen
            screen = observation.screen_dimensions
            if min_x >= screen.width or min_y >= screen.height:
                continue
            width = min(width, screen.width - min_x)
            height = min(height, screen.height - min_y)

            clean_text = text.strip()
            el_type = ElementType.TEXT
            if clean_text.lower() in BUTTON_KEYWORDS or (len(clean_text) <= 15 and width < 250 and height < 60):
                el_type = ElementType.BUTTON

            try:
                box = BoundingBox(x=min_x, y=min_y, width=width, height=height)
                el = VisualElement(
                    element_id=f"ocr_el_{idx}_{min_x}_{min_y}",
                    element_type=el_type,
                    bounding_box=box,
                    text=clean_text,
                    confidence=round(float(conf), 3),
                    source=self.source,
                    properties={"raw_confidence": float(conf)},
                )
                elements.append(el)
            except Exception:
                continue

        return elements
