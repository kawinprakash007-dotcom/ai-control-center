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


class CVPerceptionProvider(VisualPerceptionProvider):
    """
    Deterministic Computer Vision adapter for UI region, container, and button boundary extraction.
    Uses OpenCV contour detection without deep learning models.
    """
    source = PerceptionSource.CV

    def __init__(
        self,
        min_area: int = 150,
        max_area_ratio: float = 0.8,
        min_width: int = 20,
        min_height: int = 15,
    ):
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.min_width = min_width
        self.min_height = min_height
        self._cv2 = None
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import cv2  # noqa
            self._available = True
        except ImportError:
            self._available = False
        return self._available

    def _get_cv2(self):
        if self._cv2 is None and self.is_available():
            try:
                import cv2
                self._cv2 = cv2
            except Exception:
                self._cv2 = None
                self._available = False
        return self._cv2

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
        cv2 = self._get_cv2()
        if not cv2:
            return []

        img = self._load_image(observation)
        if img is None:
            return []

        try:
            import numpy as np
            img_np = np.array(img.convert("RGB"))
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            # Edge detection + morphological dilation to group UI borders
            edges = cv2.Canny(gray, 50, 150)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated = cv2.dilate(edges, kernel, iterations=1)
            contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        except Exception:
            return []

        screen = observation.screen_dimensions
        max_area = int(screen.width * screen.height * self.max_area_ratio)

        elements: List[VisualElement] = []
        for idx, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h

            if area < self.min_area or area > max_area:
                continue
            if w < self.min_width or h < self.min_height:
                continue
            if not screen.contains(x, y):
                continue

            # Classify element type based on aspect ratio and geometry
            aspect = float(w) / float(h)
            el_type = ElementType.PANEL
            if aspect > 1.5 and h <= 60:
                el_type = ElementType.BUTTON if w < 300 else ElementType.INPUT
            elif 0.8 <= aspect <= 1.3 and w <= 80 and h <= 80:
                el_type = ElementType.ICON

            try:
                box = BoundingBox(x=x, y=y, width=w, height=h)
                el = VisualElement(
                    element_id=f"cv_el_{idx}_{x}_{y}",
                    element_type=el_type,
                    bounding_box=box,
                    text=None,
                    confidence=0.75,
                    source=self.source,
                    properties={"aspect_ratio": round(aspect, 2), "area": area},
                )
                elements.append(el)
            except Exception:
                continue

        return elements
