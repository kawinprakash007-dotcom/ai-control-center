from typing import List, Dict, Any, Optional

from core.interfaces.perception_interface import VisualPerceptionProvider
from core.models.computer import ComputerObservation
from core.models.perception import (
    VisualElement,
    BoundingBox,
    ElementType,
    PerceptionSource,
)


ROLE_MAP: Dict[str, ElementType] = {
    "button": ElementType.BUTTON,
    "pushbutton": ElementType.BUTTON,
    "edit": ElementType.INPUT,
    "textbox": ElementType.INPUT,
    "link": ElementType.LINK,
    "hyperlink": ElementType.LINK,
    "checkbox": ElementType.CHECKBOX,
    "check_box": ElementType.CHECKBOX,
    "radiobutton": ElementType.RADIO,
    "radio_button": ElementType.RADIO,
    "image": ElementType.IMAGE,
    "menu": ElementType.MENU,
    "menuitem": ElementType.MENU,
    "window": ElementType.WINDOW,
    "pane": ElementType.PANEL,
    "panel": ElementType.PANEL,
    "text": ElementType.TEXT,
    "label": ElementType.TEXT,
}


class AccessibilityPerceptionProvider(VisualPerceptionProvider):
    """
    Adapter extracting UI elements from desktop accessibility and UI tree metadata.
    Provides semantic roles, accessible names, and exact bounding coordinates.
    """
    source = PerceptionSource.ACCESSIBILITY

    def is_available(self) -> bool:
        return True

    def perceive(self, observation: ComputerObservation) -> List[VisualElement]:
        raw_tree = observation.ui_tree_metadata
        if not raw_tree:
            return []

        elements: List[VisualElement] = []
        nodes = raw_tree.get("elements") or raw_tree.get("nodes") or []
        if isinstance(raw_tree, list):
            nodes = raw_tree

        screen = observation.screen_dimensions

        for idx, node in enumerate(nodes):
            if not isinstance(node, dict):
                continue

            role_str = str(node.get("role", "unknown")).strip().lower()
            el_type = ROLE_MAP.get(role_str, ElementType.UNKNOWN)
            label = node.get("name") or node.get("label") or node.get("text")
            bounds = node.get("bounds") or node.get("bounding_box") or node.get("rect")

            if not bounds or len(bounds) != 4:
                continue

            try:
                x, y, w, h = int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3])
                if not screen.contains(x, y):
                    continue
                w = min(w, screen.width - x)
                h = min(h, screen.height - y)

                box = BoundingBox(x=x, y=y, width=w, height=h)
                el = VisualElement(
                    element_id=node.get("id") or f"acc_el_{idx}_{x}_{y}",
                    element_type=el_type,
                    bounding_box=box,
                    text=str(label).strip() if label else None,
                    confidence=1.0,  # Accessibility tree provides authoritative semantic data
                    source=self.source,
                    properties={
                        "role": role_str,
                        "is_enabled": node.get("is_enabled", True),
                        "is_visible": node.get("is_visible", True),
                    },
                )
                elements.append(el)
            except Exception:
                continue

        return elements
