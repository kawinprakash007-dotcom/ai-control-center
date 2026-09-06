from typing import List, Dict, Any, Tuple
from core.models.perception import VisualElement, BoundingBox, ElementType, PerceptionSource


def deduplicate_elements(
    elements: List[VisualElement],
    iou_threshold: float = 0.4,
) -> List[VisualElement]:
    """
    Deterministically deduplicate and fuse multi-source visual elements.
    When elements from different sources (e.g. OCR and Accessibility) overlap,
    they are unified into a single element with higher confidence and preserved provenance.
    """
    if not elements:
        return []

    # Sort deterministically by area descending, then position
    sorted_elements = sorted(
        elements,
        key=lambda el: (-el.bounding_box.area, el.bounding_box.y, el.bounding_box.x, el.element_id),
    )

    merged: List[VisualElement] = []
    used = set()

    for i, el1 in enumerate(sorted_elements):
        if i in used:
            continue

        matched_indices = [i]
        sources = {el1.source.value}
        best_text = el1.text
        best_type = el1.element_type
        max_conf = el1.confidence
        props: Dict[str, Any] = dict(el1.properties)

        for j in range(i + 1, len(sorted_elements)):
            if j in used:
                continue
            el2 = sorted_elements[j]

            # Check geometric overlap
            iou = el1.bounding_box.iou(el2.bounding_box)
            text_match = False
            if el1.text and el2.text:
                t1 = el1.text.strip().lower()
                t2 = el2.text.strip().lower()
                if t1 == t2 or t1 in t2 or t2 in t1:
                    text_match = True

            # If strong IOU or moderate IOU with matching text
            if iou >= iou_threshold or (iou >= 0.15 and text_match):
                matched_indices.append(j)
                used.add(j)
                sources.add(el2.source.value)
                max_conf = max(max_conf, el2.confidence)

                # Prioritize meaningful text
                if not best_text and el2.text:
                    best_text = el2.text
                elif el2.text and len(el2.text) > len(best_text or ""):
                    best_text = el2.text

                # Prioritize specific type over UNKNOWN or TEXT
                if best_type in (ElementType.UNKNOWN, ElementType.TEXT) and el2.element_type not in (
                    ElementType.UNKNOWN,
                    ElementType.TEXT,
                ):
                    best_type = el2.element_type

                props.update(el2.properties)

        used.add(i)

        # Multi-source agreement boosts confidence deterministically
        boosted_conf = max_conf
        if len(sources) > 1:
            boosted_conf = min(1.0, max_conf + 0.15)

        props["sources"] = sorted(list(sources))
        props["fused_from_count"] = len(matched_indices)

        primary_source = el1.source
        if PerceptionSource.ACCESSIBILITY.value in sources:
            primary_source = PerceptionSource.ACCESSIBILITY
        elif PerceptionSource.OCR.value in sources:
            primary_source = PerceptionSource.OCR

        fused_el = VisualElement(
            element_id=el1.element_id,
            element_type=best_type,
            bounding_box=el1.bounding_box,
            text=best_text,
            confidence=round(boosted_conf, 3),
            source=primary_source,
            properties=props,
        )
        merged.append(fused_el)

    return merged
