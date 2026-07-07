class BlockBuilder:
    def __init__(self):
        pass

    def build(self, layout_result, ocr_results):
        """
        Build structured text blocks from PP-Structure layout and OCR lines.

        PP-Structure sometimes classifies a whole certificate page as one
        figure/table. In that case, field extraction should work on OCR lines
        instead of one giant concatenated block.
        """
        layout_blocks = layout_result.get("blocks", [])

        ocr_by_block = {}
        unassigned_ocr = []
        for result in ocr_results:
            block_id = result.get("block_id")
            if block_id:
                ocr_by_block.setdefault(block_id, []).append(result)
            else:
                unassigned_ocr.append(result)

        structured_blocks = []

        for layout_block in layout_blocks:
            block_id = layout_block["block_id"]
            lines = self._sort_by_position(ocr_by_block.get(block_id, []))
            full_text = " ".join(line["text"] for line in lines)
            avg_conf = self._average_confidence(lines)

            if self._should_split_layout_block(layout_block, lines, full_text):
                for line_index, line in enumerate(lines):
                    structured_blocks.append(
                        {
                            "block_id": f"{block_id}_line_{line_index}",
                            "parent_block_id": block_id,
                            "label": "text",
                            "parent_label": layout_block["label"],
                            "bbox": line["bbox"],
                            "text": line["text"],
                            "text_lines": [line],
                            "reading_order": layout_block.get("reading_order", 0),
                            "confidence": line["confidence"],
                        }
                    )
                continue

            structured_blocks.append(
                {
                    "block_id": block_id,
                    "label": layout_block["label"],
                    "bbox": layout_block["bbox"],
                    "text": full_text,
                    "text_lines": lines,
                    "reading_order": layout_block.get("reading_order", 0),
                    "confidence": avg_conf,
                }
            )

        max_order = max([b.get("reading_order", 0) for b in layout_blocks]) if layout_blocks else 0
        for index, result in enumerate(self._sort_by_position(unassigned_ocr)):
            structured_blocks.append(
                {
                    "block_id": f"unassigned_block_{index}",
                    "label": "text",
                    "bbox": result["bbox"],
                    "text": result["text"],
                    "text_lines": [result],
                    "reading_order": max_order + index + 1,
                    "confidence": result["confidence"],
                }
            )

        structured_blocks = self._sort_by_position(structured_blocks)
        for reading_order, block in enumerate(structured_blocks):
            block["reading_order"] = reading_order

        return structured_blocks

    def _sort_by_position(self, items):
        return sorted(items, key=lambda item: (item["bbox"][1], item["bbox"][0]))

    def _average_confidence(self, lines):
        confidences = [line["confidence"] for line in lines]
        return sum(confidences) / len(confidences) if confidences else 0.0

    def _should_split_layout_block(self, layout_block, lines, full_text):
        label = layout_block.get("label", "").lower()
        if label in {"figure", "table"} and lines:
            return True
        if len(lines) >= 8:
            return True
        if len(full_text) >= 500:
            return True
        return False
