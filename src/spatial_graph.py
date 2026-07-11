class SpatialGraph:
    def __init__(self, blocks):
        """
        Khởi tạo đồ thị không gian từ các khối.
        Args:
            blocks (list[dict]): Danh sách Structured Blocks
        """
        self.blocks = {b["block_id"]: b for b in blocks}
        self.edges = {b["block_id"]: {"right": [], "below": [], "same_row": []} for b in blocks}
        self._build_graph()

    def _same_row(self, bbox_a, bbox_b, min_overlap=0.4):
        a_y1, a_y2 = bbox_a[1], bbox_a[3]
        b_y1, b_y2 = bbox_b[1], bbox_b[3]
        overlap = max(0, min(a_y2, b_y2) - max(a_y1, b_y1))
        height_a = a_y2 - a_y1
        height_b = b_y2 - b_y1
        if height_a == 0 or height_b == 0:
            return False
        return overlap / min(height_a, height_b) >= min_overlap

    def _get_distance(self, bbox_a, bbox_b):
        import math
        cx_a = (bbox_a[0] + bbox_a[2]) / 2.0
        cy_a = (bbox_a[1] + bbox_a[3]) / 2.0
        cx_b = (bbox_b[0] + bbox_b[2]) / 2.0
        cy_b = (bbox_b[1] + bbox_b[3]) / 2.0
        return math.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)

    def _build_graph(self):
        block_list = list(self.blocks.values())
        for i, a in enumerate(block_list):
            for j, b in enumerate(block_list):
                if i == j:
                    continue
                
                dist = self._get_distance(a["bbox"], b["bbox"])
                a_x1, a_y1, a_x2, a_y2 = a["bbox"]
                b_x1, b_y1, b_x2, b_y2 = b["bbox"]
                b_cx = (b_x1 + b_x2) / 2.0
                b_cy = (b_y1 + b_y2) / 2.0

                is_same_row = self._same_row(a["bbox"], b["bbox"])
                
                if is_same_row:
                    self.edges[a["block_id"]]["same_row"].append((b["block_id"], dist))
                    if b_cx > a_x2 - 10:
                        self.edges[a["block_id"]]["right"].append((b["block_id"], dist))
                
                if b_cy > a_y2 - 10:
                    # Calculate a specific distance for 'below'
                    dy = max(0, b_y1 - a_y2)
                    dx = abs(a_x1 - b_x1) # Compare left alignment
                    # To prevent skipping lines, dy MUST be the dominant factor.
                    below_dist = dy + dx * 0.1
                    self.edges[a["block_id"]]["below"].append((b["block_id"], below_dist))

        # Sort edges by distance
        for b_id in self.edges:
            for rel in self.edges[b_id]:
                self.edges[b_id][rel].sort(key=lambda x: x[1])

    def get_neighbors(self, block_id, direction, max_distance=500):
        if block_id not in self.edges:
            return []
        if direction == "right_of":
            direction = "right"
        neighbors = self.edges[block_id].get(direction, [])
        return [self.blocks[n_id] for n_id, dist in neighbors if dist <= max_distance]
