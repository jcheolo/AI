import gdstk
import numpy as np


class layout_agent:
    """
    GDS/OAS 파일에서 특정 레이어, FOV 영역의 vertex 정보를 추출하는 클래스
    모든 좌표는 um 단위를 가정

    사용 예시:
        agent = layout_agent("design.gds")
        vertices = agent.extract_fov_vertices(
            layer=1, datatype=0,
            fov_x=100.0, fov_y=200.0,
            fov_width=10.0, fov_height=10.0
        )
    """

    def __init__(self, gds_path: str):
        """
        Args:
            gds_path: GDS 또는 OAS 파일 경로
        """
        self.gds_path = gds_path
        self.library = self._load_gds(gds_path)
        self.top_cells = self.library.top_level()

        if not self.top_cells:
            raise ValueError("Top level cell이 없습니다.")

        top_cell_names = [c.name for c in self.top_cells]
        print(f"GDS 로드 완료: {gds_path}")
        print(f"Top cell ({len(self.top_cells)}개): {top_cell_names}")

    # ------------------------------------------------------------------ #
    #  내부 메서드
    # ------------------------------------------------------------------ #

    def _load_gds(self, gds_path: str) -> gdstk.Library:
        """GDS 또는 OASIS 파일을 로드하여 Library 객체를 반환"""
        ext = gds_path.lower().rsplit(".", 1)[-1]

        if ext in ("gds", "gds2", "gdsii"):
            library = gdstk.read_gds(gds_path)
        elif ext in ("oas", "oasis"):
            library = gdstk.read_oas(gds_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: .{ext}")

        return library

    def _get_fov_bounds(
        self, fov_x: float, fov_y: float, fov_width: float, fov_height: float
    ) -> tuple:
        """FOV 중심+크기 → (xmin, ymin, xmax, ymax) 반환"""
        hw = fov_width / 2
        hh = fov_height / 2
        return (fov_x - hw, fov_y - hh, fov_x + hw, fov_y + hh)

    def _make_ref_transform(self, ref) -> np.ndarray:
        """Reference의 변환 정보를 3x3 affine matrix로 생성"""
        mat = np.eye(3)

        if ref.x_reflection:
            mat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=float) @ mat

        angle = ref.rotation if ref.rotation else 0.0
        if angle != 0:
            rad = np.radians(angle)
            c, s = np.cos(rad), np.sin(rad)
            rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
            mat = rot @ mat

        mag = ref.magnification if ref.magnification else 1.0
        if mag != 1.0:
            scale = np.array([[mag, 0, 0], [0, mag, 0], [0, 0, 1]], dtype=float)
            mat = scale @ mat

        ox, oy = ref.origin if ref.origin else (0.0, 0.0)
        mat[0, 2] += ox
        mat[1, 2] += oy

        return mat

    def _transform_points(self, pts: np.ndarray, mat: np.ndarray) -> np.ndarray:
        """점 배열(N,2)에 3x3 affine 변환 적용"""
        ones = np.ones((len(pts), 1))
        homogeneous = np.hstack([pts, ones])
        transformed = (mat @ homogeneous.T).T[:, :2]
        return transformed

    def _collect_polygons_in_fov(
        self, cell, layer: int, datatype: int,
        fov_bounds: tuple, parent_mat: np.ndarray,
        stats: dict,
    ) -> list:
        """hierarchy를 재귀 순회하며 FOV와 겹치는 branch의 폴리곤만 수집"""
        fov_xmin, fov_ymin, fov_xmax, fov_ymax = fov_bounds
        collected = []

        # 1) 이 셀의 직접 폴리곤 처리
        for poly in cell.polygons:
            if poly.layer != layer or poly.datatype != datatype:
                continue

            pts = np.array(poly.points, dtype=float)
            pts = self._transform_points(pts, parent_mat)

            p_min = pts.min(axis=0)
            p_max = pts.max(axis=0)
            if p_max[0] < fov_xmin or p_min[0] > fov_xmax or \
               p_max[1] < fov_ymin or p_min[1] > fov_ymax:
                continue

            collected.append(
                gdstk.Polygon(pts, layer=layer, datatype=datatype)
            )
            stats["poly_count"] += 1

        # 2) Reference 순회 — bbox로 가지치기
        for ref in cell.references:
            ref_bbox = ref.bounding_box()
            if ref_bbox is None:
                stats["ref_skip"] += 1
                continue

            corners = np.array([
                [ref_bbox[0][0], ref_bbox[0][1]],
                [ref_bbox[1][0], ref_bbox[0][1]],
                [ref_bbox[1][0], ref_bbox[1][1]],
                [ref_bbox[0][0], ref_bbox[1][1]],
            ])
            world_corners = self._transform_points(corners, parent_mat)
            wc_min = world_corners.min(axis=0)
            wc_max = world_corners.max(axis=0)

            if wc_max[0] < fov_xmin or wc_min[0] > fov_xmax or \
               wc_max[1] < fov_ymin or wc_min[1] > fov_ymax:
                stats["ref_skip"] += 1
                continue

            stats["ref_search"] += 1
            ref_mat = parent_mat @ self._make_ref_transform(ref)

            collected.extend(
                self._collect_polygons_in_fov(
                    ref.cell, layer, datatype, fov_bounds, ref_mat, stats
                )
            )

        return collected

    def _classify_polygon(self, poly, fov_bounds: tuple) -> str:
        """
        폴리곤과 FOV의 관계를 분류
        Returns: 'outside' | 'inside' | 'partial'
        """
        fov_xmin, fov_ymin, fov_xmax, fov_ymax = fov_bounds
        pts = np.array(poly.points)
        p_xmin, p_ymin = pts.min(axis=0)
        p_xmax, p_ymax = pts.max(axis=0)

        if p_xmax < fov_xmin or p_xmin > fov_xmax or \
           p_ymax < fov_ymin or p_ymin > fov_ymax:
            return "outside"

        if p_xmin >= fov_xmin and p_xmax <= fov_xmax and \
           p_ymin >= fov_ymin and p_ymax <= fov_ymax:
            return "inside"

        return "partial"

    # ------------------------------------------------------------------ #
    #  공개 메서드
    # ------------------------------------------------------------------ #

    def get_layer_list(self) -> list:
        """GDS에 존재하는 모든 (layer, datatype) 쌍을 반환"""
        layer_set = set()
        for cell in self.top_cells:
            for poly in cell.get_polygons():
                layer_set.add((poly.layer, poly.datatype))
        return sorted(layer_set)

    def get_polygons_by_layer(self, layer: int, datatype: int = 0) -> list:
        """
        모든 top cell을 기준으로 hierarchy를 풀어
        특정 layer/datatype의 폴리곤을 좌표 변환 적용하여 추출
        """
        polygons = []
        for cell in self.top_cells:
            polys = cell.get_polygons(layer=layer, datatype=datatype)
            polygons.extend(polys)

        print(f"레이어 {layer}/{datatype}: 총 {len(polygons)}개 폴리곤")
        return polygons

    def clip_polygons_to_fov(
        self,
        polygons: list,
        fov_x: float,
        fov_y: float,
        fov_width: float,
        fov_height: float,
    ) -> list:
        """
        FOV 영역으로 폴리곤을 클리핑 (bbox 사전 필터링 적용)

        최적화:
            1) bbox가 FOV와 겹치지 않는 폴리곤 → 즉시 제외
            2) bbox가 FOV 안에 완전히 포함된 폴리곤 → boolean 없이 그대로 유지
            3) 부분 겹침 폴리곤만 → boolean AND 수행
        """
        fov_bounds = self._get_fov_bounds(fov_x, fov_y, fov_width, fov_height)

        fov_rect = gdstk.rectangle(
            (fov_bounds[0], fov_bounds[1]),
            (fov_bounds[2], fov_bounds[3]),
        )

        clipped = []
        n_outside = 0
        n_inside = 0
        n_partial = 0

        for poly in polygons:
            category = self._classify_polygon(poly, fov_bounds)

            if category == "outside":
                n_outside += 1
                continue
            elif category == "inside":
                n_inside += 1
                clipped.append(poly)
            else:  # partial
                n_partial += 1
                result = gdstk.boolean(poly, fov_rect, "and")
                if result:
                    clipped.extend(result)

        print(f"FOV 클리핑: 전체 {len(polygons)}개 → "
              f"outside {n_outside} / inside {n_inside} / "
              f"partial {n_partial} (boolean 수행) → 결과 {len(clipped)}개")
        return clipped

    def get_vertices(self, polygons: list) -> list:
        """
        폴리곤 리스트에서 vertex 좌표(um)를 추출

        Returns:
            list of numpy.ndarray - 각 폴리곤의 vertex 좌표 [(N,2), ...]
        """
        vertices_list = []
        for poly in polygons:
            if isinstance(poly, np.ndarray):
                vertices_list.append(poly)
            else:
                vertices_list.append(np.array(poly.points))
        return vertices_list

    def extract_fov_vertices(
        self,
        layer: int,
        datatype: int,
        fov_x: float,
        fov_y: float,
        fov_width: float,
        fov_height: float,
    ) -> list:
        """
        GDS에서 특정 레이어 + FOV 영역의 vertex 정보를 추출하는 통합 메서드

        최적화 흐름:
            1) hierarchy를 재귀 순회하며 FOV와 겹치는 branch만 탐색
            2) FOV 근처 폴리곤만 수집 (bbox 사전 필터링)
            3) 경계에 걸치는 폴리곤만 boolean AND 클리핑

        Args:
            layer: 레이어 번호
            datatype: 데이터타입 번호
            fov_x, fov_y: FOV 중심 좌표 (um)
            fov_width, fov_height: FOV 크기 (um)

        Returns:
            list of numpy.ndarray - 각 폴리곤의 vertex 좌표 (um) [(N,2), ...]
        """
        fov_bounds = self._get_fov_bounds(fov_x, fov_y, fov_width, fov_height)
        identity = np.eye(3)
        stats = {"ref_skip": 0, "ref_search": 0, "poly_count": 0}

        # 1) hierarchy 재귀 순회 — FOV 근처 폴리곤만 수집
        nearby_polygons = []
        for cell in self.top_cells:
            nearby_polygons.extend(
                self._collect_polygons_in_fov(
                    cell, layer, datatype, fov_bounds, identity, stats
                )
            )

        print(f"레이어 {layer}/{datatype} hierarchy 탐색: "
              f"ref 스킵 {stats['ref_skip']} / 탐색 {stats['ref_search']} → "
              f"FOV 근처 폴리곤 {stats['poly_count']}개 수집")

        if not nearby_polygons:
            print("FOV 내에 폴리곤이 없습니다.")
            return []

        # 2) 정밀 클리핑 (inside/partial 분류 후 boolean)
        clipped = self.clip_polygons_to_fov(
            nearby_polygons, fov_x, fov_y, fov_width, fov_height
        )
        if not clipped:
            print("FOV 내에 폴리곤이 없습니다.")
            return []

        # 3) vertex 추출
        vertices = self.get_vertices(clipped)

        return vertices


# ------------------------------------------------------------------ #
#  사용 예시
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    agent = layout_agent("design.gds")

    print("레이어 목록:", agent.get_layer_list())

    results = agent.extract_fov_vertices(
        layer=1,
        datatype=0,
        fov_x=100.0,
        fov_y=200.0,
        fov_width=10.0,
        fov_height=10.0,
    )

    for i, verts in enumerate(results):
        print(f"\nPolygon {i}: {len(verts)} vertices")
        print(verts)
