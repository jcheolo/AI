# layout_agent 개발 요약

## 개발 환경
- Python 3.12, gdstk 0.9.60, klayout 0.30.1, pytorch 2.7

## 목적
GDS/OAS 파일에서 특정 레이어 + FOV 영역의 vertex 정보를 추출하는 클래스 (`layout_agent`)

## 핵심 설계 결정사항

### 단위
- GDS의 user unit과 입출력 좌표 모두 **um으로 가정**, 단위 변환 로직 없음
- 참고: `library.unit`(1 user unit = ? meters), `library.precision`(1 dbu = ? meters)은 존재하지만 코드에서 사용하지 않음
- gdstk는 GDS 읽을 때 이미 user unit(float)으로 변환하여 반환함 (원시 dbu 정수가 아님)

### 파일 포맷
- `_load_gds()`에서 확장자 기반 분기: `.gds` → `gdstk.read_gds()`, `.oas` → `gdstk.read_oas()`
- 반환되는 Library 객체는 동일하므로 이후 코드 공통

### Cell 선택
- `library.top_level()`로 top cell 자동 판별 (다른 셀에 참조되지 않는 셀)
- top cell이 여러 개일 수 있음 → 모든 top cell을 순회
- `get_polygons_by_layer()`는 전체 hierarchy를 flatten하는 범용 메서드로 유지

### FOV 좌표 기준
- (fov_x, fov_y)는 **FOV 중심** 좌표
- fov_width, fov_height로 사각형 영역 정의

## 최적화 구조 (extract_fov_vertices 흐름)

### 1단계: hierarchy 재귀 순회 + 가지치기 (`_collect_polygons_in_fov`)
- 각 Reference의 `bounding_box()`를 FOV와 비교
- FOV와 겹치지 않는 branch → 하위 셀 전체 스킵
- 겹치는 branch만 재귀 진입
- 셀 내 개별 폴리곤도 bbox 체크 후 수집
- affine matrix(3x3)를 누적하며 좌표 변환 (reflection → rotation → magnification → translation)

### 2단계: 정밀 클리핑 (`clip_polygons_to_fov`)
- 수집된 폴리곤을 3분류: outside / inside / partial
- outside → 스킵
- inside → boolean 없이 그대로 유지
- partial → `gdstk.boolean(poly, fov_rect, "and")` 수행 (비용 큰 연산은 여기서만)

### 3단계: vertex 추출 (`get_vertices`)
- `poly.points` → numpy.ndarray (N,2) 변환

## 클래스 구조

```
layout_agent
├── __init__(gds_path)              # GDS/OAS 로드, top cell 확인
│
├── 내부 메서드
│   ├── _load_gds()                 # 확장자 분기 로드
│   ├── _get_fov_bounds()           # 중심+크기 → xmin,ymin,xmax,ymax
│   ├── _make_ref_transform()       # Reference → 3x3 affine matrix
│   ├── _transform_points()         # 점 배열에 affine 적용
│   ├── _collect_polygons_in_fov()  # 재귀 hierarchy 순회 + bbox 가지치기
│   └── _classify_polygon()         # outside / inside / partial 분류
│
├── 공개 메서드
│   ├── get_layer_list()            # 전체 (layer, datatype) 쌍 조회
│   ├── get_polygons_by_layer()     # 특정 레이어 전체 폴리곤 (범용, flatten)
│   ├── clip_polygons_to_fov()      # FOV 클리핑 (3분류 최적화)
│   ├── get_vertices()              # 폴리곤 → numpy vertex 배열
│   └── extract_fov_vertices()      # 통합 메서드 (1→2→3단계)
```

## 논의된 이슈 & 해결

| 이슈 | 결론 |
|------|------|
| .oas 파일 로드 시 에러 | GDS/OAS는 다른 포맷, `read_oas()` 사용 |
| get_polygons 전체 추출 비효율 | hierarchy 재귀 순회 + ref bbox 가지치기로 해결 |
| 단위 변환 필요 여부 | um 가정으로 단순화, 변환 로직 제거 |
| top cell 자동 판별 원리 | 다른 셀에 참조되지 않는 셀 = top cell |
| precision 역할 | 파일 저장 시 최소 해상도 (읽기 전용에서는 무관) |

## 향후 확장 가능 영역
- 결과 시각화 (klayout 또는 matplotlib)
- 복수 FOV 일괄 처리
- GDS 출력 (이때 precision 고려 필요)
- Path/Label 추출 지원
