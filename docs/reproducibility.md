# Reproducibility notes

## Included

- Brier Score와 기준 점수 계산
- 시즌 순서 기반 expanding-window split
- 현재 행에서만 계산하는 파생 특성 예시
- 과거 데이터로 적합하는 smoothed rate lookup
- v251 앙상블 가중치와 추론 단계
- residual overlay와 logit sharpening 구현
- artifact 이름, SHA-256, 점수의 구분

## Excluded

- 대회 원본 데이터와 전처리 캐시
- CatBoost, LightGBM, PyTorch 모델 가중치
- 선수별 lookup, 예측 배열, 제출 CSV와 제출 ZIP
- 서버 및 개인 PC의 절대경로
- 점수 차이만 확인된 중복 실험 파일

`.gitignore`는 위 파일 형식을 기본적으로 차단합니다.

## Expected workflow

1. 공개되지 않은 원본 데이터를 별도 `data/` 폴더에 둡니다.
2. `01_problem_and_eda.ipynb`에서 스키마와 시즌별 타깃 비율을 확인합니다.
3. `02_time_series_validation.ipynb`에서 과거 시즌만 학습에 사용합니다.
4. 각 모델의 검증 확률과 테스트 확률을 별도로 생성합니다.
5. `03_v251_reference_pipeline.ipynb`의 인터페이스에 모델 확률과 배포 계수가 반영된 잔차 배열을 전달합니다.
6. Brier Score와 시즌별 안정성을 확인한 뒤에만 제출 확률을 생성합니다.

## Leakage checklist

- 검증 시즌의 타깃은 집계 특성 적합에 사용하지 않습니다.
- 같은 경기 또는 같은 투구 이후의 정보가 현재 투구 특성에 포함되지 않게 정렬합니다.
- 전체 데이터 평균 대신 학습 기간에서 계산한 fallback rate를 사용합니다.
- 확률 보정 계수도 해당 fold의 validation 예측으로 추정합니다.
- 최종 테스트 행을 사용해 특성 선택이나 계수를 최적화하지 않습니다.

## Scope of reproducibility

이 공개 저장소는 **알고리즘과 추론 순서를 재현**합니다. 비공개 데이터와 가중치가 없으므로 v251 제출 확률을 byte-for-byte로 다시 만들지는 않습니다. 또한 v251은 final leaderboard score 1159.52968의 artifact라고 표시하지 않습니다.
