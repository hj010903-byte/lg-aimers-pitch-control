# Pitch Control Success Probability Prediction

투구 직전까지 확인 가능한 정보로 `control_success`의 확률을 예측한 프로젝트입니다. 이 저장소는 검증 가능한 **v251 제출물을 공개 reference implementation의 기준**으로 삼아, 문제 정의부터 시계열 검증·앙상블·후처리까지의 핵심 흐름을 다시 구성합니다.

원본 데이터, 학습된 모델 가중치, 제출 ZIP은 공개하지 않습니다. 저장소의 코드는 해당 파일 없이도 설계와 검증 절차를 검토할 수 있도록 정리했습니다.

## Problem

각 투구에 대해 다음 값을 예측합니다.

\[
P(\text{control\_success}=1 \mid \text{투구 직전까지의 정보})
\]

제구 성공 여부는 이진값이지만, 제출값은 확률입니다. 따라서 분류 정확도뿐 아니라 확률의 보정 상태가 중요합니다. 이 프로젝트에서 다룬 핵심 과제는 다음과 같습니다.

- 시즌별 제구 성공률 변화에 대응하는 시간 순서 기반 검증
- 투수, 타자, 볼카운트, 주자 상황 사이의 희소한 상호작용 처리
- 미래 정보가 과거 행에 섞이지 않도록 하는 누수 방지
- 서로 다른 모델의 확률을 결합하고 최종 확률을 보정하는 방법

## Data

2019~2024년 KBO 정규시즌의 투구 단위 데이터를 사용했습니다. 주요 정보는 볼카운트, 주자 상황, 선수 식별자와 특성, 과거 투구 이력이며, 2019~2024년 Trackman 투구 특성을 보조 정보로 활용했습니다.

약 150만 행 규모의 원본 데이터는 대회 규정과 용량 문제로 포함하지 않습니다. 공개 코드는 다음 형태의 입력을 가정합니다.

- 타깃: `control_success` (`0` 또는 `1`)
- 시간 구분: `season`
- 투구 직전 상황: `balls_before`, `strikes_before`, 주자 상태 등
- 선수 정보: 투수·타자 식별자와 손잡이 등
- 과거 기록으로만 계산한 집계 및 잔차 특성

## Method

기본 예측은 서로 다른 귀납 편향을 갖는 모델을 결합했습니다.

- CatBoost 계열 2개
- LightGBM GBDT와 DART
- MLP
- FT-Transformer

그 위에 과거 데이터로 만든 구조적 잔차, 투수 상태, 카운트 상태, forward expert, pitcher-state 보정치를 순서대로 더했습니다. 마지막에는 기준점 중심의 작은 logit sharpening을 적용했습니다.

```text
pre-pitch features
      ↓
CatBoost / LightGBM / MLP / FT-Transformer
      ↓
weighted probability ensemble
      ↓
historical residual overlays
      ↓
pivot-centered logit sharpening
      ↓
control_success probability
```

공개 reference의 앙상블 가중치와 후처리 순서는 [`configs/v251_reference.json`](configs/v251_reference.json)에 기록했습니다. 가중치 파일 없이 결합 과정을 확인할 수 있는 구현은 [`src/reference_pipeline.py`](src/reference_pipeline.py)에 있습니다.

## Validation

랜덤 분할 대신 시즌 순서를 보존하는 expanding-window 검증을 사용했습니다.

| 학습 시즌 | 검증 시즌 |
|---|---:|
| 2019~2021 | 2022 |
| 2019~2022 | 2023 |
| 2019~2023 | 2024 |

한 시즌에서만 좋아진 변경보다 여러 미래 시즌에서 일관되게 개선되는 변경을 우선했습니다. 집계 특성은 검증 행보다 앞선 데이터에서만 계산했습니다.

평가의 기본 단위는 Brier Score입니다.

\[
\text{Brier Score}=\frac{1}{n}\sum_{i=1}^{n}(p_i-y_i)^2
\]

여기서 \(p_i\)는 i번째 투구의 제구 성공 예측 확률, \(y_i\)는 실제 정답입니다. 검증 세트의 평균 제구율을 \(r=\operatorname{mean}(y_i)\)라 하면 평균 제구율만 예측하는 기준 모델의 점수는 다음과 같습니다.

\[
\text{Reference Brier Score}=r(1-r)
\]

코드에서는 Brier Skill Score도 함께 계산합니다.

\[
\text{BSS}=1-\frac{\text{Brier Score}}{r(1-r)}
\]

## Result

### Competition result

- **67 / 1,906**
- **Top 약 3.5%**
- **Final leaderboard score: 1159.52968**

### Verified public reference

- Artifact: `ours_v251_sharpen_submit.zip`
- SHA-256: `f28306e05af625f75fc5e111de1ea2414c8760a5803892a4eb3984cddedff4af`
- Recorded v251 score: **1152.3953935303**

**v251의 1152.3953935303은 검증된 중간 reference artifact의 기록이며, final leaderboard score 1159.52968과 동일한 artifact의 결과로 간주하지 않습니다.** 최종 성과는 대회 종료 시점의 팀 결과이고, 공개 구현은 파일과 계보를 확인할 수 있는 v251을 기준으로 합니다.

## My Contribution

- 시즌별 타깃 비율 변화를 확인하고 시간 순서 기반 검증 구조를 설계했습니다.
- CatBoost, LightGBM, MLP, FT-Transformer의 확률 예측을 비교하고 앙상블 파이프라인을 구성했습니다.
- 선수·상황별 과거 집계와 잔차 보정, pitcher-state 및 logit 후처리를 실험했습니다.
- Trackman 선수 매핑과 보조 특성의 효과를 검증했으며, 효과가 불안정한 변경은 최종 기준에서 제외했습니다.
- 제출물의 해시, 설정, 실행 순서를 추적해 재현 가능한 v251 reference를 보존했습니다.

최종 대회 모델은 팀의 실험과 결합 결과이며, 이 저장소는 그중 제가 정리하고 검증할 수 있는 분석 및 파이프라인 범위를 공개합니다.

## Limitations

- 대회 데이터와 모델 가중치가 없으므로 저장소만으로 공식 제출값을 완전히 재생성할 수는 없습니다.
- v251은 공개 가능한 검증 기준이며, 최종 점수 1159.52968을 만든 artifact와 동일하다고 주장하지 않습니다.
- leaderboard 피드백을 참고한 후반 실험은 독립적인 일반화 성능을 과대평가할 수 있습니다.
- Trackman 매핑은 모든 선수·시즌을 완전히 포괄하지 않으며, 일부 특성은 결측 또는 표본 부족의 영향을 받습니다.
- 다른 시즌이나 리그에 적용하려면 타깃 비율과 확률 보정을 다시 추정해야 합니다.

### Repository structure

```text
portfolio/
├── README.md
├── notebooks/
│   ├── 01_problem_and_eda.ipynb
│   ├── 02_time_series_validation.ipynb
│   └── 03_v251_reference_pipeline.ipynb
├── src/
│   ├── features.py
│   ├── metrics.py
│   ├── postprocess.py
│   ├── reference_pipeline.py
│   └── validation.py
├── configs/v251_reference.json
├── docs/
│   ├── experiments.md
│   ├── model_lineage.md
│   └── reproducibility.md
├── requirements.txt
└── .gitignore
```

### Quick start

```bash
python -m venv .venv
pip install -r requirements.txt
jupyter lab
```

노트북은 저장소 루트에서 실행하는 것을 기준으로 하며, 데이터가 없을 때도 문서와 코드 구조를 읽을 수 있도록 작성했습니다.
