# Experiment summary

이 문서는 원본 실험 파일의 중복을 제거하고, 모델 선택에 영향을 준 관찰만 정리한 것입니다. 점수의 평가 위치가 확인되지 않은 경우 서로 직접 비교하지 않습니다.

| 단계 | 관찰 | 결정 |
|---|---|---|
| Target drift | 연도별 평균 제구 성공률이 달라졌고, 2019년 약 0.5647에서 2024년 약 0.4861로 변화했습니다. | 랜덤 분할보다 시즌 순서 기반 검증과 확률 보정을 우선했습니다. |
| Baseline | 초기 XGBoost 계열 기준의 leaderboard 기록은 766.2418이었습니다. | 단일 모델의 분류 성능보다 확률 품질과 다양성을 개선했습니다. |
| Feature ablation | `same_hand`를 추가했을 때 2024 local 기록은 696에서 735로 좋아졌지만 leaderboard는 808.0206에서 788.6046으로 나빠졌습니다. | 한 검증 시즌의 개선만으로 특성을 채택하지 않았습니다. |
| Model family | CatBoost 단일 모델 기록은 881.4927까지 개선되었습니다. | 범주형 상호작용을 잘 처리하는 트리 모델을 핵심 구성원으로 유지했습니다. |
| Trackman mapping | 580개 선수 쌍을 복원해 주 데이터 행의 약 96.9%를 연결했습니다. | 매핑 자체는 보존하되, A/B 실험에서 효과가 미미하거나 음수인 Trackman 특성은 무조건 채택하지 않았습니다. |
| Ensemble | CatBoost, LightGBM, MLP, FT-Transformer의 예측을 결합했습니다. | 서로 다른 모델의 오차를 평균화하고 후처리 잔차를 순차 적용했습니다. |
| Calibration | 시즌별 base rate와 Brier Score를 함께 점검했습니다. | 마지막에 작은 logit sharpening만 적용해 과도한 보정을 피했습니다. |
| Verified reference | v251 artifact의 SHA-256과 실행 설정을 확인했고 기록 점수는 1152.3953935303입니다. | 공개 reference implementation의 기준으로 채택했습니다. |

## What did not transfer reliably

- 특정 단일 시즌에서만 좋아진 파생 특성
- Trackman 원천값을 직접 추가한 일부 실험
- leaderboard 변화만 있고 artifact 계보를 확인할 수 없는 후반 제출물

이 실패 사례는 삭제하지 않고 의사결정 근거로 요약했습니다. 공개 폴더에는 동일한 변형을 반복한 임시 스크립트와 제출 파일을 복사하지 않았습니다.
