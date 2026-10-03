# Model lineage and score boundary

## Verified v251 reference

| Field | Value |
|---|---|
| Artifact | `ours_v251_sharpen_submit.zip` |
| SHA-256 | `f28306e05af625f75fc5e111de1ea2414c8760a5803892a4eb3984cddedff4af` |
| Recorded score | `1152.3953935303` |
| Public role | Reference implementation 기준 |
| Final competition artifact | No |

v251은 기반 모델의 가중 확률 평균, 여러 historical residual overlay, 마지막 logit sharpening으로 구성됩니다. 원본 ZIP은 공개 폴더에 포함하지 않으며, 모델 가중치와 대용량 lookup도 제외합니다.

## Final competition result

| Field | Value |
|---|---:|
| Rank | 67 / 1,906 |
| Percentile | Top 약 3.5% |
| Final leaderboard score | 1159.52968 |

최종 점수는 대회 종료 시점의 팀 성과입니다. 현재 보존된 증거만으로는 이 점수를 만든 최종 제출 ZIP을 v251과 동일시할 수 없습니다. 따라서 README와 설정 파일 모두에서 다음을 분리해 기록합니다.

```text
verified v251 reference score: 1152.3953935303
final competition score:       1159.52968
same artifact:                 false
```

## Public pipeline order

1. CatBoost, LightGBM, MLP, FT-Transformer 확률의 가중 평균
2. 구조적 상태 잔차
3. forward expert 잔차
4. pitcher-state 잔차
5. 최근 투수 상태 잔차
6. 추가 residual model 보정
7. Bayesian offset 보정
8. pivot 중심 logit sharpening

정확한 순서와 공개 가능한 수치는 [`../configs/v251_reference.json`](../configs/v251_reference.json)에 보존했습니다. 공개 코드의 residual 입력은 이미 배포 시 계수가 반영된 최종 probability delta를 의미합니다. 설정의 `documented_beta`와 `documented_gamma`는 계보 확인용이며 두 번 곱하지 않습니다.
