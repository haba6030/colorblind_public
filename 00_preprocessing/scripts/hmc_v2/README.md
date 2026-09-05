# 서버에서 회수한 `hmc_v2` arm 생성 스크립트 (2026-08-24)

`hmc_v2` 는 4 arm 민감도 분석의 공간축이면서 개인 수준 주장 강등의 근거인데, 그 트리를 만든 스크립트가 저장소에 없고 서버에만 있었다. 회수해 여기에 둔다.

| 파일 | 서버 원본 | 역할 |
|---|---|---|
| `run_hmc_array.sbatch` | `node1:/storage/connectome/haba6030/pilot/hmc_full/` | **정본.** exp1 9명 × 6런 = 54런 단일 보간 HMC |
| `run_hmc_array_exp2_harm.sbatch` | `node1:/storage/connectome/haba6030/pilot/hmc_exp2_harm/` | exp2 통일본 |
| `run_hmc_pilot.sh` | `node1:/storage/connectome/haba6030/pilot/` | sub-08 run-1 파일럿 (같은 체인) |
| `analyze_hmc.sh` | `node1:/storage/connectome/haba6030/pilot/hmc_full/` | QC. `hmc_summary.csv` 와 그림 27장 생성 |
| `run_c010_hmc_v2.sbatch` | `node1:/scratch/.../hmc_reanalysis/` | 진폭 추출 (전처리 아님) |

## 코드 감사 결과

**변환 합성 순서는 옳다.** `convert_xfm -omat M_v.mat -concat b2t.mat mc.mat/MAT_v` 는 FSL 규약상 뒤 인자를 먼저 적용하므로 `볼륨 v → 기준 볼륨 → T1w` 순서가 맞다.

**기준 볼륨 규약도 일치한다.** `REF=$((NV/2))` 하나를 `mcflirt -refvol` 과 `fslroi ... boldref $REF 1` 에 함께 쓰고, 그 `boldref` 로 `tkregister2 --fslregout b2t.mat` 을 만든다. 즉 MCFLIRT 가 정렬한 기준 볼륨과 정합 행렬의 출처 볼륨이 같다.

**정본 arm 과 정합·워프·보간 설정이 동일하다.** `run_method3_header_mi_all_subjects.sbatch:433-451` 도 같은 `tkregister2 --fslregout` → `applywarp --warp=... --premat=... --interp=trilinear` 이며, 유일한 차이가 premat 에 볼륨별 강체가 합성되는지 여부다. 따라서 arm 차이의 원인 귀속이 성립한다.

**보간은 1회다.** 원본 BIDS BOLD 의 볼륨을 `fslroi` 로 뽑아 곧장 `applywarp` 에 넣는다. 폐기된 구버전(`run_method3_hmc_all_subjects.sbatch`)의 `mcflirt -out` → `applywarp` 이중 보간과 다르다.

## 항등 검사 — 통과 (2026-08-24)

기준 볼륨에서는 `MAT_REF` 가 항등이므로 합성 결과가 `b2t` 와 같아야 하고, 따라서 **`hmc_v2` 의 기준 볼륨은 정본과 같아야 한다.** 합성 순서나 기준 볼륨 규약이 틀렸다면 이 검사가 곧바로 깨진다.

| | NV | REF | 기준 볼륨 \|hmc − canon\| | 인접 볼륨 차이 (척도 비교용) | 평균 신호 |
|---|---|---|---|---|---|
| sub-01 run-1 | 288 | 144 | **0.000** (완전 일치) | 13.76 | 254.1 |
| sub-08 run-1 | 288 | 144 | 1.000 (차이 나는 복셀이 전부 ±1) | 12.34 | 247.9 |
| sub-09 run-1 | 292 | 146 | **0.000** (완전 일치) | 11.54 | 242.3 |

sub-08 의 ±1 은 정수 저장 반올림이다(평균 신호 248 대비 0.4%, 인접 볼륨 차이의 1/12). 변환 오류라면 인접 볼륨 차이와 같은 자릿수가 나와야 한다.

재현 명령은 `identity_check.sh` 에 있다.

## 남은 검토 항목

- `--interp=trilinear` 이다. `spline` 재산출 1회로 감쇠가 줄어드는지 보면 `hmc_v2` 의 종점 감쇠가 보간 기인인지 확정된다.
- `analyze_hmc.sh` 의 tSNR 은 **과제 신호와 drift 를 제거하지 않은 원자료** 위에서 `Tmean/Tstd` 로 계산된다. "HMC 가 품질을 개선하지 않는다" 는 논거가 여기 걸려 있으므로 GLM 잔차 기준으로 다시 재는 편이 낫다. C010 이 이미 잔차를 저장한다.
- 뇌 마스크를 영상별 `fslstats -P 40` 으로 잡으므로 두 arm 의 임계값이 함께 움직인다. ROI 겹침 +0.6% 를 실질 개선으로 읽지 않는 이유다(기존 문서에 이미 기록됨).
