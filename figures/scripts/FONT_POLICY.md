# 그림 폰트 정책 — Imaging Neuroscience (2026-08-17)

IN 요구: *"Please use Arial or Helvetica fonts within figures."*

## 적용 방법

생성 스크립트를 개별 수정하지 않고 `MATPLOTLIBRC` 로 일괄 강제한다.

```bash
export MATPLOTLIBRC="<repo>/docs/PAPER/Figures/scripts/inrc"
python3 generate_fig2.py
```

`inrc/matplotlibrc` 가 `font.sans-serif` 1순위를 Arial 로 두고 **mathtext 폰트셋까지** Arial 로 지정한다.

## 이전 시도가 실패한 이유

일부 스크립트는 이미 `font.family: 'Arial'` 을 설정하고 있었는데도 출력 PDF 에 DejaVu 가 남아 있었다. **mathtext 는 별도 폰트셋(`mathtext.fontset`)을 쓰고 기본값이 `dejavusans`** 이기 때문이다. `$\beta_s$` 같은 수식 라벨이 전부 DejaVuSans-Oblique 로 박혔다. `mathtext.fontset: custom` + `mathtext.rm/it/bf/sf: Arial` 이 실제 해법이다.

`"font.family": "sans-serif"` 만 지정한 스크립트는 `font.sans-serif` 목록 1순위(matplotlib 기본 = DejaVu Sans)로 떨어진다. rc 가 이 목록을 Arial 우선으로 바꾼다.

## 스크립트 직접 수정이 필요했던 2건

| 파일 | 수정 | 이유 |
|---|---|---|
| `phase2/generate_fig6_landscape.py`, `phase2/generate_fig7_filter.py` | `font.sans-serif` 를 Helvetica 우선 → **Arial 우선** | Helvetica 도 IN 허용이나 나머지 7개와 섞이면 조판 일관성이 깨진다 |
| `../../Scripts/generate_forward_encoder_fig.py` | `$\to$` → `→`(평문), `\mathbf{W}^\top` → `\mathbf{W}^\mathsf{T}` | `\to`·`\top` 은 Arial 에 없어 **Cmsy10**(Computer Modern) 으로 폴백했다 |

## 재생성 후 발견·수정한 레이아웃 결함

| 파일 | 결함 | 조치 |
|---|---|---|
| `generate_fig2.py` | 패널 문자와 제목이 **붙어서 한 단어로 읽힘** (`ADiscrimination`). 문자 x=0.0, 제목 x=0.07(axes fraction)이 10 pt 볼드에서 간격 0 이 됨 | 제목·부제 offset 0.07 → **0.105** |

## 현재 상태

| 그림 | 임베드 폰트 | 판정 |
|---|---|---|
| fig2_loro_loco | ArialMT, Arial-BoldMT | ✅ |
| fig3_geometry_r6 | ArialMT, Arial-BoldMT | ✅ |
| fig6_landscape | ArialMT, Arial-Bold/ItalicMT | ✅ |
| fig7_filter | ArialMT, Arial-Bold/ItalicMT | ✅ |
| fig8_filter_eval | ArialMT, Arial-BoldMT | ✅ |
| fig_forward_encoder | ArialMT, Arial-Bold/ItalicMT | ✅ |
| figS3_forward_tuning | ArialMT, Arial-BoldMT | ✅ |
| figS2_adjacc_saturation | ArialMT, Arial-Bold/ItalicMT | ✅ |
| fig1_paradigm_v3 (Figure 1) | ArialMT, Arial-Bold/ItalicMT | ✅ **2026-08-18 해결** — `generate_fig1_v3.py` 로 전면 재작성 |
| fig3_workflow (Figure 3) | ArialMT, Arial-BoldMT | ✅ **2026-08-18 해결** — PowerPoint 재출력 |

**잔여 없음. `main.pdf` 전체에서 DejaVu·Aptos·Cmsy 가 모두 사라졌다.**

## fig3_workflow — PowerPoint 경로

원본 = `Figures/fig3_assets/Presentation1.pptx` (수작업 합성, `generate_box2_delta_rdm_r6.py` 주석에 기록).
글꼴이 **Aptos**(Office 신규 기본값)라 IN 요구를 위반한다.

**LibreOffice 자동 변환은 실패했다.** XML 에서 Aptos→Arial 치환 후 `soffice --headless --convert-to pdf` 한 결과, 페이지 크기가 842×299.5 → 960×540 으로 바뀌고 텍스트 메트릭 차이로 레이아웃이 깨졌다:
박스 제목 잘림(`Candidate loss atoms`→`Candidate loss`, `Personalized filter`→`Personalized`), `Not selected` 배지→`Not`, 박스 1 수식이 테두리 밖으로 넘침.

### 해결 (2026-08-18)

사용자가 PowerPoint 에서 재출력했다 → `fig3_assets/Presentation1_arial_fontpatched.{pdf,png}`.

검증 결과:

| 항목 | 결과 |
|---|---|
| 임베드 폰트 | ArialMT + Arial-BoldMT **만** (Aptos·DejaVu·Cmsy 전무) |
| LibreOffice 가 깨뜨렸던 지점 | 5개 박스 제목 전부 완전, `Not selected` 배지 완전, 박스 1 수식이 테두리 안 — **전부 정상** |
| CropBox | `[43.37, 139.89, 800.24, 417.55]` = 756.87×277.66 — 콘텐츠 밴드로 잡혀 LaTeX 가 여백을 자동 제거 (원본보다 타이트) |

**부수 이득**: 원고가 이제 `fig3_workflow.pdf`(벡터)를 집는다. 이전에는 래스터 PNG 였으므로 인쇄 품질이 개선됐다.

반영: `fig3_workflow.pdf` / `fig3_workflow_composited.pdf` / `fig3_workflow.png`(RGBA→RGB 평탄화, 300 dpi) 갱신. 구 Aptos 판본은 `archive/fig3_aptos_superseded_2026-08-18/` 로 이동. 원본 `Presentation1.pptx` 는 손대지 않았다.

## fig1_generated_v2 (Figure 1) — 유일한 잔여

이미지 생성 AI 산출물이고 생성 스크립트가 없다. 아카이브의 `generate_fig1.py` 는 다른 산출물(`fig1_paradigm`)을 만들지만, 그 도크스트링에 **실제 해부 렌더링**이 있다고 기록되어 있다.

> `B  Real ROI flatmap: V1/V2/V3/hV4 on inflated brain (nilearn + VTPM Wang 2015 maxprob atlas)`

권장 경로: 뇌·ROI 패널은 이 실제 렌더링을 쓰고, 박스·화살표·텍스트는 PowerPoint(Arial)로 조립한다. 그러면 ① 폰트 요건 충족 ② AI 이미지가 사라져 공개 문안이 코드 첨삭만으로 축소(`COVER_LETTER_DRAFT.md` 블록 4 초안 B) ③ AI 생성 뇌 해부의 신뢰성 위험 제거.
