# 변형 욕설까지 잡아내는 문맥 보존형 비속어 순화기

기계학습 미니프로젝트 #1 (NLP 모델 구현 및 개선)

RNN 탐지기로 욕설 **위치**를 찾고, RNN 언어 모델로 문맥에 맞는 순화 표현을 골라 **문제 구간만** 바꾼 뒤, 바꾼 문장을 다시 검사합니다.

```
시1발 오늘 존나 덥네   →  오늘 정말 덥네
ㅈㄴ 맛있다           →  정말 맛있다
개가 너무 귀엽다       →  개가 너무 귀엽다   (정상 문장은 그대로)
```

## 저장소 구성

이 저장소는 교재 『밑바닥부터 시작하는 딥러닝 2』 코드([WegraLee/deep-learning-from-scratch-2](https://github.com/WegraLee/deep-learning-from-scratch-2), MIT 라이선스)를 그대로 두고, 우리 코드를 `project/` 폴더에 새로 작성한 것입니다. **교재 원본 파일은 수정하지 않았습니다.**

| 위치 | 내용 |
|---|---|
| `common/`, `ch01`~`ch08/`, `dataset/` | 교재 원본 코드 (원본 README는 `README_원본교재.md`) |
| [`project/`](project/) | 우리 구현 (자모 처리, 양방향 RNN 탐지기, 순화기, 평가, 실험) — 자세한 내용은 [project/README.md](project/README.md) |

## 실행

```bash
git clone <이 저장소 주소>
cd <저장소>/project
python3 -m venv .venv && .venv/bin/pip install numpy matplotlib
.venv/bin/python train_demo.py      # 예시 데이터로 학습 + E1~E4 비교 (CPU 약 20초)
.venv/bin/python experiments.py     # E5~E7
```

실제 데이터 실험(`real_data.py`)과 KcBERT 비교(`kcbert_fill.py`)는 [project/README.md](project/README.md)의 안내대로 데이터셋을 각 저장소에서 받은 뒤 실행합니다. 데이터셋 원본은 라이선스 때문에 이 저장소에 포함하지 않았습니다.

## 참조 코드

| 참조 코드 | 데이터셋 | 쓰임 |
|---|---|---|
| [WegraLee/deep-learning-from-scratch-2](https://github.com/WegraLee/deep-learning-from-scratch-2) | PTB | 구현 기반 (ch04 Embedding·CBOW, ch05 TimeRNN·SimpleRnnlm·RnnlmTrainer) |
| [2runo/Curse-detection](https://github.com/2runo/Curse-detection) | [Curse-detection-data](https://github.com/2runo/Curse-detection-data) | 자모 분해 아이디어, 욕설 문장 평가 |
| [smilegate-ai/korean_unsmile_dataset](https://github.com/smilegate-ai/korean_unsmile_dataset) | UnSmile | 깨끗한 문장, 심판 분류기 |
| [kocohub/korean-hate-speech](https://github.com/kocohub/korean-hate-speech) | Korean HateSpeech | 깨끗한 문장 (언어 모델) |
| [adlnlp/K-MHaS](https://github.com/adlnlp/K-MHaS) | K-MHaS | 깨끗한 문장 확장 |
| [boychaboy/KOLD](https://github.com/boychaboy/KOLD) | KOLD | 공격 구간 대비 평가 |
| [minkyeongjeon/kda](https://github.com/minkyeongjeon/kda) | K/DA | 비교 대상 (LLM 순화 연구) |
| [beomi/kcbert-base](https://huggingface.co/beomi/kcbert-base) | - | E8 비교군 (Mask-and-Fill) |

## 팀

양태규 (인공지능융합공학부) · 송용현 (IoT전자공학과)
