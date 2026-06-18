# AI 뉴스 요약 및 분석기

뉴스 기사 URL을 입력하면 기사 본문을 수집하고, AI 요약·감정 분석·주제 분류 결과를 보여주는 Streamlit 애플리케이션입니다.

## 주요 기능

- `newspaper3k`를 이용한 뉴스 기사 제목 및 본문 추출
- KoBART 기반 한국어 기사 요약
- 다국어 BERT 기반 감정 분석
- 키워드 기반 뉴스 주제 분류

## 실행 방법

Python 3.10 환경을 권장합니다.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run news_analyzer.py
```

처음 실행할 때 Hugging Face에서 다음 모델을 내려받으므로 인터넷 연결과 충분한 저장 공간이 필요합니다.

- `gogamza/kobart-summarization`
- `nlptown/bert-base-multilingual-uncased-sentiment`

일부 언론사는 접속 정책이나 페이지 구조에 따라 본문 추출이 제한될 수 있습니다.

