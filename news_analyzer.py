import streamlit as st
from newspaper import Article
from transformers import pipeline
import torch.nn.functional as F
import torch
import re

# --- 전처리 함수 ---
def clean_article_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'▶.*?ⓒ.*?무단전재.*', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'이미지.*?제공.*?\n', '', text)
    return text.strip()

# --- 키워드 기반 주제 분류 함수 ---
def keyword_based_topic(text):
    keywords = {
        "정치": ["대통령", "정당", "국회", "총선", "정책", "의회", "정부", "선거", "장관", "국무총리"],
        "경제": ["주식", "금리", "환율", "기업", "부동산", "경기", "소비자", "무역", "투자", "물가"],
        "사회": ["사건", "사고", "경찰", "법원", "시민", "노동", "범죄", "실종", "폭력", "안전"],
        "문화": ["전시", "공연", "축제", "예술", "문화재", "박물관", "문학", "전통", "공예", "유산"],
        "과학": ["기후", "기술", "의학", "연구", "우주", "실험", "질병", "백신", "생명과학", "자연과학"],
        "스포츠": ["경기", "선수", "득점", "골", "리그", "월드컵", "올림픽", "야구", "축구", "농구"],
        "기술": ["AI", "인공지능", "로봇", "소프트웨어", "개발자", "빅데이터", "IT", "디지털", "코딩", "컴퓨터"],
        "국제": ["미국", "중국", "일본", "전쟁", "외교", "국제사회", "유엔", "분쟁", "정상회담", "핵"],
        "연예": ["드라마", "아이돌", "연예인", "영화", "음악", "방송", "가수", "예능", "앨범", "팬"],
    }
    for topic, words in keywords.items():
        if any(word in text for word in words):
            return topic
    return "기타"

# --- 요약 모델 ---
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="gogamza/kobart-summarization", device=0 if torch.cuda.is_available() else -1)

# --- 감정 분석 (3분류 + 설명 포함) ---
@st.cache_resource
def load_sentiment():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", device=0 if torch.cuda.is_available() else -1)

def interpret_sentiment(result):
    label = result['label']
    score = result['score']
    label_map = {
        "1 star": "부정적",
        "2 stars": "부정적",
        "3 stars": "중립적",
        "4 stars": "긍정적",
        "5 stars": "긍정적",
    }
    explanation_map = {
        "부정적": "사고 소식, 부정적인 정책, 경제 위기, 범죄, 갈등 등 부정적인 분위기의 뉴스로 보입니다.",
        "중립적": "정보 전달 위주이거나 애매한 내용으로 감정을 명확히 판단하긴 어렵습니다.",
        "긍정적": "희소식, 좋은 정책, 긍정적인 성과, 개선된 제도 등 긍정적인 분위기의 뉴스로 보입니다."
    }
    sentiment_label = label_map.get(label, "알 수 없음")
    explanation = explanation_map.get(sentiment_label, "")
    percent = round(score * 100)
    return sentiment_label, percent, explanation

# --- 모델 로딩 ---
summarizer = load_summarizer()
sentiment_pipe = load_sentiment()

# --- UI 구성 ---
st.title("📰 AI 뉴스 요약 및 분석기")
st.write("뉴스 URL을 입력하면, AI가 요약하고 감정 및 주제 분석을 자연스럽게 설명해드립니다.")

url = st.text_input("🔗 뉴스 기사 URL을 입력하세요:")

if url:
    with st.spinner("📥 뉴스 기사 불러오는 중..."):
        try:
            article = Article(url, language='ko')
            article.download()
            article.parse()
            content = clean_article_text(article.text)
            title = article.title
        except Exception as e:
            st.error(f"❌ 기사 불러오기 실패: {e}")
            content = None

    if content:
        st.subheader("📰 기사 제목")
        st.write(title)

        st.subheader("📄 원본 기사 내용")
        st.write(content[:1000] + "..." if len(content) > 1000 else content)

        # --- 요약 ---
        with st.spinner("📝 요약 중..."):
            try:
                summary = summarizer(content[:1024], max_length=150, min_length=40, do_sample=False)[0]['summary_text']
            except Exception as e:
                summary = f"❌ 요약 실패: {e}"

        st.subheader("✍️ 요약 결과")
        st.success(summary)

        # --- 감정 분석 ---
        with st.spinner("😊 감정 분석 중..."):
            try:
                result = sentiment_pipe(content[:512])[0]
                sentiment_label, percent, explanation = interpret_sentiment(result)
                st.subheader("😃 긍정도 평가 결과")
                st.markdown(f"""
모델은 약 **{percent}%의 확신**으로 이 기사를 **{sentiment_label} 내용**이라고 판단했습니다.  
📝 {explanation}
""")
            except Exception as e:
                st.error(f"❌ 감정 분석 실패: {e}")

        # --- 주제 분류 (키워드 기반) ---
        with st.spinner("🏷️ 주제 분류 중..."):
            try:
                topic = keyword_based_topic(content)
                st.subheader("📌 주제 분류 결과")
                st.markdown(f"뉴스 내용에서 주요 단어를 분석한 결과, **{topic} 뉴스**로 분류됩니다.")
            except Exception as e:
                st.error(f"❌ 주제 분류 실패: {e}")
