# 메인 페이지 설정
import streamlit as st
import pandas as pd
# 타이틀 텍스트 출력
st.title('C117023 김민성')
st.header('K팝 데몬 헌터스 온라인 데이터 분석')
st.subheader('팬덤 형성 핵심 요인 다각도 분석 및 인사이트 제공')


st.set_page_config(                        # 페이지 설정
    page_title="3차시험_김민성의 Streamlit",        # 페이지 Tab의  타이틀 
    page_icon="🔥",                        # 페이지 Tab의  아이콘
    layout="wide",                         # 페이지 레이아웃: centered, wide
    # 사이드바 초기 상태: auto, collapsed, expanded
    initial_sidebar_state="expanded",
)


st.sidebar.title('다양한 사이드바 위젯들')


########################################################3
# 5개 이상 위젯





# ##############################
# WordCloud 시각화
import re
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from matplotlib import font_manager


@st.cache_data
def load_data():
    return pd.read_csv("kpopdemonhunters.csv")

df = load_data()
st.dataframe(df.head())

# 분석에 사용할 텍스트 전처리
text_raw = " ".join(df["title"].astype(str).tolist())

def clean_text(text):
    text = re.sub(r"<[^>]*>", "", text)
    text = re.sub(r"[^\w\sㄱ-ㅎㅏ-ㅣ가-힣]", "", text)
    return text

text_cleaned = clean_text(text_raw)


# 사이드바 위젯1 슬라이더
st.sidebar.header("워드클라우드 옵션")
max_words = st.sidebar.slider("최대 단어 수", 10, 100, 50)

background_color = "black"

# 불용어 설정
stop_str = "kpop 데몬 헌터스 콘텐츠 작품 팬덤 영상 장면 공개"
stop_words = set(stop_str.split(" "))
STOPWORDS.update(stop_words)


# 한글 폰트 설정
han_font_path = font_manager.findfont("Gulim")

# WordCloud 생성
st.subheader("팬덤 핵심 키워드 워드클라우드")

wordcloud = WordCloud(
    font_path=han_font_path,
    max_words=max_words,
    stopwords=STOPWORDS,
    background_color=background_color,
    width=800,
    height=800,
    colormap="coolwarm"
).generate(text_cleaned)

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(wordcloud)
ax.axis("off")

st.pyplot(fig)

st.divider()
##########################################################################
# 네트워크 시각화
st.subheader(" 키워드 관계 네트워크")

import networkx as nx
from konlpy.tag import Okt
from itertools import combinations
from collections import Counter

# 텍스트 전처리
descriptions = df["description"].astype(str).tolist()

okt = Okt()

# 7 불용어 사전 불러오기
with open('korean_stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = f.read().splitlines()

all_nouns = []

for text in descriptions:
    # 한글만 남기기
    text_cleaned = re.sub(r"[^가-힣\s]", "", text)
    # 명사 추출
    nouns = okt.nouns(text_cleaned)
    # 한 글자 제거
    nouns = [word for word in set(nouns) if len(word) > 1 and (word not in stopwords)]
    all_nouns.append(nouns)

# Edge 리스트 생성 
edge_list = []

for nouns in all_nouns:
    if len(nouns) > 1:
        edge_list.extend(combinations(sorted(nouns), 2))

edge_counts = Counter(edge_list)

# 가장 많이 등장한 10개의 엣지 출력
print(edge_counts.most_common(10))

min_count = 20
filtered_edges = {edge: weight for edge, weight in edge_counts.items() if weight >= min_count}
st.write(f"네트워크에 사용된 엣지 수: {len(filtered_edges)}")


# NetworkX 그래프 생성
G = nx.Graph()

for (node1, node2), weight in filtered_edges.items():
    G.add_edge(node1, node2, weight=weight)

# 시각화 (spring layout)
# 레이아웃 생성
pos_spring = nx.spring_layout(
    G, # 그래프 객체
    k=0.3, # 노드 간격 조절 파라미터
    iterations=50, # 반복 횟수
    seed=42
)

# 9 노드 크기 설정 (차수 기반)
node_sizes = [G.degree(node) * 100 for node in G.nodes()]

# 12 엣지 두께 설정 (가중치 기반)
edge_widths = [G[u][v]['weight'] * 0.05 for u, v in G.edges()]

# 15 그래프 그리기
fig, ax = plt.subplots(figsize=(15, 15))

nx.draw_networkx(
    G,
    pos_spring,
    with_labels=True,
    node_size=node_sizes,
    width=edge_widths,
    font_family=plt.rcParams['font.family'],
    font_size=12,
    node_color='skyblue',
    edge_color='gray',
    alpha=0.8,
    ax=ax
)

ax.set_title("K-POP 데몬 헌터스 키워드 네트워크", size=18)
ax.axis("off")

st.pyplot(fig)

st.divider()
######################################################
# Seaborn 그래프
# 막대그래프
import pandas as pd
import seaborn as sns


df = pd.read_csv("kpopdemonhunters.csv")

df.head()

# 리스트로 변환
descriptions = df["description"].astype(str).tolist()

okt = Okt()

all_nouns = []

for text in descriptions:
    # 한글과 공백만 남기기
    text_cleaned = re.sub(r"[^가-힣\s]", "", text)
    # 명사 추출
    nouns = okt.nouns(text_cleaned)
    # 한 글자 단어 제거
    nouns = [word for word in nouns if len(word) > 1]
    all_nouns.extend(nouns)

word_count = Counter(all_nouns)

top_words = word_count.most_common(10)

# 데이터프레임 변환
# ai 코드 참조(그래프 생성부분)
word_df = pd.DataFrame(top_words, columns=["keyword", "count"])

word_df

st.subheader("K-POP 데몬 헌터스 키워드 빈도 (Seaborn)")

fig, ax = plt.subplots(figsize=(8, 5))

# 막대 그래프 그리기
sns.barplot(
    data=word_df,
    x="count",
    y="keyword",
    ax=ax
)

ax.set_title("뉴스 기사 기반 키워드 언급 빈도")
ax.set_xlabel("언급 횟수")
ax.set_ylabel("키워드")

st.pyplot(fig)

#####################################################
# Altair
import altair as alt
df = pd.read_csv("kpopdemonhunters.csv")

# 날짜 컬럼을 datetime 타입으로 변환
df["pubDate"] = pd.to_datetime(df["pubDate"])

# 날짜만 추출
df["date"] = df["pubDate"].dt.date

# 날짜별로 기사 개수 세기
date_count = (
    df.groupby("date")
      .size()
      .reset_index(name="count")
)
date_count.head()

# 그래프 그리기
# ai 코드 참조(그래프 생성부분)
st.subheader("K-POP 데몬 헌터스 뉴스 언급 추이")
chart = (
    alt.Chart(date_count)
    .mark_line(point=True)
    .encode(
        x=alt.X("date:T", title="날짜"),
        y=alt.Y("count:Q", title="기사 수"),
        tooltip=["date:T", "count:Q"]
    )
)
st.altair_chart(chart, use_container_width=True)

st.text('뉴스 기사 개수가 12월 초까지 증가하는 추세인 것을 보아 오랫동안 이슈화되었음을 알 수 있다.')
st.divider()
#########################################################3
# Plotly
import plotly.express as px

df = pd.read_csv("kpopdemonhunters.csv")

descriptions = df["description"].astype(str).tolist()

okt = Okt()

nouns_all = []

for text in descriptions:
    text_cleaned = re.sub(r"[^가-힣\s]", "", text)
    nouns = okt.nouns(text_cleaned)
    nouns = [word for word in nouns if len(word) > 1]
    nouns_all.extend(nouns)

word_count = Counter(nouns_all)

# ai 코드 참조(그래프 생성부분)
st.subheader("팬덤 핵심 키워드 빈도 (Plotly)")

fig = px.bar(
    word_df,
    x="count",
    y="keyword",
    orientation="h",
    title="K-POP 데몬 헌터스 핵심 키워드 빈도{top_n}",
    labels={
        "count": "언급 횟수",
        "keyword": "키워드"
    }
)

st.plotly_chart(fig, use_container_width=True)

# ai 코드 그대로 참고
# 슬라이더 위젯 (상위 키워드 개수)
top_n = st.slider(
    "보고 싶은 키워드 개수 선택",
    min_value=5,
    max_value=30,
    value=10,
    step=1
)

top_words = word_count.most_common(top_n)


word_df = pd.DataFrame(top_words, columns=["keyword", "count"])
