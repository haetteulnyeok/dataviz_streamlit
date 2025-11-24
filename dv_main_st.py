import streamlit as st

# 타이틀 텍스트 출력
st.title('이것은 나의 첫번째 Streamlit 웹 어플')

import streamlit as st

st.set_page_config(
    page_title="김민성의 Streamlit",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get help': "https://docs.streamlit.io",
        'Report a bug' : "https://streamlit.io",
        'About' : "### 김민성 \n - [홍익대학교 산업데이터공학과]"
    }
)

st.title("🔥 김민성의 Streamlit 앱")
st.write("여기부터 내용을 채워가면 돼!")

st.sidebar.title('다양한 사이드바 위젯들')
st.sidebar.checkbox('외국인 포함')
st.sidebar.checkbox('고령인구 포함')
st.sidebar.divider()
st.sidebar.radio('데이터 타입',['전체','남성','여성'])
st.sidebar.slider('나이',0,100,(20,50))
st.sidebar.selectbox('지역',['서울','경기','인천','대전','대구','부산','광주'])
