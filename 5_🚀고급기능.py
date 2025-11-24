import time
import streamlit as st 
@st.cache_data
def long_running_function(param1):
    time.sleep(5)
    return param1*param1

start = time.time()

# 숫자 입력은 입력된 값을 반환
num_1 = st.number_input('입력한 숫자의 제곱을 계산합니다.')
st.write(f'{num_1}의 제곱은 {long_running_function(num_1)}입니다.'+
         f'계산시간은 {time.time()-start:.2f}초 소요')
st.write('🏷:green[캐싱이 적용되면 동일한 계산은 저장된  결과를 사용하여 빠르게 처리함]')

import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(20,2), columns=["x","y"])

st.write('### :orange[session_state를 사용하지 않은 경우]')
color1 = st.color_picker("Color1","#FF0000")
st.divider()
st.scatter_chart(df, x="x", y="y", color=color1)

if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(np.random.randn(20,2), columns=['x','y'])

st.write('### :orange[session_state를 사용한 경우]')
color2 = st.color_picker("Color2", "#FF0000")
st.divider()
st.scatter_chart(st.session_state.df, x="x", y="y", color=color2)
st.write(':green[session_state를 사용하면, 저장된 state를 사용하므로 값이 고정됨]')