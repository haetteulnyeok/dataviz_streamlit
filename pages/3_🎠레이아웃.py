import streamlit as st 
'### : orange[컬럼: st.columns()]'
col_1, col_2, col_3 = st.columns([1,2,1])

with col_1:
    st.write('##1번 컬럼')
    st.checkbox('이것은 1번 컬럼에 속한 체크박스 1')
    st.checkbox('이것은 1번 컬럼에 속한 체크박스 2')