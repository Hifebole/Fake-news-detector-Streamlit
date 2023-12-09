import streamlit as st
import predict_gen as pg

st.title("Fake News Detector")

st.subheader("Title")

input_title = st.text_input("Enter Title")

input_text = st.text_area("Enter Article body")

comb_text = input_text + " " + input_title



if st.button("Predict"):
    prediction = pg.gen(comb_text)    
    if prediction == 0 :            
            st.error('This is Fake News')                  
    elif prediction == 1 :            
            st.success('This News is Authentic') 

if st.button("Predict Next"):
    # Trigger a rerun to reset the input fields
    st.experimental_rerun()

       