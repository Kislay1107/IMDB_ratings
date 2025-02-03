import pandas as pd
import numpy as np
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

dict_imdb = imdb.get_word_index()

model = load_model('IMBD.keras')

st.title("IMDB movie ratings")
input = st.text_area("Write your review: ")

if st.button('Classify'):
    encoded_sent = [dict_imdb[word] + 3 if word in dict_imdb else 2 for word in input.lower().split()]
    
    encoded_sent = np.array(encoded_sent)
    encoded_sent = sequence.pad_sequences([encoded_sent], 500)
    prediction = model.predict(encoded_sent)
    
    if prediction[0][0] > 0.5:
        st.write("Positive Review")
    else:
        st.write("Negative Review")
        
else:
    st.write('Please enter a movie review.')