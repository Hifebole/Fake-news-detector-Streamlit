from bs4 import BeautifulSoup
from nltk.corpus import stopwords

import string 

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# Define function to remove html
def remove_html(text):
    soup = BeautifulSoup(text, 'html.parser')
    cleantext = soup.get_text()
    return cleantext

def remove_stopwords(txt):
    # Define stopwords
    stop = set(stopwords.words('english'))
    words = []
    for word in txt.split():
        if word.strip().lower() not in stop:
            words.append(word.strip())
    cleantext = " ".join(words)
    return cleantext

def remove_punc(txt):
    punctuation = list(string.punctuation)
    punctuation.append("“")
    punctuation.append("”")
    punctuation.append("’")
    no_punct = ""
    for char in txt:
        if char not in punctuation:
            no_punct += char
    return no_punct


def embedding_tf(txt):
    text =[txt]
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    tokenizer_text = Tokenizer(num_words = 10000, oov_token=oov_tok)
    tokenizer_text.fit_on_texts(text)
    sequences_text = tokenizer_text.texts_to_sequences(text)
    final_text = pad_sequences(sequences_text, maxlen=300, padding=padding_type, truncating=trunc_type)
    return final_text

def final_clean(txt):
    stp1 = remove_html(txt)
    stp2 = remove_stopwords(stp1)
    stp3 = remove_punc(stp2)
    stp4 = embedding_tf(stp3)
    stp5 = stp4.reshape(1, -1)
    stp6 = tf.convert_to_tensor(stp5, dtype=tf.float32)    
    return stp6

