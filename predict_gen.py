import tensorflow as tf
import clean_text as ct

def gen(comb_text):
    
    processed_text = ct.final_clean(comb_text)

    imported = tf.saved_model.load("saved_model/1")

    prediction = imported(processed_text)

    output_value = prediction.numpy()[0, 0]

    if output_value >= 0.5:
        return 1
    else:
        return 0