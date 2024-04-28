import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    return text

# Function to generate text
def generate_text(seed_text, model, tokenizer, max_sequence_len, num_words_to_generate):
    input_text = seed_text
    for _ in range(num_words_to_generate):
        token_list = tokenizer.texts_to_sequences([input_text])[0]
        token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        input_text += " " + output_word
    return input_text

# Load pre-trained model
model = tf.keras.models.load_model("new_model1.h5")
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)




# Streamlit app
def main():
    st.title("Text Generation of University of Colombo")
    num_words_to_generate = st.number_input('Number of words to generate:', min_value=10, max_value=100)

    # Text generation
    seed_text = st.text_input("Enter Seed Text", "university of colombo")
    if st.button("Generate Text"):
       
        generated_text = generate_text(seed_text, model, tokenizer, max_sequence_len=152, num_words_to_generate=num_words_to_generate)
        st.write("Generated Text:")
        st.write(generated_text)

# Run the app
if __name__ == "__main__":
    main()
