# app.py
import streamlit as st
from digit_generator import load_model, generate_digits

st.set_page_config(page_title="Digit Generator", layout="centered")
st.title("🧠 Handwritten Digit Generator")

# User selects digit
digit = st.selectbox("Select a digit to generate:", list(range(10)))

# Generate button
if st.button("Generate 5 Images"):
    with st.spinner("Generating images..."):
        model = load_model()
        images = generate_digits(model, digit)

    st.success("Here are your generated digits:")
    cols = st.columns(5)
    for i, img in enumerate(images):
        with cols[i]:
            st.image(img, caption=f"Digit {digit}", width=100)
