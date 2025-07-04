import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

st.set_page_config(page_title="RBI Chatbot", page_icon="??", layout="centered")

st.markdown("""
    <style>
    .main {
        background-color: black;
    }
    .stButton>button {
        background-color: #002868;
        color: white;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 8px;
        margin-top: 10px;
    }
    .stTextArea>div>textarea {
        border-radius: 8px;
        padding: 10px;
    }
    .rbi-header {
        text-align: center;
        font-size: 32px;
        color: #002868;
        font-weight: 800;
        padding-top: 10px;
    }
    .chatbox {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.image("RBI_logo.png", width=100)
st.markdown("<div class='rbi-header'>RBI Banking Regulation Chatbot</div>", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='chatbox'>", unsafe_allow_html=True)

    user_input = st.text_area("Ask your banking query:")

    if st.button("Submit") and user_input:
        with st.spinner("Processing..."):
            # Load model & tokenizer
            base_model = AutoModelForCausalLM.from_pretrained(
                "mistral-7b-instruct-local",
                device_map="auto",
                torch_dtype=torch.float16
            )
            model = PeftModel.from_pretrained(base_model, "qlora-rbi-lora-adapter_temp")
            tokenizer = AutoTokenizer.from_pretrained("qlora-rbi-lora-adapter_temp")

            # Prompt construction
            prompt = f"### Instruction:\nAnswer the query about banking regulation.\n\n### Input:\n{user_input}\n\n### Response:\n"
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            output = model.generate(**inputs, max_new_tokens=200)
            response = tokenizer.decode(output[0], skip_special_tokens=True)

            # Extract only the answer portion
            if "### Response:" in response:
                response = response.split("### Response:")[1].strip()

        st.markdown("**Answer:**")
        st.success(response)

    st.markdown("</div>", unsafe_allow_html=True)
