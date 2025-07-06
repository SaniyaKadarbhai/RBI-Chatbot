# chatbot_app.py

import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load fine-tuned model
base_model = AutoModelForCausalLM.from_pretrained("mistral-7b-instruct-local", device_map="auto", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base_model, "qlora-rbi-lora-adapter")
tokenizer = AutoTokenizer.from_pretrained("qlora-rbi-lora-adapter")

st.title("RBI Banking Chatbot")

user_input = st.text_area("Ask a question about banking regulations:")

if st.button("Submit") and user_input:
    prompt = f"### Instruction:\nAnswer the query about banking regulation.\n\n### Input:\n{user_input}\n\n### Response:\n"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=200)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    if "### Response:" in response:
        response = response.split("### Response:")[1].strip()

    st.markdown("**Answer:**")
    st.write(response)
