import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import streamlit as st

model_id = "Writer/Palmyra-Fin-70B-32K"

SYSTEM_PROMPT="You are a highly knowledgeable and experienced expert in the financial sector, possessing extensive knowledge and practical expertise in financial analysis, markets, investments, and economic principles."

USER_PROMPT_DEFAULT="Can you explain how central banks printing more money (quantitative easing) affects the stock market and how investors might react to it?"



model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="flash_attention_2",
    token=access_token,
)
tokenizer = AutoTokenizer.from_pretrained(model_id) 

st.title("Your Financial Assistant")

st.text_input("System Prompt", key="sysprompt")

messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT,
    },
    {
        "role": "user",
        "content": USER_PROMPT_DEFAULT,
    },
]

if "messages" not in st.session_state:
  st.session_state.messages = messages

for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])

def run_llm(messages):
  input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
  gen_conf = {
    "max_new_tokens": 1024,
    "eos_token_id": tokenizer.eos_token_id,
    "temperature": 0.0,
    "top_p": 0.9,
  }
  with torch.inference_mode():
    output_id = model.generate(input_ids, **gen_conf)
  output_text = tokenizer.decode(output_id[0][input_ids.shape[1] :])
  return output_text


if prompt := st.chat_input(USER_PROMPT_DEFAULT):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
      st.markdown(prompt)

    with st.chat_message("assistant"):
      response = run_llm(st.session_state.messages)
      st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

