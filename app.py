import random
import time
import gradio as gr
import os
from utils import extract_answer


from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507",
    torch_dtype="auto",
    device_map="cuda"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
#tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-chat")

def chat_with_assistant(message, history):
    for m in history:
        #print(m)
        del m['metadata']
        del m['options']
        m['content'] = m['content'][0]['text']
        
    messages = ([{"role": "system", "content": "You are a helpful assistant.Please always include <answer> </answer> tag. Please think before answer request. You should think inside <thinking> </thinking> tag and answer in <answer> </answer> tag"}] 
           + history 
           + [{"role": "user", "content": message}
    ])

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    print("TEXT:")
    print(text)
    print("==============")

    model_inputs = tokenizer([text], return_tensors="pt")

    generated_ids = model.generate(
        model_inputs.input_ids.cuda(),
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(response)
    answer = extract_answer(response)
    
    for i in range(len(answer)):
        time.sleep(0.01)
        yield answer[: i+1]
    

gr.ChatInterface(
    fn=chat_with_assistant
).launch()