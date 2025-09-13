import random
import time
import gradio as gr

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")



def random_response(message, history):
    return random.choice(["Yes", "No"])

def slow_echo(message, history):
    for i in range(len(message)):
        time.sleep(0.01)
        yield "You typed: " + message[: i+1]
    print(history)
def chat_with_assistant(message, history):
    prompt = message
    for m in history:
        del m['metadata']
        del m['options']
    messages = history + [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt")

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(history)
    for i in range(len(response)):
        time.sleep(0.01)
        yield response[: i+1]
    

gr.ChatInterface(
    fn=chat_with_assistant,
    type="messages"
).launch()