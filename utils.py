import re
import json
from langchain.tools import tool


def ________extract_answer(text):
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)

    if match:
        # group(1) refers to the captured content inside the parentheses
        extracted_text = match.group(1)
        return extracted_text
    else:
        return "No answer tag found"
        
def extract_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_think_and_left(text: str) -> str:
    '''
    Extract thinking part and the remain part from LLM response
    '''
    think_and_left =  text.split("</think>")
    think = think_and_left[0].split("<think>")[0].strip()
    left = think_and_left[1]
    return think, left
    
def merge_small_chunks(chunks, min_length = 200):
    merged = []
    buffer = ""
    
    for chunk in chunks:
        if len(chunk.split(" ")) < min_length:
            buffer += " " + chunk
            if len(buffer.split(" ")) >=min_length:
                merged.append(buffer.strip())
                buffer = ""
    if buffer:
        merged.append(buffer.strip())
    
    return merged

def extract_json(text):
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except:
        return None