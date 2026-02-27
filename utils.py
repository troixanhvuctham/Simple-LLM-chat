import re
import json
from langchain.tools import tool


def extract_answer(text):
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)

    if match:
        # group(1) refers to the captured content inside the parentheses
        extracted_text = match.group(1)
        return extracted_text
    else:
        return "No answer tag found"
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
@tool
def document_search(query: str) -> str:
    """
    Use this tool to answer questions about story,
    and something relevant such as characters, story summarization.
    Input should be a natural language question.
    """
    docs = retriever.get_relevant_documents(query)
    return combine_docs(docs)