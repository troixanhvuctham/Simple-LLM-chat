import time
import gradio as gr
import os
from utils import extract_answer, merge_small_chunks, extract_json, extract_think_and_left

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pymupdf
from langchain_core.documents import Document as LangchainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from Constant import SYSTEM_PROMPT2, RAG_PROMPT

MAX_LOOP = 3
MAX_TOKEN = 4000
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
PLANNER_MODEL_NAME = "Qwen/Qwen3-4B-Thinking-2507"
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    multi_process=True,
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
)

model = AutoModelForCausalLM.from_pretrained(
    PLANNER_MODEL_NAME,
    torch_dtype="auto",
    device_map="cuda"
)
tokenizer = AutoTokenizer.from_pretrained(PLANNER_MODEL_NAME)

RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(
    RAG_PROMPT, tokenize=False, add_generation_prompt=True
)
DOCUMENT_FILE = "Recipe-Book.pdf"
#DOCUMENT_FILE = "sub_grimm.pdf"

def load_knowledge():
    print("LOADING KNOWLEDGE...")
    doc = pymupdf.open(DOCUMENT_FILE) # open a document
    all_text = "\n\n".join([page.get_text() for page in doc])
    chunks = chunking_text(all_text)
    KNOWLEDGE_BASE = [LangchainDocument(page_content=chunk) for chunk in chunks]
    KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(KNOWLEDGE_BASE, embedding_model, distance_strategy=DistanceStrategy.COSINE)

    return KNOWLEDGE_VECTOR_DATABASE

def chunking_text(all_text):
    print("Chunking text")
    MARKDOWN_SEPARATORS = [
        "\n\n",
        "\n",
        " ",
        "",
    ]
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer,
            chunk_size=500,
            chunk_overlap=100,
            add_start_index=True,
            strip_whitespace=True,
            separators=MARKDOWN_SEPARATORS,
        )
    chunks = text_splitter.split_text(all_text)
    #chunks = merge_small_chunks(chunks)
    return chunks

def retrieve_document(user_query):
    print(f"\nStarting retrieval for {user_query=}...")
    retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=5)
    retrieved_docs_text = [doc.page_content for doc in retrieved_docs]
    context = "\nExtracted documents:\n"
    context += "".join(
        [f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)]
    )

    return context

def model_inferencing(text):
    model_inputs = tokenizer([text], return_tensors="pt")

    generated_ids = model.generate(
        model_inputs.input_ids.cuda(),
        max_new_tokens=MAX_TOKEN
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

HISTORY = [{"role": "system", "content": SYSTEM_PROMPT2}] 

def chat_with_assistant(message, history):

    new_message = {"role": "user", "content": message}
    HISTORY.append(new_message)
    messages = HISTORY

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    print("TEXT:")
    print(text)
    print("==============")

    response = model_inferencing(text)

    print("RESPONSE 1:")
    print(response)

    _, left = extract_think_and_left(response)
    data = extract_json(left)


    loop = 0
    while loop<3:
        loop+=1
        print("DATA JSON EXTRACTED: ", data)

        if data:
            result = execute_action(data)

            print("OUTPUT RAG:")
            print(result)
            print("=======OUTPUT RAG=========")
            HISTORY.append({"role": "tool", "content": result})

            print("HISTORY")
            print(HISTORY)
            print("=======END HISTORY=========")

            text = tokenizer.apply_chat_template(
                HISTORY,
                tokenize=False,
                add_generation_prompt=True)

            response = model_inferencing(text)
            print("RESPONSE 2:")
            print(response)
            print("END RESPONSE 2:")
            data = extract_json(response)
            if data == None:
                break
        else:
            break

    _, left = extract_think_and_left(response)
    answer = extract_answer(response)
    answer = left.strip()
    if answer == "No answer tag found":
        answer = response

    HISTORY.append({"role": "assistant", "content": answer})
    for i in range(len(answer)):
        time.sleep(0.01)
        yield answer[: i+1]

def call_rag(question):
    """
    Use this tool to answer questions about story,
    and something relevant such as characters, story summarization.
    """
    context = retrieve_document(question)
    #return context
    final_prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)

    print("PROMPT RAG:")
    print(final_prompt)

    answer = model_inferencing(final_prompt)

    answer = answer.replace("<think>","").replace("</think>","")
    return answer
    
def execute_action(data):
    action = data["action"]

    if action not in TOOLS:
        raise ValueError(f"Unknown tool: {action}")

    tool_function = TOOLS[action]
    result = tool_function(data["action_input"])

    return result
    
def main():
    gr.ChatInterface(
        fn=chat_with_assistant
    ).launch()
    
TOOLS = {
    "call_rag": call_rag,
}

if __name__ == "__main__":
    global KNOWLEDGE_VECTOR_DATABASE
    KNOWLEDGE_VECTOR_DATABASE = load_knowledge()
    print("STARTING...")
    gr.ChatInterface(
        fn=chat_with_assistant
    ).launch(share=False)
