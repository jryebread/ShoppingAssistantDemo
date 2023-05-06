import os
import re
import uuid

import streamlit as st
import pandas as pd
import pinecone
from dotenv import load_dotenv
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.vectorstores import Pinecone
load_dotenv()
embed_model = "text-embedding-ada-002"
import openai

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment="us-east4-gcp"
)

from langchain.chains.query_constructor.base import AttributeInfo

BASE_META_PROMPT="""You are an AI Assistant.
Here are some sources to items that match the users search criteria.
Use the following pieces of filtered items to answer the users questions about the items.

Begin!
--------
Sources:
{summaries}
"""

metadata_field_info=[
    AttributeInfo(
        name="item_name",
        description="Name of Item",
        type="string",
    ),
    AttributeInfo(
        name="item_desc",
        description="Item description",
        type="string",
    ),
    AttributeInfo(
        name="item_cost",
        description="Item cost",
        type="float"
    ),
]

@st.cache_resource
def load_pinecone_existing_index(uuid):
    docsearch = Pinecone.from_existing_index(index_name="NAME_OF_YOUR_INDEX", embedding=OpenAIEmbeddings(), namespace=uuid)
    return docsearch
@st.cache_resource
def load_pinecone_index():
    index = pinecone.Index("NAME_OF_YOUR_INDEX")
    return index
index = load_pinecone_index()

def chat(query, uuid):
    vectorstore = load_pinecone_existing_index(uuid)
    llm = OpenAI(temperature=0.1, verbose=True)
    retriever = SelfQueryRetriever.from_llm(llm, vectorstore, "Item description", metadata_field_info, verbose=True)
    docs = retriever.get_relevant_documents(query)
    streaming_llm = ChatOpenAI(model_name="gpt-3.5-turbo", streaming=True, verbose=True, temperature=0, max_tokens=600)

    sysTemplate = SystemMessagePromptTemplate.from_template(BASE_META_PROMPT)
    messages = [
        sysTemplate,
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    prompt = ChatPromptTemplate.from_messages(messages)

    doc_chain = load_qa_chain(llm=streaming_llm, document_variable_name="summaries", chain_type="stuff", prompt=prompt,
                              verbose=True)
    ret = doc_chain(
        {
            "input_documents": docs,
            "question": query,
        },
        return_only_outputs=True
    )
    return ret, docs

def csv_embed(data, uuid_str):
    batch_size = 100
    ids = [], embeds = [], metas = []
    for i in range(0, len(data.item_name), batch_size):
        batch_texts = [' '.join(row) for row in zip(data.item_name[i:i + batch_size], data.item_desc[i:i + batch_size], data.item_cost[i:i + batch_size])]
        res = openai.Embedding.create(input=batch_texts, engine=embed_model)
        for j, row in enumerate(
                zip(data.item_name[i:i + batch_size], data.item_desc[i:i + batch_size], data.item_img[i:i + batch_size],
                    data.item_cost[i:i + batch_size])):
            embed = res['data'][j]['embedding']
            id = str(uuid.uuid4())
            meta = {"text": batch_texts[j], "item_name": row[0], "item_desc": row[1], "item_img": row[2],
                    "item_cost": float(re.sub(r'[^\d.]+', '', row[3])) if row[3] != '' and any(c.isnumeric() for c in row[3]) else 0.0}
            ids.append(id)
            embeds.append(embed)
            metas.append(meta)

    vectors = list(zip(ids, embeds, metas))
    index.upsert(vectors=vectors, namespace=uuid_str)

# Function to display chat bubbles
def display_chat_bubble(sender, message, image=None):
    if sender == "user":
        st.markdown(f'<div style="display: flex; justify-content: flex-start; margin-bottom: 10px;"><div style="background-color: #f0f0f0; border-radius: 15px; padding: 10px; max-width: 80%;">{message}</div></div>', unsafe_allow_html=True)
    elif sender == "ai":
        if image:
            st.image(image, width=200)
        st.markdown(f'<div style="display: flex; justify-content: flex-end; margin-bottom: 10px;"><div style="background-color: #4f8bf9; color: white; border-radius: 15px; padding: 10px; max-width: 80%;">{message}</div></div>', unsafe_allow_html=True)

# Main app
def main():
    st.title("Shopping Assistant Chatbot")

    # Upload CSV
    st.header("Upload CSV")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    uuid_new = str(uuid.uuid4())
    if uploaded_file is not None:
        # Load CSV
        df = pd.read_csv(uploaded_file, header=None)

        df.columns = ['', 'item_desc', 'item_name', 'item_cost', 'actual_price', 'url', 'item_img']
        df = df.drop(df.columns[0], axis=1)

        csv_embed(df, uuid_new) # create vector embeddings for semantic search

        # Chat interface
        st.header("Chat")
        user_input = st.text_input("Type your message here...")

        if st.button("Send"):
            # Display user message
            display_chat_bubble("user", user_input)

            ai_response, docs = chat(user_input, uuid_new)

            # Display AI response
            display_chat_bubble("ai", ai_response['output_text'], image=docs[0].metadata['item_img'])

if __name__ == "__main__":
    main()