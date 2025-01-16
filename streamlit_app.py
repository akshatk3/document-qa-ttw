import streamlit as st
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI 


st.title("📄 QA Doc Mbbs ttw")

openai_api_key = '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'

client = OpenAI(api_key=openai_api_key)
llm_model = "gpt-4o" 
llm = ChatOpenAI(temperature = 0.7, model=llm_model)


uploaded_file = st.file_uploader(
    "Upload a document", type=("txt", "pdf", "etc.")
    )

#embed it and perform similarity search etc
user_text = st.text_input("What you want huh?: ")

if uploaded_file and user_text:

    # Process the uploaded file and question.
        document = uploaded_file.read().decode()
        vector_store=InMemoryVectorStore.from_documents(document, OpenAIEmbeddings())
        retriever = vector_store.as_retriever()
        
        results = vector_store.similarity_search(user_text, retriever=retriever, top_k=5)
        
        r = results[0].page_content
        
        # Long text displayed as separate lines
        long_text_lines = r.splitlines()

        # Scrollable container
        with st.container():
            for line in long_text_lines:
                st.write(line)

        
        
        messages = [
            {
                "role": "user",
                "content": f"Here's the source document: {results[0].cast_id_to_str} \n\n---\n\n {user_text}",
            }
        ]

        # Generate an answer using the OpenAI API.
        stream = client.chat.completions.create(
            model=llm_model,
            messages=messages,
            stream=True,
        )

        # Stream the response to the app using `st.write_stream`.
        st.write_stream(stream)
