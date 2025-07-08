__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import sys
from chain import get_gemini_response, retrieve_documents
from vectorizer import create_vector_store_from_files, get_available_documents

print("Python Executable:", sys.executable)
print("Python Path:", sys.path)


def main():
    st.set_page_config(page_title="Chat With Documents", page_icon="📚")
    st.title("Chat With Your Documents 💬")
    st.write("Upload your documents and ask questions about them!")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"]=="assistant" and "sources" in message:
                with st.expander("View Sources"):
                    st.write(message["sources"])

    with st.sidebar:
        st.header("Your Documents")
        st.subheader("Available Documents")
        available_docs = get_available_documents()
        for doc_name in available_docs:
            st.write(f"📄 {doc_name}")
        st.divider()
        uploaded_files = st.file_uploader(
            "Upload your PDF or DOCX file here",
            accept_multiple_files=True,
            type=["pdf", "docx"]
        )
        if st.button("Process Documents"):
            with st.spinner("Processing..."):
                chunks = create_vector_store_from_files(uploaded_files)
                st.write(fr"Created {chunks} chunks")
                st.success("Done!")

    if query:=st.chat_input("Ask a question about your documents"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                context, sources = retrieve_documents(query=query)
            with st.spinner("Thinking..."):
                response = get_gemini_response(query, context)
                st.markdown(response)
                with st.expander("View Sources"):
                    st.write(sources)
            st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources})


if __name__ == "__main__":
    main()
