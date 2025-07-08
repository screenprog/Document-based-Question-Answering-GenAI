# Retrieval-Augmented Generation (RAG)

The entire system is an implementation of a pattern called Retrieval-Augmented Generation (RAG).

## What is RAG?

Think of a large language model (LLM) like Gemini as a very knowledgeable person who has read a vast amount of public information (like the internet) up to a certain date. However, it doesn't know anything about your *private* or *specific* documents (like a PDF you have).

If you ask the LLM a question about your document, it won't be able to answer it.

RAG solves this problem. Instead of just asking the LLM the question directly, we first:

1.  **Retrieve:** Find the most relevant snippets of text from your document that relate to the question.
2.  **Augment:** Take those relevant snippets and "augment" the LLM's knowledge by providing them as context along with your original question.
3.  **Generate:** Ask the LLM to generate an answer *based only on the context we just gave it*.

This way, the LLM can answer questions about information it has never seen before. Your project is a classic example of this pattern.

# Embeddings

## What are Embeddings?

In simple terms, an embedding is a way to represent text (or other data like images and audio) as a list of numbers, called a **vector**. This numerical representation captures the *semantic meaning* of the text.

Think of it like this:

*   The words "king" and "queen" are semantically similar. Their vector representations would be close to each other in a multi-dimensional space.
*   The words "king" and "cabbage" are not semantically similar. Their vector representations would be far apart.

## Why are Embeddings Important in RAG?

Embeddings are the magic that allows us to find the "most relevant snippets of text" from your document. Here's how it works in your project:

1.  **Document Embedding**: When you run `load_data.py`, it takes your document, splits it into chunks, and then uses an embedding model (in your case, from Google) to convert each chunk into a vector. These vectors are then stored in your ChromaDB vector database.
2.  **Query Embedding**: When you ask a question in `main.py`, your question is *also* converted into a vector using the *same* embedding model.
3.  **Similarity Search**: The system then compares the vector of your question to all the vectors of the document chunks in the database. The chunks whose vectors are "closest" to the question's vector are considered the most relevant.

This "closeness" is measured using mathematical techniques like cosine similarity.

So, embeddings are the bridge that allows the computer to understand the *meaning* of your question and find the parts of your document that are most likely to contain the answer.

# Vector Stores

## What is a Vector Store?

A vector store (or vector database) is a special kind of database designed specifically to store and search through embeddings (those lists of numbers we just discussed).

Imagine you have a giant library where every book is just a single, very long number (its embedding). If you were given a new number (your query's embedding) and asked to find the books with the most similar numbers, it would be incredibly slow to compare your number to every single book one by one.

A vector store is like a magical librarian for this library. It organizes the books (vectors) in a very clever way so that when you give it your number, it can instantly point you to the most similar ones without having to check every single book.

## How does it work in your project?

1.  **Storing:** In `load_data.py`, after the documents are chunked and converted into embeddings, they are loaded into a `Chroma` vector store. ChromaDB takes these embeddings and organizes them efficiently in a persistent location (the `chroma_storage` directory).

2.  **Searching (Retrieval):** In `main.py`, when you ask a question, the application takes your query's embedding and asks the ChromaDB vector store: "Please give me the top 'k' document chunks that are most similar to this query embedding."

ChromaDB performs a very fast **similarity search** (also called a "vector search" or "nearest neighbor search") and returns the most relevant document chunks. This is the "Retrieval" step in RAG.

Without a specialized vector store like ChromaDB, this retrieval process would be far too slow and inefficient to be practical, especially with large documents or many documents.

# LangChain

## What is LangChain?

If the LLM is the brain, the embedding model is the translator, and the vector store is the library, then **LangChain is the conductor of the orchestra**.

LangChain is a framework that makes it much easier to build applications powered by LLMs. It provides a set of tools, components, and "chains" that handle the complex "plumbing" required to get all the different parts of a RAG system to work together.

Instead of you having to manually write code to:
1.  Take the user's query.
2.  Send it to the embedding model.
3.  Take the resulting vector and send it to the vector store.
4.  Receive the relevant document chunks back.
5.  Format those chunks into a prompt.
6.  Send the final prompt to the LLM.
7.  Get the answer back...

...LangChain provides pre-built components and abstractions that handle most of this for you.

## How is it used in your project?

In `main.py`, you can see LangChain being used in several key places:

1.  **`GoogleGenerativeAIEmbeddings`**: This is a LangChain component that provides a standardized way to interact with Google's embedding models. You don't have to write the raw API calls yourself.
2.  **`Chroma`**: LangChain has a wrapper for ChromaDB, making it easy to use as a vector store and connect it to other components.
3.  **`ChatGoogleGenerativeAI`**: Similar to the embeddings, this is a standard interface for interacting with Google's chat models.
4.  **`SelfQueryRetriever`**: This is one of the most powerful LangChain components in your project. The `SelfQueryRetriever` is a "chain" that does something very clever: it uses an LLM to analyze the user's initial query and automatically generate a more structured query for the vector store. For example, if a user asks, "What did the president say about taxes in the 2022 state of the union?", the self-query retriever can automatically figure out that it should filter the search to only include documents where `filename` is `state_of_the_union_2022.txt`. This makes the retrieval step much more precise.
5.  **Chains (in general)**: LangChain allows you to link all these components together into a "chain." The `retriever.invoke(query)` call in your code kicks off a chain of events that seamlessly handles the entire RAG pipeline: from self-querying to retrieval to passing the context to the final LLM call (which you do manually in `get_gemini_response`).

In short, LangChain acts as a high-level framework that orchestrates all the other components, saving you a massive amount of boilerplate code and allowing you to focus on the logic of your application.

# What is Streamlit?

Streamlit is an open-source Python library that makes it incredibly easy to create beautiful, custom web applications for machine learning and data science.

Think of it as a way to turn your Python scripts into interactive web apps with minimal effort. You write pure Python code, and Streamlit handles all the complex web development stuff (HTML, CSS, JavaScript) behind the scenes.

Here are its core concepts:

1.  **Python-Native:** You build your UI using standard Python functions and variables. No web development experience is required.
2.  **Widgets:** Streamlit provides simple functions for common UI elements like buttons (`st.button`), text input (`st.text_input`), file uploaders (`st.file_uploader`), sliders, checkboxes, etc.
3.  **Data Flow:** Streamlit apps re-run from top to bottom every time a user interacts with a widget. This "data flow" model simplifies how you think about state and updates.
4.  **Session State (`st.session_state`):** Because apps re-run, you need a way to preserve information across re-runs (like chat history or loaded documents). `st.session_state` is a dictionary-like object that allows you to store and retrieve variables persistently.
5.  **Layout:** You can arrange your widgets and content using simple functions like `st.sidebar` for a sidebar, `st.columns` for multi-column layouts, and `st.expander` for collapsible sections.