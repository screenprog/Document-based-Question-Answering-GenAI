# Document-based Question Answering System (GenAI)

## Objective
The primary goal of this project is to create a Generative AI application that allows users to upload documents (specifically PDFs and Word files) and ask questions based on the content of those documents.

## Tech Stack
*   **Language Model Framework**: LangChain
*   **LLM Provider**: Google AI Studio (any model)
*   **Vector Database**: ChromaDB
*   **User Interface**: Streamlit

## Description
This application leverages the power of Large Language Models (LLMs) and vector search to provide intelligent, context-aware answers from within a user-provided document.

When a user uploads a document, the application performs the following steps:
1.  **Text Extraction**: Extracts the text from the uploaded file.
2.  **Chunking**: Splits the extracted text into smaller, manageable chunks.
3.  **Embedding**: Uses Google AI Studio embeddings to convert each text chunk into a numerical vector representation.
4.  **Vector Storage**: Stores these embeddings in a ChromaDB vector database.

When a user asks a question:
1.  The user's query is also embedded using the same embedding model.
2.  The system performs a similarity search to find the most relevant text chunk(s) from the vector database.
3.  These relevant chunks are then passed as context to the LLM.
4.  The LLM generates a comprehensive answer based on the provided context.

## Key Features
*   **File Upload**: Supports uploading PDF and Word documents.
*   **Text Chunking and Vector Embedding**: Automatically processes and embeds the content of the uploaded documents.
*   **Context-Aware Answer Generation**: Provides answers that are grounded in the content of the uploaded documents.
*   **User-Friendly Interface**: A simple and professional user interface built with Streamlit.

## Use Case
This application is particularly useful in scenarios where users need to quickly find information within large documents. Potential applications include:
*   **Legal Firms**: For quickly searching through legal documents, case files, and contracts.
*   **Educational Platforms**: To help students find answers to their questions within textbooks and research papers.
*   **Knowledge Management**: As a tool for employees to easily query internal documentation, reports, and wikis.

## Future Enhancements
*   **Support for More File Types**: Extend the application to support other common file formats like `.txt`, `.csv`, and `.pptx`.
*   **Chat History**: Implement a feature to maintain a history of the user's conversation with the document.
*   **Improved UI/UX**: Enhance the user interface with features like a "Clear Chat" button, loading spinners, and more interactive elements.
*   **"Show Sources" Feature**: Add a button or a section to display the exact text chunks that were used to generate the answer, providing more transparency and context.
*   **Dockerization**: Containerize the application using Docker for easier setup, deployment, and scalability.
*   **Advanced Error Handling**: Implement more robust error handling for file uploads, API calls, and other potential issues.
