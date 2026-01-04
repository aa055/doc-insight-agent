import gradio as gr
import traceback
from typing import Generator, List, Tuple

from src.config import settings


# Global state
class AppState:
    def __init__(self):
        self.doc_processor = None
        self.vector_store = None
        self.qa_chain = None
        self.summarizer = None
        self.notes_chain = None
        self.all_documents = []
        self.initialized = False

    def initialize(self) -> Tuple[bool, str]:
        """Initialize all components."""
        if self.initialized:
            return True, "Already initialized"

        try:
            from src.chains.notes_chain import NotesChain
            from src.chains.qa_chain import QAChain
            from src.chains.summarization_chain import SummarizationChain
            from src.core.document_processor import DocumentProcessor
            from src.core.vector_store import VectorStoreManager

            print("Initializing document processor...")
            self.doc_processor = DocumentProcessor()

            print("Initializing vector store...")
            self.vector_store = VectorStoreManager()

            print("Initializing chains...")
            self.qa_chain = QAChain(self.vector_store)
            self.summarizer = SummarizationChain()
            self.notes_chain = NotesChain()

            self.initialized = True
            print("Initialization complete!")
            return True, "Initialized successfully"

        except Exception as e:
            error_msg = f"Initialization failed: {str(e)}"
            print(f"ERROR: {error_msg}")
            traceback.print_exc()
            return False, error_msg


# Create global state
state = AppState()


def process_files(files) -> Tuple[str, list]:
    """Process uploaded PDF files."""
    if not files:
        return "No files selected. Please upload PDF files.", ["All Documents"]

    try:
        # Initialize if needed
        print("Step 1: Checking initialization...")
        success, msg = state.initialize()
        if not success:
            return msg, ["All Documents"]

        print("Step 2: Getting file paths...")
        file_paths = [f.name if hasattr(f, 'name') else f for f in files]
        print(f"  Files: {file_paths}")

        print("Step 3: Loading PDFs...")
        documents = state.doc_processor.load_multiple_pdfs(file_paths)
        print(f"  Loaded {len(documents)} pages")

        print("Step 4: Splitting documents...")
        chunks = state.doc_processor.split_documents(documents)
        print(f"  Created {len(chunks)} chunks")
        state.all_documents = chunks

        print("Step 5: Adding to vector store...")
        state.vector_store.add_documents(chunks)
        print("  Added to vector store")

        # Get document names
        doc_names = state.doc_processor.get_document_names()
        choices = ["All Documents"] + doc_names

        status = (
            f"Successfully processed {len(files)} file(s):\n"
            f"Documents: {', '.join(doc_names)}\n"
            f"Total chunks: {len(chunks)}"
        )
        print(f"Step 6: Complete - {status}")
        return status, choices

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"Processing error: {error_msg}")
        traceback.print_exc()
        return error_msg, ["All Documents"]


def respond(message: str, history: list, mode: str, doc_filter: str):
    """Generate response based on mode."""
    if not message.strip():
        return history, ""

    # Add user message to history
    history = history or []
    history.append({"role": "user", "content": message})

    # Check initialization
    if not state.initialized:
        success, msg = state.initialize()
        if not success:
            history.append({"role": "assistant", "content": msg})
            return history, ""

    # Check documents loaded
    if not state.all_documents:
        history.append({
            "role": "assistant",
            "content": "Please upload PDF documents first."
        })
        return history, ""

    try:
        filter_source = None if doc_filter == "All Documents" else doc_filter
        response_text = ""

        if mode == "Q&A":
            for chunk in state.qa_chain.query(message, filter_source):
                response_text += chunk

        elif mode == "Summarize":
            if doc_filter == "All Documents":
                for chunk in state.summarizer.summarize_all_documents(state.all_documents):
                    response_text += chunk
            else:
                for chunk in state.summarizer.summarize_document(doc_filter, state.all_documents):
                    response_text += chunk

        elif mode == "Generate Notes":
            if doc_filter == "All Documents":
                response_text = "Please select a specific document for notes."
            else:
                for chunk in state.notes_chain.generate_notes(doc_filter, state.all_documents):
                    response_text += chunk

        history.append({"role": "assistant", "content": response_text})

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"Response error: {error_msg}")
        traceback.print_exc()
        history.append({"role": "assistant", "content": error_msg})

    return history, ""


def clear_all():
    """Clear all data."""
    try:
        if state.vector_store:
            state.vector_store.clear_all()
        if state.qa_chain:
            state.qa_chain.clear_history()
        if state.doc_processor:
            state.doc_processor.clear()
        state.all_documents = []
    except Exception as e:
        print(f"Clear error: {e}")

    return [], "", ["All Documents"], "All Documents"


def clear_chat():
    """Clear chat history."""
    try:
        if state.qa_chain:
            state.qa_chain.clear_history()
    except Exception:
        pass
    return [], ""


def create_app():
    """Create the Gradio app."""
    with gr.Blocks(title="Doc Insight Agent") as demo:
        gr.Markdown("# Doc Insight Agent")
        gr.Markdown("Upload PDFs and chat with your documents using local AI")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Documents")
                file_input = gr.File(
                    label="Upload PDFs",
                    file_count="multiple",
                    file_types=[".pdf"],
                )
                process_btn = gr.Button("Process Documents", variant="primary")
                status_box = gr.Textbox(label="Status", lines=4, interactive=False)

                gr.Markdown("### Settings")
                mode = gr.Radio(
                    ["Q&A", "Summarize", "Generate Notes"],
                    value="Q&A",
                    label="Mode"
                )
                doc_dropdown = gr.Dropdown(
                    choices=["All Documents"],
                    value="All Documents",
                    label="Document Filter",
                    allow_custom_value=True
                )

                clear_all_btn = gr.Button("Clear All")

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Chat", height=500)
                msg_box = gr.Textbox(
                    label="Message",
                    placeholder="Type your message...",
                    lines=2
                )
                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary")
                    clear_chat_btn = gr.Button("Clear Chat")

        # Wire up events
        process_btn.click(
            fn=process_files,
            inputs=[file_input],
            outputs=[status_box, doc_dropdown]
        )

        send_btn.click(
            fn=respond,
            inputs=[msg_box, chatbot, mode, doc_dropdown],
            outputs=[chatbot, msg_box]
        )

        msg_box.submit(
            fn=respond,
            inputs=[msg_box, chatbot, mode, doc_dropdown],
            outputs=[chatbot, msg_box]
        )

        clear_chat_btn.click(
            fn=clear_chat,
            outputs=[chatbot, msg_box]
        )

        clear_all_btn.click(
            fn=clear_all,
            outputs=[chatbot, status_box, doc_dropdown, doc_dropdown]
        )

    return demo


def main():
    """Main entry point."""
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("Starting Doc Insight Agent")
    print("=" * 50)

    app = create_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
    )


if __name__ == "__main__":
    main()
