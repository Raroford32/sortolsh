import gradio as gr
from model_utils import load_model, generate_text

model, tokenizer = load_model()

def chat(message, history):
    history.append(message)
    response = generate_text(model, tokenizer, message)
    history.append(response)
    return history, history

iface = gr.Interface(
    fn=chat,
    inputs=["text", "state"],
    outputs=["chatbot", "state"],
    title="Hugging Face Chat-UI for LLM",
    description="Chat with the Large Language Model",
)

if __name__ == "__main__":
    iface.launch()
