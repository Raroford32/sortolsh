import argparse
from cli import run_cli
from model_utils import load_model, generate_text
from chat_ui import iface as chat_ui

def main():
    parser = argparse.ArgumentParser(description="Large Language Model Inference CLI")
    parser.add_argument("--prompt", type=str, help="Initial prompt for text generation")
    parser.add_argument("--max_length", type=int, default=40000, help="Maximum length of generated text. Can be set to higher values for generating large amounts of code.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for text generation")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--chat-ui", action="store_true", help="Launch the Hugging Face chat-ui")

    args = parser.parse_args()

    model, tokenizer = load_model()

    if args.chat_ui:
        chat_ui.launch()
    elif args.interactive:
        run_cli(model, tokenizer, args.max_length, args.temperature)
    elif args.prompt:
        generated_text = generate_text(model, tokenizer, args.prompt, args.max_length, args.temperature)
        print(generated_text)
    else:
        print("Please provide a prompt, use interactive mode, or launch the chat-ui.")

if __name__ == "__main__":
    main()
