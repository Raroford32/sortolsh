from model_utils import generate_text

def run_cli(model, tokenizer, max_length, temperature):
    print("Welcome to the Large Language Model Inference CLI!")
    print("Type 'exit' to quit the program.")

    while True:
        prompt = input("\nEnter your prompt: ")
        if prompt.lower() == 'exit':
            print("Exiting the program. Goodbye!")
            break

        try:
            generated_text = generate_text(model, tokenizer, prompt, max_length, temperature)
            print("\nGenerated text:")
            print(generated_text)
        except Exception as e:
            print(f"An error occurred during text generation: {str(e)}")

