import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

def load_model():
    print("Loading model: mlabonne/Hermes-3-Llama-3.1-70B-lorablated")
    tokenizer = AutoTokenizer.from_pretrained("mlabonne/Hermes-3-Llama-3.1-70B-lorablated")

    # Initialize an empty model
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained("mlabonne/Hermes-3-Llama-3.1-70B-lorablated")

    # Load the model checkpoint and dispatch across GPUs
    model = load_checkpoint_and_dispatch(
        model,
        "mlabonne/Hermes-3-Llama-3.1-70B-lorablated",
        device_map="auto",
        no_split_module_classes=["OPTDecoderLayer", "LlamaDecoderLayer", "BloomBlock", "GPTNeoXLayer", "GPTJBlock"],
    )

    print("Model loaded successfully.")
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=40000, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Set the maximum length for each generation step
    max_step_length = 1024
    generated_text = prompt

    with torch.no_grad():
        while len(tokenizer.encode(generated_text)) < max_length:
            # Generate text in smaller chunks
            outputs = model.generate(
                input_ids=tokenizer.encode(generated_text[-1024:], return_tensors="pt").to(model.device),
                max_length=min(max_step_length, max_length - len(tokenizer.encode(generated_text))),
                num_return_sequences=1,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            
            new_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text += new_text[len(generated_text):]
            
            if new_text.endswith(tokenizer.eos_token):
                break

    return generated_text
