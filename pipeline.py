import torch

from llama_cpp import Llama
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

llm = Llama(
        model_path = "/path/to/model/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
        n_ctx=2048,
        n_gpu_layers = -1,
)

prompt_guard_model_id = "/path/to/model/safe_unsafe/Prompt-Guard-86M"
prompt_guard_tokenizer = AutoTokenizer.from_pretrained(prompt_guard_model_id)
prompt_guard_model = AutoModelForSequenceClassification.from_pretrained(prompt_guard_model_id)

llama_guard_model_id = "/path/to/model/safe_unsafe/Llama-Guard-3-8B"
device = "cuda"
llama_guard_model = AutoModelForCausalLM.from_pretrained(
    llama_guard_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
llama_guard_tokenizer = AutoTokenizer.from_pretrained(llama_guard_model_id)

system_prompt_llm = "You are an assistant that does blablabla etc."

# Prompt Guard Jailbreak Score Calculation
def get_jailbreak_score(model, tokenizer, text, temperature=1.0, device='cpu'):
    """
    Evaluate the probability that a given string contains malicious jailbreak or prompt injection.
    Appropriate for filtering dialogue between a user and an LLM.
    
    Args:
        text (str): The input text to evaluate.
        temperature (float): The temperature for the softmax function. Default is 1.0.
        device (str): The device to evaluate the model on.
        
    Returns:
        float: The probability of the text containing malicious content.
    """
    probabilities = get_class_probabilities(model, tokenizer, text, temperature, device)
    return probabilities[0, 2].item()


# Prompt Guard Prompt Injection Score Calculation
def get_indirect_injection_score(model, tokenizer, text, temperature=1.0, device='cpu'):
    """
    Evaluate the probability that a given string contains any embedded instructions (malicious or benign).
    Appropriate for filtering third party inputs (e.g., web searches, tool outputs) into an LLM.
    
    Args:
        text (str): The input text to evaluate.
        temperature (float): The temperature for the softmax function. Default is 1.0.
        device (str): The device to evaluate the model on.
        
    Returns:
        float: The combined probability of the text containing malicious or embedded instructions.
    """
    probabilities = get_class_probabilities(model, tokenizer, text, temperature, device)
    return (probabilities[0, 1] + probabilities[0, 2]).item()


# Prompt Guard Class Probabilities Score Calculation
def get_class_probabilities(model, tokenizer, text, temperature=1.0, device='cpu'):
    """
    Evaluate the model on the given text with temperature-adjusted softmax.
    Note, as this is a DeBERTa model, the input text should have a maximum length of 512.
    
    Args:
        text (str): The input text to classify.
        temperature (float): The temperature for the softmax function. Default is 1.0.
        device (str): The device to evaluate the model on.
        
    Returns:
        torch.Tensor: The probability of each class adjusted by the temperature.
    """
    # Encode the text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    # Get logits from the model
    with torch.no_grad():
        logits = model(**inputs).logits
    # Apply temperature scaling
    scaled_logits = logits / temperature
    # Apply softmax to get probabilities
    probabilities = softmax(scaled_logits, dim=-1)
    return probabilities


# Llama Guard 3 8B Safe/Unsafe Checking Function
def moderate(chat):
    input_ids = llama_guard_tokenizer.apply_chat_template(
        chat, 
        return_tensors="pt",
        excluded_category_keys=["S6"]).to(device)
    output = llama_guard_model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    result = (llama_guard_tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True))
    print(f"{result}\n")
    return result

classes = {
    "S1": "Violent Crimes",
    "S2": "Non-Violent Crimes",
    "S3": "Sex-Related Crimes",
    "S4": "Child Sexual Exploitation",
    "S5": "Defamation",
    "S6": "Specialized Advice",
    "S7": "Privacy",
    "S8": "Intellectual Property",
    "S9": "Indiscriminate Weapons",
    "S10": "Hate",
    "S11": "Suicide & Self-Harm",
    "S12": "Sexual Content",
    "S13": "Elections"
}

while True:
    text = input(f"\n\nEnter:")

    # Prompt Guard Part
    print(f"\njailbreak_score: {get_jailbreak_score(prompt_guard_model, prompt_guard_tokenizer, text=text)}")
    print(f"prompt_injection_score: {get_indirect_injection_score(prompt_guard_model, prompt_guard_tokenizer, text=text)}")
        
    inputs = prompt_guard_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = prompt_guard_model(**inputs).logits

    predicted_class_id = logits.argmax().item()
    prompt_guard_class=prompt_guard_model.config.id2label[predicted_class_id]
    print(f"\nprompt_guard_class: {prompt_guard_class}")

    if prompt_guard_class == "JAILBREAK":
        print(f"JAILBREAK DETECTED! THIS INCIDENT WILL BE REPORTED TO THE ADMIN.\n")
        continue

    llama_guard_text = [
    {f"role": "system", "content": {system_prompt_llm}},
    {f"role": "user", "content": {text}}
    ]

    # Llama Guard 3 8B Part
    temp_llama_guard = moderate([text])

    class_code = temp_llama_guard.split('\n')[-1]
    if class_code in classes:
        print(f"{classes[class_code]}\n")

    if "unsafe" in temp_llama_guard:
        print(f"UNSAFE TEXT DETECTED! THIS INCIDENT WILL BE REPORTED TO THE ADMIN.\n")
        continue
    
    # Llama 3.1 8B Part
    messages = [
    {f"role": "system", "content": {system_prompt_llm}},
    {f"role": "user", "content": {text}}
    ]

    output = llm.create_chat_completion(
        messages,
        max_tokens=1024,
        temperature=0.6,
        #stop=["Q", "\n"],
        top_p=0.9,
        top_k=4,
        repeat_penalty=1.1
    )
    output = output["choices"][0]["message"]["content"]

    # Llama Guard 3 8B Part
    temp_llama_guard = moderate([output])

    class_code = temp_llama_guard.split('\n')[-1]
    if class_code in classes:
        print(f"{classes[class_code]}\n")

    if "unsafe" in temp_llama_guard:
        print(f"UNSAFE TEXT DETECTED! THIS INCIDENT WILL BE REPORTED TO THE ADMIN.\n")
        continue
    
    print(f"\n{output}")