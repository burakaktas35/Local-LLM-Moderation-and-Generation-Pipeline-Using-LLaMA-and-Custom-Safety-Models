# Local-LLM-Moderation-and-Generation-Pipeline-Using-LLaMA-and-Custom-Safety-Models
This repository presents a fully local pipeline for controlled text generation using LLMs. The system employs the LLaMA-3.1-8B-Instruct model for text generation, integrated with a Prompt Guard model to detect jailbreak and prompt injection attempts, and a Llama Guard model for classifying and moderating potentially unsafe output.


## 🚀 Project Overview

This project offers a fully local LLM pipeline for controlled text generation, focusing on security, privacy, and speed. The moderation process begins with evaluating user inputs for jailbreak and prompt injection risks using Prompt Guard, then monitors generated outputs for unsafe content with Llama Guard, and finally provides refined, moderated responses using the LLaMA-3.1-8B-Instruct model. Crucially, the use of ```llama-cpp-python``` significantly enhances inference speed compared to traditional Transformers-based methods for LLaMA usage, making the entire pipeline more efficient.

**Key technologies used:**

**Generation Model:** Meta-LLaMA-3.1-8B-Instruct for contextual text generation (you will need to have ```.gguf``` version of the model)

**Moderation Models:** Prompt Guard (for input filtering) and Llama Guard (for output moderation)

**Frameworks:** PyTorch and Transformers

**Device:** Fully local, operating on available hardware resources (CPU/GPU)


## 🌟 Features

**Local Moderation:** No data leaves the local environment, ensuring maximum security and privacy.

**Multi-Layered** Input and Output Checking: Input text is evaluated for safety before generating a response, and generated responses are checked for potential unsafe content.

**Optimized Inference with llama-cpp-python:** Enhances inference speed for LLaMA models compared to traditional Transformers-based usage, ensuring faster response times without compromising on model performance.

**Customizable Moderation Models:** Integrates fine-tuned local models for adaptable safety checks.

**Efficient Text Generation:** LLaMA-based contextual generation with configurable parameters like temperature, top_p, and top_k.


## 🛠️ Setup

Ensure you have the required models downloaded locally and accessible to your hardware.

After setting up models locally, you will need to change a file inside ```meta-llama/Llama-Guard-3-8B``` model. ```tokenizer_config.json``` file needs to be replaced as provided in repository. This is because, the usage differences. If you want you use it as in the code provided, then you have to change.

## 📈 Advantages

**Enhanced Inference Speed:** Leverages llama-cpp-python for faster model inference, outperforming traditional Transformers-based methods in speed for LLaMA model usage.

**Complete Data Privacy:** All moderation and generation processes occur locally, with no data leaving your environment.

**Multi-Layered Content Safety:** Reduces the risk of harmful content by employing layered checks for both inputs and outputs.

**Flexible Customization:** Models can be fine-tuned and adjusted to match different moderation requirements.

**Resource Efficiency:** Optimized operations to make effective use of local hardware resources.


## 📖 How It Works

**Input Moderation:** User input is checked for jailbreak and prompt injection using the Prompt Guard model, which assigns scores and determines if the input is safe for further processing.

**Output Moderation:** The LLaMA model generates responses, which are then checked using Llama Guard to classify and assess for potential safety violations.

**Text Generation:** If input and output are determined to be safe, LLaMA generates and returns the contextual response. This process benefits from the speed enhancements offered by llama-cpp-python.


## 📊 Examples

- **Moderating Input Texts:**

Evaluate a user-provided input string for jailbreak or prompt injection risks.

- **Running the Pipeline:**

    ```bash
    python run_pipeline.py  # Adjust paths and parameters as necessary

## 🤖 Future Improvements

**Support for Additional Models:** Expand the system to work with different generation and moderation models.

**Enhanced Moderation Rules:** Refine moderation techniques to better detect nuanced or complex cases.

**Advanced Optimization:** Improve memory management and inference speeds to support larger models more effectively.


## 💡 Technical Considerations

**Memory Usage:** Large models require significant memory; ensure sufficient GPU resources are available if applicable. You will need at least 32 GiB VRAM to work on the same pipeline. But you may use different models for your GPU VRAM capacity, this is just a demo of how pipeline is being used.

**Customization:** Moderation thresholds and settings can be tailored to fit different applications.

**Data Security:** Consider additional security measures for deploying this system in shared environments.
