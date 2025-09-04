# crypto_wallet_classifier


# Ethereum Wallet Transaction Classifier

This project is a machine-learning classifier that analyzes recent transactions of an Ethereum wallet and predicts their category (e.g., Token Swap, NFT Purchase, DeFi Interaction, Simple Transfer, Contract Deployment, Unknown).  

It combines blockchain data from the Etherscan API with a LoRA-fine-tuned Large Language Model (LLM), and exposes a simple FastAPI backend plus a minimal web UI.

---

## Features
- Input any valid Ethereum wallet address.  
- Fetches the most recent transactions (normal, ERC-20, and NFT transfers).  
- Classifies each transaction into one of 6 categories using the fine-tuned LLM.  
- Minimal clean web interface with transaction hash + classification results.  

---

##Setup & Running Locally

#1. Clone the Repository
```bash
git clone https://github.com/<your-username>/crypto_wallet_classifier.git
cd crypto_wallet_classifier

2. Create & Activate a Virtual Environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

4. Configure API Keys

Create a .env file in the root directory with:

ETHERSCAN_API_KEY=your_etherscan_api_key
BASE_ID=microsoft/Phi-3.5-mini-instruct
ADAPTER_DIR=lora_out
USE_GPU=1

Get a free API key from Etherscan
.

BASE_ID points to the base LLM model.

ADAPTER_DIR contains your LoRA fine-tuned weights.

Set USE_GPU=0 if running on CPU only

5. Run the Backend
uvicorn main:app --reload

6. Access the Web UI

Navigate to:
 http://127.0.0.1:8000/ui/


Design Choices
LLM Selection

Used Microsoft’s Phi-3.5-mini-instruct as the base model:

Lightweight and efficient compared to larger models (fits on consumer GPUs).

Instruction-tuned for classification-style tasks.

Easy integration with Hugging Face and LoRA adapters.

Fine-Tuning Approach
Fine-tuned using LoRA (Low-Rank Adaptation) for efficiency.

Training data was structured as JSON-style prompts, each containing:
Transaction metadata (from, to, value, input).
A labeled category (Token Swap, NFT Purchase, etc.).
This structure closely matches inference-time prompts, improving generalization.

Tech Stack

Backend: Python + FastAPI

Easy to build APIs, integrates smoothly with ML models.

Frontend: Vanilla HTML/JS (minimal index.html)

Keeps deployment simple; no heavy React/Next.js needed for the challenge.

ML Framework: Hugging Face Transformers + PEFT (LoRA)

Standard tooling for efficient fine-tuning and inference.

Data Source: Etherscan API

Widely used Ethereum data provider with reliable endpoints.

Challenges & Solutions
1. GPU Memory Constraints

Running the model on an 8GB GPU caused CUDA OOM errors.

Solution: Added support for CPU inference and quantized inference (4-bit/8-bit) as fallbacks.

2. Model Over-Offloading

Hugging Face accelerate tried to offload layers incorrectly.

Solution: Disabled device_map=auto and instead controlled placement manually.

3. Consistent JSON Outputs

LLM sometimes produced Markdown-wrapped JSON.

Solution: Implemented a lightweight regex/JSON extraction layer in hf_infer.py.

4. Web Deployment

Initial idea was Ollama, but it didn’t fit our fine-tuned weights.

Switched to Hugging Face Transformers + FastAPI for portability.

Used ngrok / Render for quick public exposure.

