# Medical Visual Question Answering (VQA) Analysis

This is an educational research project focused on evaluating different Deep Learning architectures for **Visual Question Answering** in the medical domain. Using the **VQA-RAD** dataset, we compare a traditional discriminative approach with a modern generative approach.

## 📌 Project Overview
The goal of this project is to build a model that can understand a medical image (X-ray, CT, MRI) and provide a relevant answer to a natural language question. We explore two distinct methodologies:

1.  **Method 1: CNN + BERT (Classification)**
    * **Visual Stream:** ResNet-50 backbone for feature extraction.
    * **Textual Stream:** BERT (Bidirectional Encoder Representations from Transformers) for question understanding.
    * **Mechanism:** Multimodal fusion followed by a classification head.
    * **Best for:** Closed-ended questions (Yes/No).

2.  **Method 2: BLIP-1 (Generative)**
    * **Architecture:** Bootstrapping Language-Image Pre-training.
    * **Mechanism:** An encoder-decoder transformer model that generates answers token-by-token.
    * **Best for:** Open-ended descriptive questions and complex clinical reasoning.

---

## 📂 Project Structure

```text
medical-vqa-project/
├── data/                   # Dataset documentation and loading instructions
│   └── README.md           # Links to Hugging Face and data schema
├── notebooks/              # Interactive Jupyter/Colab notebooks
│   ├── 01_cnn_bert_vqa.ipynb   # Implementation of Method 1 (Classification)
│   └── 02_blip_vqa.ipynb       # Implementation of Method 2 (Generative BLIP)
├── src/                    # Modular source code
│   ├── datasets.py         # Custom Dataset classes and medical-safe augmentations
│   ├── models.py           # Model architectures for CNN+BERT and BLIP loaders
│   └── utils.py            # Utility functions for VRAM management and plotting
├── checkpoints/            # Local storage for trained model weights (.pth)
├── requirements.txt        # Python dependencies (Torch, Transformers, etc.)
├── .gitignore              # Rules to exclude large weights and cache folders
└── README.md               # Main project documentation