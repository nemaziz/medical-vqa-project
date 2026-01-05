# 📁 Medical VQA-RAD Dataset

This directory provides details on the dataset used for training and evaluating our VQA models. We utilize the **VQA-RAD** dataset, which consists of clinically verified images and question-answer pairs.

## 🔗 Dataset Source
The data is accessed via the Hugging Face Hub:
* **Identifier:** `flaviagiammarino/vqa-rad`
* **Repository:** [Hugging Face - VQA-RAD](https://huggingface.co/datasets/flaviagiammarino/vqa-rad)

## 📊 Dataset Statistics
VQA-RAD is a high-quality, small-scale medical dataset:
* **Total Samples:** ~2,244 QA pairs.
* **Images:** 315 unique medical images (CT, MRI, X-ray).
* **Splits used in this project:**
    * **Training:** 1,793 rows (85%)
    * **Test:** 451 rows (15%)

## 📑 Data Features
Each entry in the dataset contains:
- `image`: The medical scan (PIL image).
- `question`: The clinical query (e.g., "Is the heart enlarged?").
- `answer`: The ground truth response (e.g., "Yes").



## 🛠 Pre-processing & Categorization
For our comparative analysis, we separate the data based on question types:

1. **Closed-Ended:** Questions requiring a 'Yes' or 'No' response. 
   - *Strategy:* Treated as a binary classification task in Method 1.
2. **Open-Ended:** Questions requiring descriptive answers (e.g., "What is the abnormality?").
   - *Strategy:* Evaluated using the generative capabilities of BLIP-1 in Method 2.

## 🚀 How to Load
The dataset is loaded automatically through our notebooks using the `datasets` library:

```python
from datasets import load_dataset

# Load the dataset directly from Hugging Face
vqa_rad = load_dataset("flaviagiammarino/vqa-rad")