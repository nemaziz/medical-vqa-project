import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class MedVQADataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, ans_to_idx, transform=None, max_len=32):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.ans_to_idx = ans_to_idx
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Processing image
        image = item['image'].convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Processing question
        question = str(item['question'])
        inputs = self.tokenizer.encode_plus(
            question,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            return_tensors='pt'
        )

        # Processing answer
        answer_text = item['answer'].lower().strip()
        # If in test data we face new answer we set 0 or process separately
        answer_label = self.ans_to_idx.get(answer_text, 0)

        return {
            'image': image,
            'ids': inputs['input_ids'].flatten(),
            'mask': inputs['attention_mask'].flatten(),
            'label': torch.tensor(answer_label, dtype=torch.long)
        }


class VQADataset(Dataset):
    def __init__(self, dataset, processor, tokenizer, label2id, image_processor=None, train=True):
        self.dataset = dataset
        self.processor = processor
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.train = train
        
        # Аугментации (без флипов для сохранения мед. контекста)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) if train else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image'].convert("RGB")
        question = item['question']
        answer = item['answer']

        # Обработка изображения
        image_tensor = self.transform(image)
        
        # Токенизация вопроса (BERT)
        encoding = self.tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # Метка (Classification)
        label = torch.tensor(self.label2id.get(answer, 0))

        return {
            "pixel_values": image_tensor,
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": label
        }