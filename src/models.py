import torch
import torch.nn as nn
from torchvision import models
from transformers import BertModel, BlipForQuestionAnswering

class CNNBERTClassifier(nn.Module):
    def __init__(self, num_labels):
        super(CNNBERTClassifier, self).__init__()
        # Visual Stream (ResNet50)
        resnet = models.resnet50(pretrained=True)
        self.visual_backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Textual Stream (BERT)
        self.text_backbone = BertModel.from_pretrained("bert-base-uncased")
        
        # Fusion & Classifier
        self.visual_projection = nn.Linear(2048, 768)
        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels)
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        # Image features
        vis_feat = self.visual_backbone(pixel_values).flatten(1) # [batch, 2048]
        vis_feat = self.visual_projection(vis_feat) # [batch, 768]
        
        # Text features (CLS token)
        text_feat = self.text_backbone(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        
        # Concatenate and Classify
        combined = torch.cat((vis_feat, text_feat), dim=1)
        return self.classifier(combined)

def load_blip_model():
    """Helper to load Method 2 model"""
    return BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")