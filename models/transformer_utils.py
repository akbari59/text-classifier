from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tqdm import tqdm

class TransformerFeatureExtractor:
    def __init__(self, model_name='emilyalsentzer/Bio_ClinicalBERT', max_length=512, device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.max_length = max_length

    def encode(self, texts):
        embeddings = []
        with torch.no_grad():
            for text in tqdm(texts, desc='Encoding with BERT'):
                inputs = self.tokenizer(text, return_tensors='pt', truncation=True,
                                        padding='max_length', max_length=self.max_length).to(self.device)
                outputs = self.model(**inputs)
                # Use [CLS] token representation
                cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
                embeddings.append(cls_embedding)
        return np.array(embeddings)
