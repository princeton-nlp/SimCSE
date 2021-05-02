import logging
import numpy as np
from numpy import ndarray
import torch
from torch import Tensor, device
import transformers
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Type, Union

logger = logging.getLogger(__name__)

"""
A simple class used for several downstream zero-shot sentence embedding tasks:
"""
class SentenceEmbedder(object):
    def __init__(self, model_name_or_path: str, device: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
    
    def encode(self, sentence: Union[str, List[str]], 
                device: str = None, 
                return_numpy: bool = False) -> Union[ndarray, Tensor]:

        target_device = self.device if device is None else device
        self.model.to(target_device)
        
        single_sentence = False
        if isinstance(sentence, str):
            sentence = [sentence]
            single_sentence = True
        
        inputs = self.tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")
        for feature in inputs:
            inputs[feature].to(target_device)
        with torch.no_grad():
            embeddings = self.model(**inputs, output_hidden_states=True, return_dict=True).pooler_output.cpu()
        
        if single_sentence:
            embeddings = embeddings[0]
        
        if return_numpy:
            return embeddings.numpy()
        return embeddings
    
    def similarity(self, queries: Union[str, List[str]], 
                    keys: Union[str, List[str], ndarray], 
                    device: str = None) -> Union[float, ndarray]:
        
        query_vecs = self.encode(queries, device=device, return_numpy=True) # suppose N queries
        
        if not isinstance(key_vecs, ndarray):
            key_vecs = self.encode(keys, device=device, return_numpy=True) # suppose M keys
        
        # check whether N == 1 or M == 1
        single_query, single_key = len(query_vecs.shape) == 1, len(key_vecs.shape) == 1 
        if single_query:
            query_vecs = query_vecs.reshape(-1,1)
        if single_key:
            key_vecs = key_vecs.reshape(-1,1)
        
        # returns a N*M similarity array
        similarities = cosine_similarity(query_vecs, key_vecs)
        
        if single_query:
            similarities = similarities[0]
            if single_key:
                similarities = float(similarities[0])
        
        return similarities


