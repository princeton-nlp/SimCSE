import logging
import numpy as np
from numpy import ndarray
import torch
from torch import Tensor, device
import transformers
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from typing import List, Dict, Tuple, Type, Union

logger = logging.getLogger(__name__)

"""
A simple class used for several downstream zero-shot sentence embedding tasks:
"""
class SentenceEmbedder(object):
    def __init__(self, model_name_or_path: str, 
                device: str = None,
                num_cells: int = 100,
                num_cells_in_search: int = 10):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.index = None
        self.is_faiss_index = False
        self.num_cells = num_cells
        self.num_cells_in_search = num_cells_in_search
    
    def encode(self, sentence: Union[str, List[str]], 
                device: str = None, 
                return_numpy: bool = False,
                normalize: bool = False,
                keep_dim: bool = False) -> Union[ndarray, Tensor]:

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
        
        if normalize:
            embeddings = normalize(embeddings, axis=1)
            if not return_numpy:
                embeddings = Tensor(embeddings)

        if single_sentence and not keep_dim:
            embeddings = embeddings[0]
        
        if return_numpy:
            return embeddings.numpy()
        return embeddings
    
    def similarity(self, queries: Union[str, List[str]], 
                    keys: Union[str, List[str], ndarray], 
                    device: str = None) -> Union[float, ndarray]:
        
        query_vecs = self.encode(queries, device=device, return_numpy=True) # suppose N queries
        
        if not isinstance(keys, ndarray):
            key_vecs = self.encode(keys, device=device, return_numpy=True) # suppose M keys
        else:
            key_vecs = keys

        # check whether N == 1 or M == 1
        single_query, single_key = len(query_vecs.shape) == 1, len(key_vecs.shape) == 1 
        if single_query:
            query_vecs = query_vecs.reshape(1,-1)
        if single_key:
            key_vecs = key_vecs.reshape(1,-1)
        
        # returns a N*M similarity array
        similarities = cosine_similarity(query_vecs, key_vecs)
        
        if single_query:
            similarities = similarities[0]
            if single_key:
                similarities = float(similarities[0])
        
        return similarities
    
    def build_index(self, sentences_or_file_path: Union[str, List[str]], 
                        use_faiss: bool = None,
                        device: str = None):

        if use_faiss is None or use_faiss == True:
            try:
                import faiss
                use_faiss = True 
            except:
                logger.warning("Fail to import faiss, try to use exact search")
                use_faiss = False
        
        # if the input sentence is a string, we assume it's the path of file that stores various sentences
        if isinstance(sentences_or_file_path, str):
            sentences = []
            with open(sentences_or_file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    sentences.append(line.strip())
            sentences_or_file_path = sentences
        
        embeddings = self.encode(sentences_or_file_path, device=device, return_numpy=True, normalize=use_faiss)

        id2sentence = {}
        for i, sentence in enumerate(sentences_or_file_path):
            id2sentence[i] = sentence
        self.index = {"id2sentence": id2sentence}
        
        if use_faiss:
            quantizer = faiss.IndexFlatL2(embeddings.shape[1])  
            index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], self.num_cells) 
            index.train(embeddings)
            index.add(embeddings)
            index.nprobe = self.num_cells_in_search
            self.index["index"] = index
            self.is_faiss_index = True
        else:
            self.index["index"] = embeddings
            self.is_faiss_index = False
    
    def search(self, queries: Union[str, List[str]], 
                device: str = None, 
                normalize: bool = False,
                threshold: float = 0.6,
                top_k: int = 5) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
        
        if not self.is_faiss_index:
            if isinstance(queries, list):
                combined_results = []
                for query in queries:
                    results = self.search(query, device, normalize)
                    combined_results.append(results)
                return combined_results
            
            similarities = self.similarity(queries, self.index["index"]).tolist()
            id_and_score = []
            for i, s in enumerate(similarities):
                if s >= threshold:
                    id_and_score.append((i, s))
            id_and_score = sorted(id_and_score, key=lambda x: x[1], reverse=True)[:top_k]
            results = [(self.index["id2sentence"][idx], score) for idx, score in id_and_score]
            return results
        else:
            query_vecs = self.encode(queries, device=device, normalize=normalize, keep_dim=True, return_numpy=True)

            distance, idx = self.index["index"].search(query_vecs, top_k)
            
            def pack_single_result(dist, idx):
                score = [1.0 - d / 2.0 for d in dist]
                results  = [(self.index["id2sentence"][i], s) for i, s in zip(idx, score)]
                return results
            
            if isinstance(queries, list):
                combined_results = []
                for i in range(len(queries)):
                    results = pack_single_result(distance[i], idx[i])
                    combined_results.append(results)
                return combined_results
            else:
                return pack_single_result(distance[0], idx[0])

if __name__=="__main__":
    example_sentences = [
        'an animal is biting a persons finger .',
        'a woman is reading .',
        'a man is lifting weights in a garage .',
        'a man plays the violin .',
        'a man is eating food .',
        'a man plays the piano .',
        'a panda is climbing .',
        'a man plays a guitar .',
        'a woman is slicing a meat .',
        'a woman is taking a picture .'
    ]
    example_queries = [
        'a man is playing music',
        'a woman is making a photo'
    ]

    model_name = "princeton-nlp/sup-simcse-roberta-base"
    embedder = SentenceEmbedder(model_name)

    print("\n=========Calculate cosine similarities between queries and sentences============\n")
    similarities = embedder.similarity(example_queries, example_sentences)
    print(similarities)

    print("\n=========Naive exact search============\n")
    embedder.build_index(example_sentences)
    results = embedder.search(example_queries)
    for i, result in enumerate(results):
        print("retrieval results for query: {}".format(example_queries[i]))
        for sentence, score in result:
            print("{}  (cosine similarity: {:.4f})".format(sentence, score))
    
    print("\n=========Search with Faiss backend============\n")
    embedder.build_index(example_sentences, use_faiss=True)
    results = embedder.search(example_queries)
    for i, result in enumerate(results):
        print("retrieval results for query: {}".format(example_queries[i]))
        for sentence, score in result:
            print("{}  (cosine similarity: {:.4f})".format(sentence, score))







