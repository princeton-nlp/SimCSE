import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
import gradio as gr

# Import our models. The package will take care of downloading the models automatically
tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")

def simcse(text1, text2, text3):
    # Tokenize input texts
    texts = [
        text1,
        text2,
        text3
    ]
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # Get the embeddings
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

    # Calculate cosine similarities
    # Cosine similarities are in [-1, 1]. Higher means more similar
    cosine_sim_0_1 = 1 - cosine(embeddings[0], embeddings[1])
    cosine_sim_0_2 = 1 - cosine(embeddings[0], embeddings[2])
    return {"cosine similarity":cosine_sim_0_1}, {"cosine similarity":cosine_sim_0_2}


inputs = [
          gr.inputs.Textbox(lines=5, label="Input Text One"),
          gr.inputs.Textbox(lines=5, label="Input Text Two"),
          gr.inputs.Textbox(lines=5, label="Input Text Three")
]

outputs = [
            gr.outputs.Label(type="confidences",label="Cosine similarity between text one and two"),
            gr.outputs.Label(type="confidences", label="Cosine similarity between text one and three")
]


title = "SimCSE"
description = "demo for Princeton-NLP SimCSE. To use it, simply add your text, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2104.08821'>SimCSE: Simple Contrastive Learning of Sentence Embeddings</a> | <a href='https://github.com/princeton-nlp/SimCSE'>Github Repo</a></p>"
examples = [
    ["There's a kid on a skateboard.",
    "A kid is skateboarding.",
    "A kid is inside the house."]
]

gr.Interface(simcse, inputs, outputs, title=title, description=description, article=article, examples=examples).launch()