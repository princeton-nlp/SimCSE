#!/bin/bash

# This example shows how to run the flask demo of SimCSE

python flaskdemo.py \
    --model_name_or_path princeton-nlp/sup-simcse-bert-base-uncased \
    --sentences_dir ./static/ \
    --example_query example_query.txt \
    --example_sentences example_sentence.txt