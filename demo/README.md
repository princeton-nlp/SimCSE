## Demo of SimCSE 
Several demos are available for people to play with our pre-trained SimCSE.

### Flask Demo
<div align="center">
<img src="../figure/demo.gif" width="750">
</div>

We provide a simple Web demo based on [flask](https://github.com/pallets/flask) to show how SimCSE can be directly used for information retrieval. The code is based on [DensePhrases](https://arxiv.org/abs/2012.12624)' [repo](https://github.com/princeton-nlp/DensePhrases) and [demo](http://densephrases.korea.ac.kr) (a lot of thanks to the authors of DensePhrases). To run this flask demo locally, make sure the SimCSE inference interfaces are setup:
```bash
git clone https://github.com/princeton-nlp/SimCSE
cd SimCSE
python setup.py develop
```
Then you can use `run_demo_example.sh` to launch the demo. As a default setting, we build the index for 1000 sentences sampled from STS-B dataset. Feel free to build the index of your own corpora. You can also install [faiss](https://github.com/facebookresearch/faiss) to speed up the retrieval process.

### Gradio Demo
[AK391](https://github.com/AK391) has provided a [Gradio Web Demo](https://gradio.app/g/AK391/SimCSE) of SimCSE to show how the pre-trained models can predict the semantic similarity between two sentences.
