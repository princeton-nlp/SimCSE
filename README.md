# SimCSE: Simple Contrastive Learning of Sentence Embeddings


****** **New** ******

Thanks for your interest in our repo! 

Probably you will think this as another *"empty"* repo of a preprint paper 🥱. 

Wait a minute! The authors are working day and night 💪, to make the code and models available, so you can explore our state-of-the-art sentence embeddings. 

We anticipate the code will be out * **in one week** *. 

Please watch👀 us and stay tuned!

4/18: We released our paper. Check it out!

****** **End new information** ******

This is the codebase for the paper [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://gaotianyu1350.github.io/assets/simcse/simcse.pdf). We propose a simple contrastive learning framework that works with both unlabeled and labeled data. Unsupervised SimCSE simply takes an input sentence and predicts itself in a contrastive learning framework, with only standard dropout used as noise. Our supervised SimCSE incorporates annotated pairs from NLI datasets into contrastive learning by using `entailment` pairs as positives and `contradiction` pairs as hard negatives. The following figure is an illustration of our models.

![](figure/model.png)
