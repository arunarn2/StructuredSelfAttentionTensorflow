# Structured Self Attention - Tensorflow implementation
This repository contains the implementation for the paper [A Structured Self-Attentive Sentence Embedding](https://arxiv.org/abs/1703.03130), which is published in ICLR 2017.

Binary classification on the IMDB Dataset from Keras and multiclass classification on the AGNews Dataset

Using the pretrained glove embeddings (glove.6B.300d.txt). Download the Glove Embeddings from [here](http://nlp.stanford.edu/data/glove.6B.zip) and place it in the glove directory

Implementation Details:
1. Binary classification on IMDB Dataset and Muticlass classification on AGNews Dataset using self attention
2. Regularization using Frobenius norm as described in the paper.
3. Model parameters are defined in `model_params.json` and configuration parameters in `config.json`.
