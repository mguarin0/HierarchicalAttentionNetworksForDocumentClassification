# Document Classification Comparisons featuring Hierarchical Attention Network

The Hierarchical Attention Network is a novel deep learning architecture that takes advantage of the hierarchical structure of documents to construct a detailed representation of the document. As words form sentences and sentences form the document, the Hierarchical Attention Network’s representation of the document uses this hierarchy in order to determine what sentences and what words in those sentences are most important in the classification of the document as a whole.

<figure>
<img src="lib/imgs/HierarchicalAttentionNetworksDiagram.png" height="1000px" width="1000px" align="center">
<figcaption> Figure 1: Hierarchical Attention Network Architecture Zichao (1) </figcaption>
</figure>

This model uses two levels of LSTM encoders at the word and sentences level in order to build the word and sentence level representations of the document. The attention mechanism is used to attribute importance at the word and sentence level.

There are two applications of the attention mechanism that attend over of the word level encoder and the sentence level encoder. These allow the model to construct a representation of the document that attribute greater levels of importance to key sentences and words throughout the document.

## Results
IMDB(filtered subset) classification

| Hierarchical Attention Network with GRU Encoders as described by [Zichao](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf) | training

## IMDB Dataset
All experiments were performed on the Stanford IMDB dataset which is a natural language dataset where movie reviews have labels that describe the sentiment of the movie review. There are 8 different classes that describe the sentiment from 0-3 for negative sentiment to 6-10 for positive sentiment, which are mapped down to negative sentiment 0 and positive sentiment 1.

## Files in this repo
* Hierarchical Attention Networks with LSTM cells: [han.py](src/han.py)
* IMDB data preprocessing: [dataProcessing.py](src/dataProcessing.py) other scripts will call this is break down downloaded IMDB data set
* Paths shared throughout files: [utils.py](src/utils.py)

## To run the experiments contained in this repo
* HAN (Hierarchical Attention Network with GRU cells for encoders) in Tensorflow can be trained with `python han_master.py --run_type train`, evaluation is performed with `python han_master.py --run_type test`

## References
1) Zichao, Yang. [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf) 25 Aug. 2017.
2) Jozefowicz, Rafal. [An Empirical Exploration of Recurrent Network Architectures](http://proceedings.mlr.press/v37/jozefowicz15.pdf) Accessed 25 Aug. 2017.
3) Sutskever, Llya. [Sequence to Sequence Learning with Neural Networks](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) Accessed 25 Aug. 2017.
4) Kim, Yoon. [Convolutional Neural Networks for Sentence Classification](http://www.aclweb.org/anthology/D14-1181) Accessed 25 Aug. 2017.
5) Zhou, Peng. [Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification](http://aclweb.org/anthology/P/P16/P16-2034.pdf) Accessed 25 Aug. 2017.
6) Goku, Mohandas. [Interpretability via attentional and memory-based interfaces, using TensorFlow](https://www.oreilly.com/ideas/interpretability-via-attentional-and-memory-based-interfaces-using-tensorflow) Accessed 25 Aug. 2017.
7) Pappas, Nikolaos. [Multilingual Hierarchical Attention Networks for Document Classification](https://arxiv.org/pdf/1707.00896.pdf) Accessed 25 Aug. 2017.
8) Wang, Yilin. [Hierarchical Attention Network for Action Recognition in Videos](https://arxiv.org/pdf/1607.06416.pdf) Accessed 25 Aug. 2017.
9) Seo, Paul Hongsuck. [Progressive Attention Networks for Visual Attribute Prediction](https://arxiv.org/pdf/1606.02393.pdf) Accessed 25 Aug. 2017.
