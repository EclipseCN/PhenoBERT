# PhenoBERT
 ![logo - Copy](C:\Users\56909\Desktop\学位论文\总版\logo - Copy.jpg) 

A novel tool for human clinical disease phenotype recognizing with deep learning.



### What is PhenoBERT?

PhenoBERT is a method that uses advanced deep learning methods (i.e. convolutional neural networks and BERT) to identify clinical disease phenotypes from free text. Currently, only English text is supported. Compared with other methods in the expert-annotated test set, PhenoBERT has reached SOTA effect.



In GSC+(Lobo et al., 2017) dataset:

| **Method**          | Precision |Recall|F1-score| **Set  Similarity** |
| ------------------- | -------------- | -------------- | -------------- | ------------- |
| **NCBO  Annotator** | 75.22          | 49.05               | 59.38         | 72.22         |
| **NeuralCR**        | 72.51          | 58.71               | 64.88         | 78.93         |
| **Clinphen**        | 55.26          | 38.13               | 45.12         | 55.86         |
| **MetaMapLite**     | 67.21          | 44.25               | 53.37         | 67.07         |
| **PhenoPro-NLP**    | 14.47          | 53.32               | 22.76         | 34.64         |
| **Ours**            | **77.97**      | **60.70**           | **68.26** | **82.41** |



### How to install PhenoBERT

#### pip

PhenoBERT supports Python 3.6 or later. We recommend that you install PhenoBERT via **pip**, the Python package manager. To install, simply run:

```shell
pip install phenobert
```

This should also help resolve all of the dependencies of PhenoBERT, for instance [PyTorch](https://pytorch.org/) 1.3.0 or above.

#### From Source



