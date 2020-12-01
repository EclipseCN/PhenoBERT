# PhenoBERT
 ![logo](https://github.com/EclipseCN/PhenoBERT/blob/main/phenobert/img/logo.gif) 

A novel tool for human clinical disease phenotype recognizing with deep learning.

[![Build Status](https://travis-ci.com/EclipseCN/PhenoBERT.svg?branch=main)](https://travis-ci.com/EclipseCN/PhenoBERT) ![Python](https://img.shields.io/badge/python->=3.6-blue)

### What is PhenoBERT?

PhenoBERT is a method that uses advanced deep learning methods (i.e. [convolutional neural networks](https://en.wikipedia.org/wiki/Convolutional_neural_network) and [BERT](https://en.wikipedia.org/wiki/BERT_(language_model))) to identify clinical disease phenotypes from free text. Currently, only English text is supported. Compared with other methods in the expert-annotated test set, PhenoBERT has reached SOTA effect.



In GSC+(Lobo et al., 2017) dataset:

| **Method**          | Precision |Recall|F1-score| **Set  Similarity** |
| ------------------- | -------------- | -------------- | -------------- | ------------- |
| NCBO  Annotator | 75.22          | 49.05               | 59.38         | 72.22         |
| NeuralCR        | 72.51          | 58.71               | 64.88         | 78.93         |
| Clinphen        | 55.26          | 38.13               | 45.12         | 55.86         |
| MetaMapLite     | 67.21          | 44.25               | 53.37         | 67.07         |
| PhenoPro-NLP    | 14.47          | 53.32               | 22.76         | 34.64         |
| **PhenoBERT**    | **77.97**      | **60.70**           | **68.26** | **82.41** |



### Citation:

coming soon



### How to install PhenoBERT

You can use PhenoBERT on your local machine. Due to some inevitable reason, the web version of PhenoBERT is not yet available.

#### From Source

1. Download total project from github.

```shell
git clone https://github.com/EclipseCN/PhenoBERT.git
```

2. Enter the project main directory.

```she
cd PhenoBERT
```

3. Install dependencies in the current Python3 environment.

   Notice: we recommend using Python virtual environment to avoid confusion.

```shell
pip install -r requirements.txt
python setup.py
```

4. Move the pretrained files into the corresponding folder.
```shell
# download files from Google Drive in advance
mv /path/to/download/embeddings/* phenobert/embeddings
mv /path/to/download/models/* phenobert/models
```



### Pretrained embeddings and models

We have prepared pre-trained [fastText](https://en.wikipedia.org/wiki/FastText) and BERT embeddings and model files with .pkl suffix on [Google Drive](https://drive.google.com/) for downloading.

| Directory Name | File Name | Description |
| ---- | ------ | -------|
| models/ | [HPOModel_H/](https://drive.google.com/drive/folders/1NriTyBqh3kxUWv1lrnYjWBpYu0F0hrCh?usp=sharing) | CNN hierarchical model file |
|  | [bert_model_max_triple.pkl](https://drive.google.com/file/d/1AwRnaB5RruFUEdMkKohZmTlD4ILCkQ_z/view?usp=sharing) | BERT model file |
| embeddings/ | [biobert_v1.1_pubmed/](https://drive.google.com/drive/folders/10lko9BpToUl3PlUWrYbFmNyVHxDX1xby?usp=sharing) | BERT embedding obtained from [BioBERT](https://github.com/dmis-lab/biobert) |
| | [fasttext_pubmed.bin](https://drive.google.com/file/d/1GFB3I46B50sDUHcSpu84jZKqJnIjc--B/view?usp=sharing) | fastText embedding trained on [pubmed](https://en.wikipedia.org/wiki/PubMed) |

Once the download is complete, please put it in the corresponding folder for PhenoBERT to load.



### How to use PhenoBERT?

We provide three ways to use PhenoBERT. Due to this [issue](https://github.com/pytorch/pytorch/issues/18325), all calls need to be in the `phenobert/utils` path.

```shell
cd phenobert/utils
```



#### Annotate corpus folder

The most common usage is recognizing human clinical disease phenotype from free text. 

Giving a set of text files, PhenoBERT will then annotate each of the text files and generate an annotation file with the same name in the target folder.

Example use `annotate.py `:

```shell
python annotate.py -i DIR_IN -o DIR_OUT
```

Arguments: 

```shell
[Required]

 -i directory for storing text files
 -o directory for storing annotation files
 
[Optional]

 -p1 parameter for CNN model [0.8]
 -p2 parameter for CNN model [0.6]
 -p3 parameter for BERT model [0.9]
 -nb flag for not use BERT
 -t  cpu threads for calculation [10]
```



#### Related API

We also provide some APIs for other programs to integrate.

```python
from api import *
```

Running the above code will import related functions and related models, and temporarily store them as global variables for quick and repeated calls. Or you can simply use Python interactive shell.

Currently we have integrated the following functions:

1. annotate directly from String

```python
print(annotate_text("I have a headache"))
```

Output:

```shell
9       17      headache        HP:0002315
```

Notice: use `output = path/`can redirect output to specified file

2. get the approximate location of the disease

```python
print(get_L1_HPO_term(["cardiac hypertrophy", "renal disease"]))
```

Output:

```shell
[['cardiac hypertrophy', {'HP:0001626'}], ['renal disease', {'HP:0000119'}]]
```

3. get most similar HPO terms.

```python
print(get_most_related_HPO_term(["cardiac hypertrophy", "renal disease"]))
```

Output:

```shell
[['cardiac hypertrophy', ['HP:0001714', 'HP:0031319', 'HP:0001627']], ['renal disease', ['HP:0000112', 'HP:0012211', 'HP:0000077']]]
```

4. determine if two phrases match

```python
print(is_phrase_match_BERT("cardiac hypertrophy", "Ventricular hypertrophy"))
```

Output:

```shell
Match
```



#### GUI application



### Dataset

We provide here two corpus with annotations used in the evaluation (`phenobert/data`), which are currently publicly available due to privacy processing.

| Dataset     | Num  | Description                                                  |
| ----------- | ---- | ------------------------------------------------------------ |
| GSC+        | 228  | Composed of 228 abstracts of biomedical literature (Lobo et al., 2017) |
| 68_clinical | 68   | Study the clinical presentation of 68 real cases of intellectual disability (Anazi et al., 2017) |



### Train your own model

For the convenience of some users who cannot log in to Google Drive or who want to customize training process for their selves.

We provide the training Python script and training set used by PhenoBERT. Of course, the training set can be customized by the user to generate specific models for other purposes.

```shell
cd phenobert/utils

# produce trained models for CNN model
python train.py
python train_sub.py

# produce trained models for BERT model
python my_bert_match.py
```

