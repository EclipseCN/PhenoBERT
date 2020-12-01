import stanza
import nltk
import logging

nltk.download('stopwords')
stanza.download('en', package='mimic', processors={'ner': 'i2b2'})
print()
logging.warning("Please download pretrained embeddings and models from Google Drive (provided in github) in your first time!")