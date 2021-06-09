import stanza
import nltk
import logging

# download nltk_data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
# download stanza models
stanza.download('en', package='mimic', processors={'ner': 'i2b2'})
print()
logging.warning("Please download pretrained embeddings and models from Google Drive (provided in github) in your first time!")