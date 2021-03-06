import os
import sys
import torch
from util import HPOTree, process_text2phrases, annotate_phrases, fasttext_model_path, ModelLoader
from model import device
import time
import fasttext
import stanza
import argparse
import warnings
from tqdm import tqdm



def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Annotate HPO terms from free text.")
    parser.add_argument('-i', '--dir-in', required=True, help='Input directory path')
    parser.add_argument('-p1', '--param1', required=False, default=0.8, type=float, help='Model param 1')
    parser.add_argument('-p2', '--param2', required=False, default=0.6, type=float, help='Model param 2')
    parser.add_argument('-p3', '--param3', required=False, default=0.9, type=float, help='Model param 3')
    parser.add_argument('-al', '--all', required=False, action="store_true", help='Not filter overlapping concept')
    parser.add_argument('-nb', '--no-bert', required=False, action="store_true", help='Not use bert')
    parser.add_argument('-o', '--dir-out', required=True, help='Output directory path')
    parser.add_argument('-t', '--n-threads', required=False, default="10", help='PyTorch cpu threads limits')

    return parser.parse_args(argv)



args = parse_arguments(sys.argv[1:])

# Environment Option
os.environ['MKL_NUMTHREADS'] = args.n_threads
os.environ['OMP_NUMTHREADS'] = args.n_threads

# ignore useless warnings
warnings.simplefilter("ignore", torch.serialization.SourceChangeWarning)

cnn_model_path = "../models/HPOModel_H/model_layer1.pkl"
bert_model_path = "../models/bert_model_max_triple.pkl"

hpo_tree=HPOTree()
hpo_tree.buildHPOTree()
fasttext_model = fasttext.load_model(fasttext_model_path)
loader = ModelLoader()

clinical_ner_model = stanza.Pipeline('en', package='mimic', processors={'ner': 'i2b2'}, verbose=False)
cnn_model = loader.load_all(cnn_model_path)
bert_model = loader.load_all(bert_model_path)

corpus_dir_path=args.dir_in
predict_dir_path=args.dir_out
if not os.path.exists(predict_dir_path):
    os.makedirs(predict_dir_path)

file_list=os.listdir(corpus_dir_path)
# file_list=["15734008"]
t0 = time.time()
for file_name in tqdm(file_list, ncols=10):
    predict_file_path=os.path.join(predict_dir_path, file_name)
    with open(os.path.join(corpus_dir_path, file_name), "r", encoding="utf-8") as text_file:
        text=text_file.read()
        phrases_list = process_text2phrases(text, clinical_ner_model)
        annotate_phrases(text, phrases_list, hpo_tree, fasttext_model, cnn_model, bert_model,
                      predict_file_path, device, param1=args.param1, param2=args.param2, param3=args.param3, use_longest=not args.all, use_step_3=not args.no_bert)

print("Complete")
t1 = time.time()
print(f"Use {(t1-t0)/len(file_list)} seconds per file.")

