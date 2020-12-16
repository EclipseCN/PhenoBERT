import os
import torch
import stanza
import fasttext
import warnings
from util import process_text2phrases, annotate_phrases, ModelLoader, HPOTree, PhraseDataSet4predict, PhraseDataSet4predictFunc, produceCandidateTriple
from util import cnn_model_path, bert_model_path, fasttext_model_path
from model import device
from torch.utils.data import DataLoader

# ignore useless warnings
warnings.simplefilter("ignore", torch.serialization.SourceChangeWarning)

os.environ['MKL_NUMTHREADS'] = "10"
os.environ['OMP_NUMTHREADS'] = "10"

clinical_ner_model = stanza.Pipeline('en', package='mimic', processors={'ner': 'i2b2'}, verbose=False)
loader = ModelLoader()
hpo_tree = HPOTree()
hpo_tree.buildHPOTree()
cnn_model = loader.load_all(cnn_model_path)
bert_model = loader.load_all(bert_model_path)
fasttext_model = fasttext.load_model(fasttext_model_path)



def annotate_text(text, output=None, param1=0.8, param2=0.6, param3=0.9, use_step_3=True):
    """
    Annotate free text api.
    :param text: free text
    :param output: output str or file, default for str
    """
    phrases_list = process_text2phrases(text, clinical_ner_model)
    result = annotate_phrases(text, phrases_list, hpo_tree, fasttext_model, cnn_model, bert_model,
                              output, device, param1, param2, param3,
                              use_step_3)
    return result


def get_L1_HPO_term(phrases_list, param1=0.8):
    """
    Given a list of phrases, get the corresponding HPO term of the L1 layer, which is the approximate location of the disease.
    :param phrases_list: list of phrases
    """
    dataset = PhraseDataSet4predict(phrases_list, fasttext_model)
    dataloader = DataLoader(dataset, batch_size=20, shuffle=False)
    samples = 0
    res = []
    with torch.no_grad():
        cnn_model.eval()
        for data in dataloader:
            input_data = [data["data"].float().to(device), data["seq_len"].int().to(device)]
            batch_num = input_data[0].size(0)
            y = cnn_model(input_data)
            phrase_items = dataset.phrase_list[samples:samples + batch_num]
            samples += batch_num
            prediction = y.argsort().tolist()
            scores_p = y.sort()[0]
            Candidate_hpos = [
                set([hpo_tree.getIdx2HPO_l1(prediction[idx1][idx2]) for idx2 in range(len(prediction[idx1])) if
                     scores_p[idx1][idx2] >= param1]) for idx1 in range(len(prediction))]
            for phrase_item, j in zip(phrase_items, Candidate_hpos):
                res.append([phrase_item, j])
    return res


def get_most_related_HPO_term(phrases_list, param1=0.8, param2=0.6, param3=0.9):
    """
    给定短语列表，给出与这些短语最相似的HPO列表
    :param phrases_list:
    :param param1:
    :param param2:
    :param param3:
    :return:
    """
    first_step = get_L1_HPO_term(phrases_list, param1)
    res = []
    total_hpo_list = []
    total_hpo2idx = []
    total_idx2hpo = []
    total_model = []
    total_l1_root = hpo_tree.layer1
    for sub_l1_root in total_l1_root:
        # get idx2hpo hpo2idx for given HPO
        root_idx, hpo_list, n_concept, hpo2idx, idx2hpo = hpo_tree.getMaterial4L1(sub_l1_root)
        total_hpo_list.append(hpo_list)
        total_hpo2idx.append(hpo2idx)
        total_idx2hpo.append(idx2hpo)
        sub_model_save_path = f"../models/HPOModel_H/model_l1_{root_idx}.pkl"
        loader = ModelLoader()
        sub_model = loader.load_all(sub_model_save_path)
        sub_model.eval()
        total_model.append(sub_model)
    for i, j in first_step:
        flag = True
        if len(j) > 0 and "None" not in j:
            Candidate_hpos_sub = set()
            for l1_hpo in j:
                l1_idx = hpo_tree.getHPO2idx_l1(l1_hpo)
                y_sub = total_model[l1_idx](PhraseDataSet4predictFunc(i, fasttext_model)).squeeze()
                if y_sub.size(0) > 10:
                    prediction_sub = y_sub.topk(10)[1].tolist()
                    scores_p_sub = torch.softmax(y_sub, dim=0).topk(10)[0].tolist()
                else:
                    prediction_sub = y_sub.topk(y_sub.size(0))[1].tolist()
                    scores_p_sub = torch.softmax(y_sub, dim=0).topk(y_sub.size(0))[0].tolist()
                Candidate_hpos_sub.update(
                    [total_idx2hpo[l1_idx][prediction_sub[idx]] for idx in range(len(prediction_sub)) if
                     scores_p_sub[idx] >= param2])
            if len(Candidate_hpos_sub) != 0 and "None" not in Candidate_hpos_sub:
                Candidate_hpos_sub = list(Candidate_hpos_sub)
                candidate_phrase = [hpo_tree.getNameByHPO(item) for item in Candidate_hpos_sub]
                raw_phrase = i
                # print(raw_phrase, candidate_phrase)
                ans_hpo, score, class_num = produceCandidateTriple(raw_phrase, candidate_phrase, bert_model, hpo_tree,
                                                                   Candidate_hpos_sub, param3)
                if ans_hpo != "None":
                    res.append([i, ans_hpo])
                    flag = False
        if flag:
            res.append([i, "None"])
    return res


def is_phrase_match_BERT(phrase1, phrase2):
    """
    Determine if two phrases match
    :param phrase1: phrase1
    :param phrase2: phrase2
    """
    from fastNLP import DataSetIter, DataSet
    from fastNLP.core.utils import _move_dict_value_to_device
    from my_bert_match import addWords, addWordPiece, processItem, processNum, addSeqlen
    # 0 for not match,1 for match
    testset = DataSet({"raw_words": [f"{phrase1}::{phrase2}"]})
    testset.apply(addWords, new_field_name="p_words")
    testset.apply(addWordPiece, new_field_name="t_words")
    testset.apply(processItem, new_field_name="word_pieces")
    testset.apply(processNum, new_field_name="word_nums")
    testset.apply(addSeqlen, new_field_name="seq_len")
    testset.field_arrays["word_pieces"].is_input = True
    testset.field_arrays["seq_len"].is_input = True
    testset.field_arrays["word_nums"].is_input = True
    # print(testset)
    with torch.no_grad():
        bert_model.eval()
        test_batch = DataSetIter(batch_size=1, dataset=testset, sampler=None)
        outputs = []
        for batch_x, batch_y in test_batch:
            _move_dict_value_to_device(batch_x, batch_y, device=device)
            outputs.append(bert_model.forward(batch_x["word_pieces"], batch_x["word_nums"], batch_x["seq_len"])['pred'])
        outputs = torch.cat(outputs)
        outputs = torch.nn.functional.softmax(outputs, dim=1)
        return ["Not Match", "Related", "Match"][outputs.argmax().item()]
