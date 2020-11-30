import os
import torch
import stanza
import fasttext
import warnings
from util import process_text2phrases, annotate_phrases, ModelLoader, HPOTree, PhraseDataSet4predict, PhraseDataSet4predictFunc
from util import cnn_model_path, bert_model_path, fasttext_model_path
from model import device
from torch.utils.data import DataLoader

# ignore useless warnings
warnings.simplefilter("ignore", torch.serialization.SourceChangeWarning)

clinical_ner_model = stanza.Pipeline('en', package='mimic', processors={'ner': 'i2b2'}, verbose=False)
loader = ModelLoader()
hpo_tree = HPOTree()
hpo_tree.buildHPOTree()
cnn_model = loader.load_all(cnn_model_path)
bert_model = loader.load_all(bert_model_path)
fasttext_model = fasttext.load_model(fasttext_model_path)


def set_thread_num(n_threads):
    """
    设置cpu reference时的线程上限
    :param n_threads: 线程数
    :return:
    """
    os.environ['MKL_NUMTHREADS'] = n_threads
    os.environ['OMP_NUMTHREADS'] = n_threads


def annotate_text(text, output=None, param1=0.8, param2=0.6, param3=0.9, use_step_3=True):
    """
    注释自由文本的api
    :param text: 自由文本
    :param output: 可指定输出字符串还是输出到文本文件
    :param param1: 阈值1
    :param param2: 阈值2
    :param param3: 阈值3
    :param use_step_3: 是否使用BERT进行refine
    :return:
    """
    phrases_list = process_text2phrases(text, clinical_ner_model)
    result = annotate_phrases(text, phrases_list, hpo_tree, fasttext_model, cnn_model, bert_model,
                              output, device, param1, param2, param3,
                              use_step_3)
    return result


def get_L1_HPO_term(phrases_list, param1=0.8):
    """
    给定短语列表，得到对应的L1层HPO节点，即大约的发病位置
    :param phrases_list: 短语列表
    :param param1: 阈值1
    :return:
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
            # 挑选每个短语超过阈值的L1层的HPO，超过0.9的我们才认为是预测正确的L1层
            Candidate_hpos = [
                set([hpo_tree.getIdx2HPO_l1(prediction[idx1][idx2]) for idx2 in range(len(prediction[idx1])) if
                     scores_p[idx1][idx2] >= param1]) for idx1 in range(len(prediction))]
            for phrase_item, j in zip(phrase_items, Candidate_hpos):
                res.append([phrase_item, j])
    return res


def get_most_related_HPO_term(phrases_list, param1=0.8, param2=0):
    """
    给定短语列表，给出与这些短语最相似的HPO列表
    :param phrases_list:
    :param param1:
    :param param2:
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
        if len(j) > 0 and "None" not in j:
            Candidate_hpos_sub = []
            for l1_hpo in j:
                l1_idx = hpo_tree.getHPO2idx_l1(l1_hpo)
                y_sub = total_model[l1_idx](PhraseDataSet4predictFunc(i, fasttext_model)).squeeze()
                prediction_sub = y_sub.topk(5)[1].tolist()
                scores_p_sub = torch.softmax(y_sub, dim=0).topk(5)[0].tolist()
                Candidate_hpos_sub.extend(
                    [[total_idx2hpo[l1_idx][prediction_sub[idx]], scores_p_sub[idx]] for idx in
                     range(len(prediction_sub)) if scores_p_sub[idx] >= param2])
            Candidate_hpos_sub.sort(key=lambda x: x[1], reverse=True)
            if len(Candidate_hpos_sub) != 0:
                res.append([i, [i[0] for i in Candidate_hpos_sub[:5]]])
            else:
                res.append([i, "None"])
    return res


def is_phrase_match_BERT(phrase1, phrase2):
    """
    判断两个短语是否匹配
    :param phrase1: 短语1
    :param phrase2: 短语2
    :return:
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