import torch.nn
from fastNLP.io.loader import CSVLoader
from fastNLP import Vocabulary
from string import punctuation
from fastNLP.core.utils import _move_model_to_device
from fastNLP.core.utils import _move_dict_value_to_device
from fastNLP.core.utils import _get_model_device
from fastNLP.core.metrics import MetricBase
from fastNLP.embeddings.My_bert_embedding import BertEmbedding

class PRMetric(MetricBase):
    def __init__(self):
        super().__init__()

        # 根据你的情况自定义指标
        self.norm = 0
        self.denorm_pr = 0
        self.denorm_re = 0

    def evaluate(self, pred, target): # 这里的名称需要和dataset中target field与model返回的key是一样的，不然找不到对应的value
        # dev或test时，每个batch结束会调用一次该方法，需要实现如何根据每个batch累加metric
        self.denorm_re += target.eq(1).sum().item()
        self.denorm_pr += pred.eq(1).sum().item()
        for i in range(pred.size(0)):
            if pred[i]==1 and target[i]==1:
                self.norm+=1

    def get_metric(self, reset=True): # 在这里定义如何计算metric
        pr = self.norm/self.denorm_pr
        re = self.norm/self.denorm_re
        if reset: # 是否清零以便重新计算
            self.norm = 0
            self.denorm_pr = 0
            self.denorm_re = 0
        return {'pr': pr,'re': re} # 需要返回一个dict，key为该metric的名称，该名称会显示到Trainer的progress bar中

data_set_loader = CSVLoader(
    headers=('raw_words', 't_target'), sep='\t'
)

embed = BertEmbedding(Vocabulary(), model_dir_or_name='../embeddings/biobert_v1.1_pubmed',include_cls_sep=True, pool_method="max")

def addWordPiece(instance):
    sentence=instance["p_words"]
    results=[]
    for word in sentence:
        results.append(embed.model.tokenzier.convert_tokens_to_ids(embed.model.tokenzier.wordpiece_tokenizer.tokenize(word)))
    return results

def addWords(instance):
    sentence = instance["raw_words"].lower()
    inter = sentence.split("::")
    word1 = inter[0]
    word2 = inter[1]
    p_word1 = ""
    for c in word1:
        if c in punctuation:
            p_word1 += " "
        else:
            p_word1 += c
    p_word2 = ""
    for c in word2:
        if c in punctuation:
            p_word2 += " "
        else:
            p_word2 += c
    words = p_word1 + " [SEP] " + p_word2
    return words.split()

def addSeqlen(instance):
    words = instance["p_words"]
    return len(words)

def processNum(instance):
    inter=instance["t_words"]
    results=[len(item) for item in inter]
    return results

def processItem(instance):
    inter = instance["t_words"]
    results=[]
    for item in inter:
        results.extend(item)
    return results

def processTarget(instance):
    target=instance["t_target"]
    return int(target)

def train():
    n_epochs=10
    train_set = data_set_loader._load('../models/all4bert_new_triple_2.txt')
    train_set,tmp_set=train_set.split(0.2)
    val_set,test_set=tmp_set.split(0.5)
    data_bundle=[train_set,val_set,test_set]

    for dataset in data_bundle:
        dataset.apply(addWords,new_field_name="p_words")
        dataset.apply(addWordPiece,new_field_name="t_words")
        dataset.apply(processItem,new_field_name="word_pieces")
        dataset.apply(processNum,new_field_name="word_nums")
        dataset.apply(addSeqlen,new_field_name="seq_len")
        dataset.apply(processTarget,new_field_name="target")

    for dataset in data_bundle:
        dataset.field_arrays["word_pieces"].is_input = True
        dataset.field_arrays["seq_len"].is_input = True
        dataset.field_arrays["word_nums"].is_input=True
        dataset.field_arrays["target"].is_target = True

    print("In total "+str(len(data_bundle))+" datasets:")
    print("Trainset has "+str(len(train_set))+" instances.")
    print("Validateset has "+str(len(val_set))+" instances.")
    print("Testset has "+str(len(test_set))+" instances.")
    train_set.print_field_meta()
    # print(train_set)
    from fastNLP.models.Mybert import BertForSentenceMatching
    from fastNLP import AccuracyMetric,DataSetIter

    from fastNLP.core.utils import _pseudo_tqdm as tqdm
    # 注意这里是表明分的类数
    model = BertForSentenceMatching(embed, 3)
    if torch.cuda.is_available():
        model = _move_model_to_device(model, device=0)
    # print(model)
    train_batch=DataSetIter(batch_size=16,dataset=train_set,sampler=None)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    Lossfunc=torch.nn.CrossEntropyLoss()
    with tqdm(total=n_epochs, postfix='loss:{0:<6.5f}', leave=False, dynamic_ncols=True) as pbar:
        print_every=10
        for epoch in range(1, n_epochs + 1):
            pbar.set_description_str(desc="Epoch {}/{}".format(epoch, n_epochs))
            avg_loss = 0
            step=0
            for batch_x,batch_y in train_batch:
                step+=1
                _move_dict_value_to_device(batch_x, batch_y, device=_get_model_device(model))
                optimizer.zero_grad()
                output = model.forward(batch_x["word_pieces"],batch_x["word_nums"],batch_x["seq_len"])
                loss = Lossfunc(output['pred'], batch_y['target'])
                loss.backward()
                optimizer.step()
                avg_loss+=loss.item()
                if step %print_every ==0:
                    avg_loss=float(avg_loss)/print_every
                    print_output = "[epoch: {:>3} step: {:>4}] train loss: {:>4.6}".format(
                        epoch, step, avg_loss)
                    pbar.update(print_every)
                    pbar.set_postfix_str(print_output)
                    avg_loss=0
            metric = AccuracyMetric()
            val_batch = DataSetIter(batch_size=8, dataset=val_set, sampler=None)
            for batch_x, batch_y in val_batch:
                _move_dict_value_to_device(batch_x, batch_y, device=_get_model_device(model))
                output = model.predict(batch_x["word_pieces"], batch_x["word_nums"], batch_x["seq_len"])
                metric(output, batch_y)
            eval_result = metric.get_metric()
            print("ACC on Validate Set:",eval_result)
            from fastNLP.io import ModelSaver
            saver = ModelSaver("../models/bert_model_max_triple_2.pkl")
            saver.save_pytorch(model, param_only=False)
        pbar.close()
    metric = AccuracyMetric()
    test_batch = DataSetIter(batch_size=8, dataset=test_set, sampler=None)
    for batch_x, batch_y in test_batch:
        _move_dict_value_to_device(batch_x, batch_y, device=_get_model_device(model))
        output = model.predict(batch_x["word_pieces"], batch_x["word_nums"], batch_x["seq_len"])
        metric(output,batch_y)
    eval_result = metric.get_metric()
    print("ACC on Test Set:",eval_result)
    from fastNLP.io import ModelSaver
    saver = ModelSaver("../models/bert_model_max_triple_2.pkl")
    saver.save_pytorch(model, param_only=False)

def test():
    from fastNLP import DataSetIter,DataSet
    # 0 for not match,1 for match
    testset=DataSet({"raw_words":["5::five"]})
    testset.apply(addWords, new_field_name="p_words")
    testset.apply(addWordPiece, new_field_name="t_words")
    testset.apply(processItem, new_field_name="word_pieces")
    testset.apply(processNum, new_field_name="word_nums")
    testset.apply(addSeqlen, new_field_name="seq_len")
    testset.field_arrays["word_pieces"].is_input = True
    testset.field_arrays["seq_len"].is_input = True
    testset.field_arrays["word_nums"].is_input = True
    # print(testset)
    from fastNLP.io import ModelLoader
    loader=ModelLoader()
    if torch.cuda.is_available():
        model = loader.load_pytorch_model("../models/bert_model_max_triple.pkl")
    else:
        model = torch.load("../models/bert_model_max_triple.pkl", map_location="cpu")

    model.eval()
    test_batch = DataSetIter(batch_size=1, dataset=testset, sampler=None)
    outputs=[]
    for batch_x, batch_y in test_batch:
        _move_dict_value_to_device(batch_x, batch_y, device=_get_model_device(model))
        outputs.append(model.forward(batch_x["word_pieces"], batch_x["word_nums"], batch_x["seq_len"])['pred'])
    outputs=torch.cat(outputs)
    outputs=torch.nn.functional.softmax(outputs,dim=1)
    return outputs

def evaluate():
    from fastNLP import DataSetIter
    testset = data_set_loader._load('eval_down.txt')
    testset.apply(addWords, new_field_name="p_words")
    testset.apply(addWordPiece, new_field_name="t_words")
    testset.apply(processItem, new_field_name="word_pieces")
    testset.apply(processNum, new_field_name="word_nums")
    testset.apply(addSeqlen, new_field_name="seq_len")
    testset.apply(processTarget, new_field_name="target")
    testset.field_arrays["word_pieces"].is_input = True
    testset.field_arrays["seq_len"].is_input = True
    testset.field_arrays["word_nums"].is_input = True
    testset.field_arrays["target"].is_target = True
    # print(testset)
    from fastNLP.io import ModelLoader
    loader = ModelLoader()
    model = loader.load_pytorch_model("model_ckpt.pkl")
    metric = PRMetric()
    # from fastNLP import AccuracyMetric
    # metric=AccuracyMetric()
    test_batch = DataSetIter(batch_size=8, dataset=testset, sampler=None)
    for batch_x, batch_y in test_batch:
        _move_dict_value_to_device(batch_x, batch_y, device=_get_model_device(model))
        output = model.predict(batch_x["word_pieces"], batch_x["word_nums"], batch_x["seq_len"])
        metric(output, batch_y)
    eval_result = metric.get_metric()
    print(eval_result)
    # print("Precision on Minor Test Set: %.2f, Recall: %.2f" % (eval_result["pr"],eval_result["re"]))

if __name__=="__main__":
    with torch.no_grad():
        # evaluate()
        print(test())
    # train()
