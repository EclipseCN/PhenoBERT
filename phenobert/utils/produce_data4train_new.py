#! /home/fyh/anaconda3/envs/NERpy3/bin/python
import random
import nltk
from util import HPOTree,HPO_class,getNames, containNum
obo_file_path="../data/hpo.json"

# wiki_file_path="../models/wikipedia.txt"
# none_list=[]
# # wiki中无意义的短语
# with open(wiki_file_path, "r", encoding="utf-8") as wiki_file:
#     max_length=10000000
#     data=wiki_file.read()[:max_length]
#     tokens=processStr(data)
#     indexes=random.sample(range(len(tokens)), 10000)
#     for sub_index in indexes:
#         none_list.append(" ".join(tokens[sub_index:sub_index+random.randint(1,10)]))

hpo_tree=HPOTree()
# p_phrase 2 HPO
p_phrase2HPO=hpo_tree.p_phrase2HPO
# HPO:[name+synonym]
HPO2phrase={}
data=hpo_tree.data
for hpo_name in data:
    struct=HPO_class(data[hpo_name])
    names=getNames(struct)
    HPO2phrase[hpo_name]=names

total_neg_lst = []
part_pos_lst = []

for true_hpo_name in HPO2phrase.keys():
    struct = HPO_class(data[true_hpo_name])
    # 子代
    Candidate_hpo_set_son = list(struct.child)
    Candidate_hpo_set_son_set = set(Candidate_hpo_set_son)
    if len(Candidate_hpo_set_son) > 10:
        Candidate_hpo_set_son = random.sample(Candidate_hpo_set_son, 10)
    # 祖先代
    Candidate_hpo_set_f = list(struct.father)
    Candidate_hpo_set_f_set = set(Candidate_hpo_set_f)
    if len(Candidate_hpo_set_f) > 10:
        Candidate_hpo_set_f = random.sample(Candidate_hpo_set_f, 10)

    # 直系父代
    Candidate_hpo_set_f_d = list(struct.is_a)

    sub_neg_phrases = []
    sub_pos_phrases = []
    sub_pos_phrases_d = []

    # 祖先关系
    tmp = []
    for hpo_name in Candidate_hpo_set_f:
        tmp.extend(HPO2phrase[hpo_name])
    sub_pos_phrases.extend(tmp)

    # 直属父代
    tmp = []
    for hpo_name in Candidate_hpo_set_f_d:
        tmp.extend(HPO2phrase[hpo_name])
    sub_pos_phrases_d.extend(tmp)

    sub_pos_phrases.extend(sub_pos_phrases_d)
    sub_fa_phrases_set = set(sub_pos_phrases)

    # 加入了祖先
    part_pos_lst.extend(
        [random.choice(HPO2phrase[true_hpo_name]) + "::" + item + "\t1\n" for item in sub_fa_phrases_set if
         not containNum(item)])

    # 子代关系
    tmp = []
    for hpo_name in Candidate_hpo_set_son:
        tmp.extend(HPO2phrase[hpo_name])
    if len(tmp) > 10:
        tmp = random.sample(tmp, 10)
    sub_neg_phrases.extend(tmp)


    # 不相关的病种
    tmp = []
    Candidate_hpo_set2 = random.sample(HPO2phrase.keys(), 5)
    for item in Candidate_hpo_set2:
        if item in Candidate_hpo_set_son_set or item in Candidate_hpo_set_f_set or item == true_hpo_name:
            continue
        tmp.extend(HPO2phrase[item])
    if len(tmp) > 5:
        tmp = random.sample(tmp, 5)
    sub_neg_phrases.extend(tmp)
    total_neg_lst.extend(
        [random.choice(HPO2phrase[true_hpo_name]) + "::" + item + "\t0\n" for item in sub_neg_phrases if
         not containNum(item)])

    sub_neg_phrases = []

    # 包含词
    phrases = set(HPO2phrase[true_hpo_name])
    part_pos_set = set(sub_pos_phrases)
    word_set = set()
    for phrase in phrases:
        word_set.update(nltk.word_tokenize(phrase))
    if len(word_set) > 10:
        word_set = random.sample(word_set, 10)
    for word in word_set:
        if word not in phrases and word not in part_pos_set:
            sub_neg_phrases.append(word)

    total_neg_lst.extend(
        [item + "::" + random.choice(HPO2phrase[true_hpo_name]) + "\t0\n" for item in sub_neg_phrases if
         not containNum(item)])



# this part produce train file for bert
write_file=open("../models/all4bert_new_triple_2.txt","w")

# get part neg list
pos_lst=[]
for hpo_name in HPO2phrase:
    names=HPO2phrase[hpo_name]
    if len(names)<=1:
        continue
    tmp_lst=[]
    for index1 in range(len(names)):
        for index2 in range(len(names)):
            if index1==index2:
                continue
            # 表示语义完全相关
            if not containNum(names[index1]) and not containNum(names[index2]):
                tmp_lst.append(names[index1]+"::"+names[index2]+"\t2\n")
    pos_lst.extend(tmp_lst)

# 78968
print("we have %d positive entries." % len(pos_lst))
# 140066
print("we have %d part positive entries." % len(part_pos_lst))
# 174602
print("we have %d part/total negative entries." % len(total_neg_lst))
total_neg_lst.extend(pos_lst)
total_neg_lst.extend(part_pos_lst)
random.shuffle(total_neg_lst)
write_file.write("".join(total_neg_lst))
write_file.close()


