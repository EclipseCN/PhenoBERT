import json
import random
from util import HPO_class, hpo_json_path, getNames, HPOTree




hpo_tree=HPOTree()
with open(hpo_json_path) as json_file:
    data=json.loads(json_file.read())



total_l1_root = hpo_tree.layer1
for sub_l1_root in total_l1_root:
    # get idx2hpo hpo2idx for given HPO
    root_idx, hpo_list, n_concept, _, _ = hpo_tree.getMaterial4L1(sub_l1_root)
    hpo_list_set = set(hpo_list)
    train_file_path=f"../models/train_source/train_{root_idx}.txt"

    phrase_list = []
    count1=0
    count2=0
    for hpo_name in hpo_list:
        struct = HPO_class(data[hpo_name])
        # 训练集中同义词部分
        names = getNames(struct)
        for name in names:
            phrase_list.append(name+"\t"+hpo_name+"\n")


    print("We have ", len(phrase_list), "phrases")

    # 其他类型的一些病种
    part_phrase_list = []
    L1_hpo = hpo_tree.layer1_set
    for l1_hpo_num in L1_hpo:
        if l1_hpo_num==sub_l1_root:
            continue
        _, hpo_list_sample, _, _, _ = hpo_tree.getMaterial4L1(l1_hpo_num)
        if len(hpo_list_sample) > 200:
            hpo_list_sample = random.sample(hpo_list_sample, 200)
        hpo_list_sample = [i for i in hpo_list_sample if i not in hpo_list_set]
        for hpo_name in hpo_list_sample:
            struct = HPO_class(data[hpo_name])
            # 训练集中同义词部分
            names = getNames(struct)
            for name in names:
                part_phrase_list.append(name + "\t" + "None" + "\n")



    # wiki_file_path="../models/wikipedia.txt"
    # none_list=set()
    # # wiki中无意义的短语
    # with open(wiki_file_path, "r", encoding="utf-8") as wiki_file:
    #     max_length=10000000
    #     data=wiki_file.read()[:max_length]
    #     tokens=processStr(data)
    #     indexes=random.sample(range(len(tokens)), len(phrase_list))
    #     for sub_index in indexes:
    #         none_list.add(" ".join(tokens[sub_index:sub_index+random.randint(1,10)])+"\tNone\n")
    #     indexes = random.sample(range(len(tokens)), len(phrase_list))
    #     for sub_index in indexes:
    #         none_list.add(" ".join(tokens[sub_index:sub_index+random.randint(1,3)])+"\tNone\n")
    # none_list=list(none_list)



    # 训练集
    # phrase_list.extend(none_list)
    phrase_list.extend(part_phrase_list)
    with open(train_file_path,"w") as write_file:
        for item in phrase_list:
            write_file.write(item)



