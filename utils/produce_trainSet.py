import json
import random
from util import HPO_class, hpo_json_path, getNames, GSCp_ann_path, HPOTree, processStr

train_file_path="../models/train.txt"
validate_file_path="../models/val.txt"
test_file_path="../models/test.txt"
defi_file_path="../models/definition.txt"
wiki_file_path="../models/wikipedia.txt"

hpo_tree=HPOTree()
with open(hpo_json_path) as json_file:
    data=json.loads(json_file.read())
phenotypic_abnormality_list=hpo_tree.hpo_list

phrase_list = []
count1=0
count2=0
for hpo_name in phenotypic_abnormality_list:
    struct = HPO_class(data[hpo_name])
    # 训练集中同义词部分
    names = getNames(struct)
    for name in names:
        phrase_list.append(name+"\t"+hpo_name+"\n")



print("We have ", len(phrase_list), "phrases")


none_list=set()
# wiki中无意义的短语
with open(wiki_file_path, "r", encoding="utf-8") as wiki_file:
    max_length=10000000
    data=wiki_file.read()[:max_length]
    tokens=processStr(data)
    indexes=random.sample(range(len(tokens)), 10000)
    for sub_index in indexes:
        none_list.add(" ".join(tokens[sub_index:sub_index+random.randint(1,10)])+"\tNone\n")
    indexes = random.sample(range(len(tokens)), 10000)
    for sub_index in indexes:
        none_list.add(" ".join(tokens[sub_index:sub_index+random.randint(1,3)])+"\tNone\n")
none_list=list(none_list)



# 训练集
phrase_list.extend(none_list)
with open(train_file_path,"w") as write_file:
    for item in phrase_list:
        write_file.write(item)

# val_list=[]
# file_list=os.listdir(GSCp_ann_path)
# random.shuffle(file_list)
# val_file_list=file_list[:45]
# test_file_list=file_list[45:]
# phrase_set=set()
# for file_name in val_file_list:
#     with open(os.path.join(GSCp_ann_path, file_name),"r") as gsc_file:
#         for line in gsc_file:
#             inter=line.strip().split("\t")[1].split(" | ")
#             phrase=inter[1]
#             hpo = "HP" + ":" + inter[0][3:]
#             if phrase not in phrase_set and hpo in hpo_tree.phenotypic_abnormality:
#                 phrase_set.add(phrase)
#                 if hpo in hpo_tree.alt_id_dict:
#                     hpo=hpo_tree.alt_id_dict[hpo]
#                 val_list.append(f"{phrase}\t{hpo}\n")
#
# test_list=[]
# phrase_set=set()
# for file_name in test_file_list:
#     with open(os.path.join(GSCp_ann_path, file_name),"r") as gsc_file:
#         for line in gsc_file:
#             inter=line.strip().split("\t")[1].split(" | ")
#             phrase=inter[1]
#             hpo = "HP" + ":" + inter[0][3:]
#             if phrase not in phrase_set and hpo in hpo_tree.phenotypic_abnormality:
#                 phrase_set.add(phrase)
#                 if hpo in hpo_tree.alt_id_dict:
#                     hpo = hpo_tree.alt_id_dict[hpo]
#                 test_list.append(f"{phrase}\t{hpo}\n")

# # 验证集
# with open(validate_file_path,"w") as write_file:
#     for item in val_list:
#         write_file.write(item)
#
# # 测试集
# with open(test_file_path,"w") as write_file:
#     for item in test_list:
#         write_file.write(item)