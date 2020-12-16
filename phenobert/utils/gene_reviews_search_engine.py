from api import get_most_related_HPO_term
import os
import re
import pickle
index_file_path = "../data/gene_reviews.idx"
gr_ann_dir_path = "../data/gene_reviews/predict"
if not os.path.exists(index_file_path):
    print("Creating index...")
    with open(index_file_path, "wb") as index_file:
        index_dict = {}
        file_list = os.listdir(gr_ann_dir_path)
        for file_name in file_list:
            with open(os.path.join(gr_ann_dir_path, file_name), encoding="utf-8") as predict_file:
                for line in predict_file:
                    line = line.strip()
                    hpo_num = line.split("\t")[3]
                    index_dict.setdefault(hpo_num, set())
                    index_dict[hpo_num].add(file_name)
        pickle.dump(index_dict, index_file)

print("Welcome!")
print("You can enter single or multiple phrases (use & to intersect, use | to union) to search related Gene Reviews NBK id that contains these phrases.")
print("Enter 'exit' to exit this program")
with open(index_file_path, "rb") as index_file:
    index_dict = pickle.load(index_file)
    while True:
        phrase = input("Please enter your query phrase: ")
        if phrase == "exit":
            break
        phrase_list = [i.strip() for i in re.split("[&|\|]", phrase)]
        hpo_nums = [i[1] for i in get_most_related_HPO_term(phrase_list) if i != "None"]
        if hpo_nums:
            result_set = index_dict[hpo_nums[0]]
            if "|" in phrase:
                for idx in range(1, len(hpo_nums)):
                    result_set |= index_dict[hpo_nums[idx]]
            elif "&" in phrase:
                for idx in range(1, len(hpo_nums)):
                    result_set &= index_dict[hpo_nums[idx]]
            if result_set:
                print("Found following article:")
                print(sorted(list(result_set)))
            else:
                print("Result not found")
        else:
            print("Result not found")
    print("bye bye")