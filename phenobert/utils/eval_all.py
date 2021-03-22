import os
from util import HPOTree

dataset="GSC+"

test_corpus_dir_path="../data/"+dataset+"/corpus"
gold_dir_path="../data/"+dataset+"/ann"
method1_dir_path="../evaluate/NCBO/predict_"+dataset
method2_dir_path="../evaluate/NCR/predict_"+dataset
method3_dir_path="../evaluate/Clinphen/predict_"+dataset
method4_dir_path="../evaluate/MetaMapLite/predict_"+dataset
method5_dir_path="../evaluate/Doc2hpo/predict_"+dataset
method6_dir_path="../evaluate/Ours/predict_"+dataset
method7_dir_path="../evaluate/PhenoTagger/predict_"+dataset

hpo_tree = HPOTree()
hpo_tree.buildHPOTree()
filelist=os.listdir(test_corpus_dir_path)
alt_id_dict=hpo_tree.alt_id_dict

# Micro Way
norm1,denorm_pr1,denorm_re1=0,0,0
norm2,denorm_pr2,denorm_re2=0,0,0
norm3,denorm_pr3,denorm_re3=0,0,0
norm4,denorm_pr4,denorm_re4=0,0,0
norm5,denorm_pr5,denorm_re5=0,0,0
norm6,denorm_pr6,denorm_re6=0,0,0
norm7,denorm_pr7,denorm_re7=0,0,0



# Macro Way
macro_re1=0
macro_pr1=0
macro_re2=0
macro_pr2=0
macro_re3=0
macro_pr3=0
macro_re4=0
macro_pr4=0
macro_re5=0
macro_pr5=0
macro_re6=0
macro_pr6=0
macro_re7=0
macro_pr7=0



# Extended Way
extend_s1=0
extend_s2=0
extend_s3=0
extend_s4=0
extend_s5=0
extend_s6=0
extend_s7=0


def calc_metric(true_hpos, method_hpos):
    pr = 0
    re = 0
    f1 = 0
    if len(true_hpos) == 0 and len(method_hpos) == 0:
        pr = 1
        re = 1
        f1 = 1
    elif len(true_hpos) != 0 and len(method_hpos) != 0:
        pr = len(true_hpos & method_hpos) / len(method_hpos)
        re = len(true_hpos & method_hpos) / len(true_hpos)
        if pr + re > 0:
            f1 = 2 * pr * re / (pr + re)
    return pr, re, f1




for file_name in filelist:
    true_path=os.path.join(gold_dir_path,file_name)
    method1_path=os.path.join(method1_dir_path,file_name)
    method2_path=os.path.join(method2_dir_path,file_name)
    method3_path=os.path.join(method3_dir_path,file_name)
    method4_path=os.path.join(method4_dir_path,file_name)
    method5_path=os.path.join(method5_dir_path,file_name)
    method6_path=os.path.join(method6_dir_path,file_name)
    method7_path=os.path.join(method7_dir_path,file_name)


    # ann
    true_hpos = set()
    with open(true_path) as true_file:
        for line in true_file:
            if dataset.startswith("GSC+") or dataset.endswith("GSC+"):
                inter = line.strip().split("\t")[1]
                tmp = inter.split(" | ")
                r_hpo_num = tmp[0]
                hpo_num = r_hpo_num[:2] + ":" + r_hpo_num[3:]
            if dataset == "68_clinical" or dataset == "gene_reviews":
                inter = line.strip().split("\t")
                hpo_num = inter[3]
            if dataset.startswith("level"):
                hpo_num = line.strip().split(",")
                true_hpos.update(hpo_num)
                continue
            if hpo_num in alt_id_dict:
                hpo_num = alt_id_dict[hpo_num]
            if hpo_num in hpo_tree.phenotypic_abnormality:
                true_hpos.add(hpo_num)

    # NCBO
    method1_hpos = set()
    with open(method1_path) as method1_file:
        for line in method1_file:
            hpo_num = line.strip().split("\t")[2]
            if hpo_num in alt_id_dict:
                hpo_num = alt_id_dict[hpo_num]
            if hpo_num in hpo_tree.phenotypic_abnormality:
                method1_hpos.add(hpo_num)
    pr1, re1, f11 = calc_metric(true_hpos, method1_hpos)
    norm1 += len(true_hpos & method1_hpos)
    denorm_pr1 += len(method1_hpos)
    denorm_re1 += len(true_hpos)
    macro_re1 += re1
    macro_pr1 += pr1
    s1 = hpo_tree.getHPO_set_similarity_max(method1_hpos, true_hpos)
    extend_s1 += s1



    # NCR
    method2_hpos = set()
    if os.path.exists(method2_path):
        with open(method2_path) as method2_file:
            for line in method2_file:
                hpo_num = line.strip().split("\t")[2]
                if hpo_num in alt_id_dict:
                    hpo_num = alt_id_dict[hpo_num]
                if hpo_num in hpo_tree.phenotypic_abnormality:
                    method2_hpos.add(hpo_num)
    pr2, re2, f12 = calc_metric(true_hpos, method2_hpos)
    norm2 += len(true_hpos & method2_hpos)
    denorm_pr2 += len(method2_hpos)
    denorm_re2 += len(true_hpos)
    macro_re2 += re2
    macro_pr2 += pr2
    s2 = hpo_tree.getHPO_set_similarity_max(method2_hpos, true_hpos)
    extend_s2 += s2

    # Clinphen
    method3_hpos = set()
    with open(method3_path) as method3_file:
        next(method3_file)
        for line in method3_file:
            hpo_num = line.strip().split("\t")[0]
            if hpo_num in alt_id_dict:
                hpo_num = alt_id_dict[hpo_num]
            if hpo_num in hpo_tree.phenotypic_abnormality:
                method3_hpos.add(hpo_num)
    pr3, re3, f13 = calc_metric(true_hpos, method3_hpos)
    norm3 += len(true_hpos & method3_hpos)
    denorm_pr3 += len(method3_hpos)
    denorm_re3 += len(true_hpos)
    macro_re3 += re3
    macro_pr3 += pr3
    s3 = hpo_tree.getHPO_set_similarity_max(method3_hpos, true_hpos)
    extend_s3 += s3

    # MetaMapLite
    method4_hpos = set()
    with open(method4_path) as method4_file:
        for line in method4_file:
            hpo_num = line.strip().split("\t")[1]
            if hpo_num in alt_id_dict:
                hpo_num = alt_id_dict[hpo_num]
            if hpo_num in hpo_tree.phenotypic_abnormality:
                method4_hpos.add(hpo_num)
    pr4, re4, f14 = calc_metric(true_hpos, method4_hpos)
    norm4 += len(true_hpos & method4_hpos)
    denorm_pr4 += len(method4_hpos)
    denorm_re4 += len(true_hpos)
    macro_re4 += re4
    macro_pr4 += pr4
    s4 = hpo_tree.getHPO_set_similarity_max(method4_hpos, true_hpos)
    extend_s4 += s4


    # Doc2hpo
    method5_hpos = set()
    with open(method5_path) as method5_file:
        for line in method5_file:
            hpo_num = line.strip().split("\t")[1]
            if hpo_num in alt_id_dict:
                hpo_num = alt_id_dict[hpo_num]
            if hpo_num in hpo_tree.phenotypic_abnormality:
                method5_hpos.add(hpo_num)
    pr5, re5, f15 = calc_metric(true_hpos, method5_hpos)
    norm5 += len(true_hpos & method5_hpos)
    denorm_pr5 += len(method5_hpos)
    denorm_re5 += len(true_hpos)
    macro_re5 += re5
    macro_pr5 += pr5
    s5 = hpo_tree.getHPO_set_similarity_max(method5_hpos, true_hpos)
    extend_s5 += s5



    # PhenoBERT
    method6_hpos = set()
    if os.path.exists(method6_path):
        with open(method6_path) as method6_file:
            for line in method6_file:
                hpo_num = line.strip().split("\t")[3]
                if hpo_num in alt_id_dict:
                    hpo_num = alt_id_dict[hpo_num]
                if hpo_num in hpo_tree.phenotypic_abnormality:
                    method6_hpos.add(hpo_num)
    pr6, re6, f16 = calc_metric(true_hpos, method6_hpos)
    norm6 += len(true_hpos & method6_hpos)
    denorm_pr6 += len(method6_hpos)
    denorm_re6 += len(true_hpos)
    macro_re6 += re6
    macro_pr6 += pr6
    s6 = hpo_tree.getHPO_set_similarity_max(method6_hpos, true_hpos)
    extend_s6 += s6

    # PhenoTagger
    method7_hpos = set()
    if os.path.exists(method7_path):
        with open(method7_path) as method7_file:
            for line in method7_file:
                inter = line.strip().split("\t")
                if len(inter) == 7:
                    entity_type = inter[4]
                    if entity_type == "Phenotype":
                        hpo_num = inter[5]
                        if hpo_num in alt_id_dict:
                            hpo_num = alt_id_dict[hpo_num]
                        if hpo_num in hpo_tree.phenotypic_abnormality:
                            method7_hpos.add(hpo_num)
    pr7, re7, f17 = calc_metric(true_hpos, method7_hpos)
    norm7 += len(true_hpos & method7_hpos)
    denorm_pr7 += len(method7_hpos)
    # print(file_name, len(true_hpos & method7_hpos), len(method7_hpos))
    denorm_re7 += len(true_hpos)
    macro_re7 += re7
    macro_pr7 += pr7
    s7 = hpo_tree.getHPO_set_similarity_max(method7_hpos, true_hpos)
    extend_s7 += s7


    # print(file_name + "\tPrecision\tRecall\tF1 score")
    # print("NCBO\t%.4f\t%.4f\t%.4f\t%.4f" % (pr1, re1, f11, s1))
    # print("NCR\t%.4f\t%.4f\t%.4f\t%.4f" % (pr2, re2, f12, s2))
    # print("Clinphen\t%.4f\t%.4f\t%.4f\t%.4f" % (pr3, re3, f13, s3))
    # print("MetaMap\t%.4f\t%.4f\t%.4f\t%.4f" % (pr4, re4, f14, s4))
    # print("Doc2hpo\t%.4f\t%.4f\t%.4f\t%.4f" % (pr5, re5, f15, s5))
    # print("PhenoBERT\t%.4f\t%.4f\t%.4f\t%.4f" % (pr6, re6, f16, s6))


print("Evaluate in Micro Way")
pr1=norm1/denorm_pr1
re1=norm1/denorm_re1
print("NCBO Precision: %.4f\tRecal: %.4f\tF1 score: %.4f" % (pr1,re1,2*pr1*re1/(pr1+re1)))
pr2=norm2/denorm_pr2
re2=norm2/denorm_re2
print("NCR Precision: %.4f\tRecal: %.4f\tF1 score: %.4f" % (pr2,re2,2*pr2*re2/(pr2+re2)))
pr3=norm3/denorm_pr3
re3=norm3/denorm_re3
print("Clinphen Precision: %.4f\tRecal: %.4f\tF1 score: %.4f" % (pr3,re3,2*pr3*re3/(pr3+re3)))
pr4=norm4/denorm_pr4
re4=norm4/denorm_re4
print("MetaMap Precision: %.4f\tRecal: %.4f\tF1 score: %.4f" % (pr4,re4,2*pr4*re4/(pr4+re4)))
pr5=norm5/denorm_pr5
re5=norm5/denorm_re5
print("Doc2hpo Precision: %.4f\tRecal: %.4f\tF1 score: %.4f" % (pr5,re5,2*pr5*re5/(pr5+re5)))
pr6=norm6/denorm_pr6
re6=norm6/denorm_re6
print("PhenoBERT Precision: %.4f\tRecal: %.4f\tF1 score: %.4f" % (pr6,re6,2*pr6*re6/(pr6+re6)))
pr7=norm7/denorm_pr7
re7=norm7/denorm_re7
print("PhenoTagger Precision: %.4f\tRecal: %.4f\tF1 score: %.4f" % (pr7,re7,2*pr7*re7/(pr7+re7)))

print("\nEvaluate in Macro Way")
pr1=macro_pr1/len(filelist)
re1=macro_re1/len(filelist)
print("NCBO Precision: %.4f\tRecal: %.4f\tF1 score: %.4f" % (pr1,re1,2*pr1*re1/(pr1+re1)))
pr2=macro_pr2/len(filelist)
re2=macro_re2/len(filelist)
print("NCR Precision: %.4f\tRecal: %.4f\tF1 score: %.4f" % (pr2,re2,2*pr2*re2/(pr2+re2)))
pr3=macro_pr3/len(filelist)
re3=macro_re3/len(filelist)
print("Clinphen Precision: %.4f\tRecal: %.4f\tF1 score: %.4f" % (pr3,re3,2*pr3*re3/(pr3+re3)))
pr4=macro_pr4/len(filelist)
re4=macro_re4/len(filelist)
print("MetaMap Precision: %.4f\tRecal: %.4f\tF1 score: %.4f" % (pr4,re4,2*pr4*re4/(pr4+re4)))
pr5=macro_pr5/len(filelist)
re5=macro_re5/len(filelist)
print("Doc2hpo Precision: %.4f\tRecal: %.4f\tF1 score: %.4f" % (pr5,re5,2*pr5*re5/(pr5+re5)))
pr6=macro_pr6/len(filelist)
re6=macro_re6/len(filelist)
print("PhenoBERT Precision: %.4f\tRecal: %.4f\tF1 score: %.4f" % (pr6,re6,2*pr6*re6/(pr6+re6)))
pr7=macro_pr7/len(filelist)
re7=macro_re7/len(filelist)
print("PhenoTagger Precision: %.4f\tRecal: %.4f\tF1 score: %.4f" % (pr7,re7,2*pr7*re7/(pr7+re7)))


print("\nEvaluate in Node Similarity Way")
s1=extend_s1/len(filelist)
print("NCBO Similarity: %.4f" % s1)
s2=extend_s2/len(filelist)
print("NCR Similarity: %.4f" % s2)
s3=extend_s3/len(filelist)
print("Clinphen Similarity: %.4f" % s3)
s4=extend_s4/len(filelist)
print("MetaMap Similarity: %.4f" % s4)
s5=extend_s5/len(filelist)
print("Doc2hpo Similarity: %.4f" % s5)
s6=extend_s6/len(filelist)
print("PhenoBERT Similarity: %.4f" % s6)
s7=extend_s7/len(filelist)
print("PhenoTagger Similarity: %.4f" % s7)