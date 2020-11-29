from model import HPO_model_Layer1
import torch
import torch.nn as nn
import fasttext
from util import PhraseDataSet4trainCNN, ModelSaver, ModelLoader, HPOTree, fasttext_model_path, EarlyStopping
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

train_file_path="../models/train.txt"
val_file_path="../models/val.txt"
test_file_path="../models/test.txt"
model_save_path="../models/HPOModel_H/model_layer1.pkl"
device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")




hpo_tree=HPOTree()
batch_size=256
fasttext_model = fasttext.load_model(fasttext_model_path)
l1_num = hpo_tree.n_concept_l1

# 图卷积中L矩阵

trainset=PhraseDataSet4trainCNN(train_file_path, fasttext_model, hpo_tree, l1_num)
trainloader=DataLoader(trainset, batch_size=batch_size, shuffle=True)
valset=PhraseDataSet4trainCNN(val_file_path, fasttext_model, hpo_tree, l1_num)
valloader=DataLoader(valset, batch_size=batch_size)
testset=PhraseDataSet4trainCNN(test_file_path, fasttext_model, hpo_tree, l1_num)
testloader=DataLoader(testset, batch_size=batch_size)






embedding_dim=fasttext_model.get_dimension()

# LSTMEncoder
# model=HPOModel(num_layer=2, hidden_size=512, embedding_dim=embedding_dim, output_dim1=1024, input_dim=embedding_dim, hidden_dim=512, output_dim2=1024, H0=H0).to(device)
# CNNEncoder embedding_dim=100
# model=HPOModel(in_channels=100, out_channels=1024, output_dim1=1024, input_dim=100, hidden_dim=1024, output_dim2=1024, H0=H0).to(device)
# BERTEncoder
# model=HPOModel(in_channels=embedding_dim, out_channels=1024, output_dim1=1024, input_dim=1024, hidden_dim1=1024, output_dim2=1024, n_concept=n_concept, indices=indices, values=values).to(device)
model=HPO_model_Layer1(in_channels=embedding_dim, out_channels=1024, output_dim1=1024, output_dim2=1024, n_class=l1_num).to(device)
# 查看可训练参数
print("Trainable Parameter lists:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)


loss_func=nn.BCELoss()
learning_rate = 2e-3
# learning_rate = 1e-5

optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)

# scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50], gamma=0.1)
early_stopping = EarlyStopping(patience=10, verbose=False, path=model_save_path)

num_epoches = 100
print_every=40
k_value=5


print("Starting Training...")
for epoch in range(num_epoches):
    # scheduler.step()
    total = 0
    count = 0
    avg_loss = 0
    train_count = 0
    train_total = 0
    step = 0
    print("[epoch: {:>3} step: {:>4}] lr rate: {:>4.6}".format(
        epoch, step, optimizer.state_dict()['param_groups'][0]['lr']))
    model.train()
    for data in trainloader:
        step+=1
        # CNN input
        input_data = [data["data"].float().to(device), data["seq_len"].int().to(device)]
        target=data["target"].to(device)
        hpo_num_idx=[set([hpo_tree.getHPO2idx_l1(j) for j in hpo_tree.getLayer1HPOByHPO(i)]) for i in data["hpo_num"]]
        y = model(input_data)
        train_loss = loss_func(y, target)
        prediction = y.topk(k_value)[1].tolist()
        scores = y.topk(k_value)[0].tolist()
        for item1, item2, score in zip(prediction, hpo_num_idx, scores):
            tmp = set()
            for i, sub_score in zip(item1, score):
                if sub_score > 0.9:
                    tmp.add(i)
            if len(tmp & item2) >0:
                count+=1
                train_count+=1
        total += target.size(0)
        train_total += target.size(0)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        avg_loss += train_loss.item()
        if step % print_every==0:
            avg_loss=float(avg_loss)/print_every
            acc = count/total
            print( "[epoch: {:>3} step: {:>4}] train loss: {:>4.6} ACC: {:>4.6}".format(
                epoch, step, avg_loss, acc))
            avg_loss = 0
            count=0
            total=0
    train_acc=train_count/train_total
    # evaluate on val set
    print("Evaluating on Validate Set...")
    model.eval()
    step = 0
    count = 0
    total = 0
    avg_loss_on_val = 0
    for data in valloader:
        step += 1
        # CNN input
        input_data = [data["data"].float().to(device), data["seq_len"].int().to(device)]
        phrases = data["phrase"]
        target = data["target"].to(device)
        hpo_num_idx=[set([hpo_tree.getHPO2idx_l1(j) for j in hpo_tree.getLayer1HPOByHPO(i)]) for i in data["hpo_num"]]
        y = model(input_data)
        val_loss = loss_func(y, target)
        prediction = y.topk(k_value)[1].tolist()
        scores = y.topk(k_value)[0].tolist()
        # 查看哪些没预测出来
        prediction_str = ""
        target_str = ""
        print("phrase\tpredict\ttarget")
        for phrase, item1, item2, score in zip(phrases, prediction, hpo_num_idx, scores):
            tmp = set()
            for i, sub_score in zip(item1, score):
                if sub_score > 0.9:
                    tmp.add(i)
            if len(tmp & item2) >0:
                count+=1
            # else:
                # print(
                #     f"{phrase}\t{[hpo_tree.getIdx2HPO_l1(i) for i in item1]}\t{[hpo_tree.getIdx2HPO_l1(i) for i in item2]}\t{score}")
        total += target.size(0)
        avg_loss_on_val += val_loss.item()
    acc = count/total
    avg_loss_on_val = avg_loss_on_val/step
    print("[epoch: {:>3}] Loss: {:>4.6} ACC: {:>4.6}".format(
        epoch, avg_loss_on_val, acc))

    # Early Stop
    # if early_stopping.flag or train_acc>0.99:
    #     early_stopping.flag=True
    #     early_stopping(acc, model)
    #
    # if early_stopping.early_stop:
    #     print("Early Stopping...")
    #     break

saver=ModelSaver(model_save_path)
saver.save(model)
print(f"Training Completed. Model saved to {model_save_path}.")
# print(f"Best ACC on Validate data: {early_stopping.best_score}")
# evaluate on test set
print("Evaluating on Test Set...")
# Load model
loader=ModelLoader()
model=loader.load_all(model_save_path)
model.eval()
count =0
total = 0
for data in testloader:
    # CNN input
    input_data = [data["data"].float().to(device), data["seq_len"].int().to(device)]
    target = data["target"].to(device)
    hpo_num_idx=[set([hpo_tree.getHPO2idx_l1(j) for j in hpo_tree.getLayer1HPOByHPO(i)]) for i in data["hpo_num"]]
    y = model(input_data)
    prediction = y.topk(k_value)[1].tolist()
    scores = y.topk(k_value)[0].tolist()
    for item1, item2, score in zip(prediction, hpo_num_idx, scores):
        tmp = set()
        for i, sub_score in zip(item1, score):
            if sub_score > 0.9:
                tmp.add(i)
        if len(tmp & item2) > 0:
            count += 1
    total += target.size(0)
acc = count / total
print("Test ACC: {:>4.6}".format(acc))