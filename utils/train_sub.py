from model import HPOModel
import numpy as np
import torch
import torch.nn as nn
import fasttext
from util import PhraseDataSet4trainCNN_sub, ModelSaver, HPOTree, fasttext_model_path, EarlyStopping
from torch.utils.data import DataLoader



hpo_tree=HPOTree()
batch_size=256
fasttext_model = fasttext.load_model(fasttext_model_path)
total_l1_root = hpo_tree.layer1
for sub_l1_root in total_l1_root:
    # get idx2hpo hpo2idx for given HPO
    root_idx, hpo_list, n_concept, hpo2idx, idx2hpo = hpo_tree.getMaterial4L1(sub_l1_root)
    train_file_path=f"../models/train_source/train_{root_idx}.txt"
    model_save_path=f"../models/HPOModel_H/model_l1_{root_idx}.pkl"
    device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")



    # 图卷积中L矩阵
    L=hpo_tree.getAdjacentMatrixAncestors(sub_l1_root, n_concept)
    indices = torch.from_numpy(np.asarray([L.row, L.col]).astype('int64')).long()
    values = torch.from_numpy(L.data.astype(np.float32))


    trainset=PhraseDataSet4trainCNN_sub(train_file_path, fasttext_model, hpo2idx)
    trainloader=DataLoader(trainset, batch_size=batch_size, shuffle=True)

    embedding_dim=fasttext_model.get_dimension()

    model=HPOModel(in_channels=embedding_dim, out_channels=1024, output_dim1=1024, input_dim=1024, hidden_dim1=1024, output_dim2=1024, n_concept=n_concept, indices=indices, values=values).to(device)
    # 查看可训练参数
    print("Trainable Parameter lists:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)


    loss_func=nn.CrossEntropyLoss()
    learning_rate = 2e-3

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50], gamma=0.1)
    early_stopping = EarlyStopping(patience=10, verbose=False, path=model_save_path)

    num_epoches = 150
    print_every=20


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
            target=data["target"].squeeze().to(device)
            y = model(input_data)
            train_loss = loss_func(y, target)
            prediction = torch.argmax(y, 1)
            res = prediction == target
            count += torch.sum(res).item()
            train_count+= torch.sum(res).item()
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

    saver=ModelSaver(model_save_path)
    saver.save(model)
    print(f"Training Completed. Model saved to {model_save_path}.")
