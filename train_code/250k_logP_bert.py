from transformers import BertModel,BertTokenizer
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split

class asmilesDataset(Dataset):
    def __init__(self,text,label,tokenizer,max_len=96):
        self.all_text = text
        self.all_label = label
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        
        result_string = self.all_text[index]
        batch_inputs = self.tokenizer(result_string,add_special_tokens=True,truncation=True,
                                                padding='max_length',max_length=self.max_len,return_tensors='pt')
        batch_text = batch_inputs['input_ids']
        batch_attention_mask = batch_inputs['attention_mask']
        batch_attention_mask = batch_attention_mask.squeeze(0)
        batch_text = batch_text.squeeze(0)

        batch_label = self.all_label[index]
        return batch_text,torch.tensor(batch_label),batch_attention_mask
    
    def __len__(self):
        return len(self.all_text)

class asmilesModel(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.bert = model
        self.classifier = nn.Linear(768,10)
        self.loss_fun = nn.L1Loss()
        self.linear = nn.Linear(10,1)
        self.lstm = nn.LSTM(768,10,batch_first=True,bidirectional=True)
        self.linear1 = nn.Linear(2,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,batch_text,batch_label,batch_attention_mask):
        bert_out0, bert_out1 = self.bert.forward(batch_text,attention_mask=batch_attention_mask,return_dict=False)
        # bertout0: batch * seq * 768
        bert_out0 = self.sigmoid(bert_out0)
        x1,x = self.lstm(bert_out0)
        # x1 : batch * seq * (2*10)
        # x[0] : 2 * batch * 10
        # x[1] : 2 * batch * 10
        x = self.linear(x[0])
        
        x = x.squeeze(2)
        x = self.linear1(x.T) 
        
        # x = self.classifier(bert_out1)
        # x = self.linear(x)
        loss = self.loss_fun(x.squeeze(1),batch_label)

        return loss,x.squeeze(1)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('/root/bert',use_fast=True)
    bertModel = AutoModel.from_pretrained('/root/bert')
    setup_seed(42)
    
    train_text = pd.read_csv('/root/250k_logP/N_250k_logP_train.csv')['smiles'].values.tolist()
    train_label = pd.read_csv('/root/250k_logP/N_250k_logP_train.csv')['logP_N'].values.tolist()
    dev_text = pd.read_csv('/root/250k_logP/N_250k_logP_dev.csv')['smiles'].values.tolist()
    dev_label = pd.read_csv('/root/250k_logP/N_250k_logP_dev.csv')['logP_N'].values.tolist()
    test_text = pd.read_csv('/root/250k_logP/N_250k_logP_test.csv')['smiles'].values.tolist()
    test_label = pd.read_csv('/root/250k_logP/N_250k_logP_test.csv')['logP_N'].values.tolist()
    
    train_batch_size = 16
    mse_loss = nn.MSELoss()
    trian_mae_list = []
    train_rmse_list = []
    train_r2_list = []
    dev_mae_list = []
    dev_rmse_list = []
    dev_r2_list = []

    epoch = 100
    lr = 0.00001
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    max_min = 15.1283
    train_dataset = asmilesDataset(train_text, train_label,tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False)

    dev_dataset = asmilesDataset(dev_text, dev_label,tokenizer)
    dev_dataloader = DataLoader(dev_dataset, batch_size=16, shuffle=False)

    test_dataset = asmilesDataset(test_text, test_label, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = asmilesModel(bertModel).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr)
    best_loss = 0.5
    # scheduler = torch.optim.lr_scheduler.StepLR(opt,step_size=3,gamma=0.1)
    for e in range(epoch):
        print(e,"*" * 100)
        
        model.train()
        train_mae = 0
        train_rmse = 0
        train_r2 = 0
        for bi, (batch_text, batch_label,batch_attention_mask) in tqdm(enumerate(train_dataloader, start=1)):
            batch_text = batch_text.to(device)
            batch_label = batch_label.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            loss, pre = model.forward(batch_text, batch_label,batch_attention_mask)
            loss.backward()
            train_mae += loss.detach().cpu().numpy() * len(batch_text) * max_min
            train_rmse += mse_loss(pre * max_min,batch_label * max_min).detach().cpu().numpy() * len(batch_text)
            train_r2 += r2_score(batch_label.detach().cpu().numpy(),pre.detach().cpu().numpy()) * len(batch_text)
            opt.step()
            opt.zero_grad()
        print(f'train_MAE:{train_mae/len(train_text)}')
        print(f'train_RMSE:{np.sqrt(train_rmse/len(train_text))}')
        print(f'train_r2:{train_r2/len(train_text)}')
        trian_mae_list.append(train_mae/len(train_text))
        train_rmse_list.append(np.sqrt(train_rmse/len(train_text)))
        train_r2_list.append(train_r2/len(train_text))
        # scheduler.step()

        model.eval()
        with torch.no_grad():
            dev_loss = 0
            dev_rmse = 0
            dev_r2 = 0
            for bi, (batch_text, batch_label,batch_attention_mask) in tqdm(enumerate(dev_dataloader, start=1)):
                batch_text = batch_text.to(device)
                batch_label = batch_label.to(device)
                batch_attention_mask = batch_attention_mask.to(device)
                loss, pre = model.forward(batch_text, batch_label,batch_attention_mask)
                dev_loss += loss.detach().cpu().numpy() * len(batch_text)  * max_min
                dev_rmse += mse_loss(pre * max_min, batch_label * max_min).detach().cpu().numpy() * len(batch_text)
                if len(batch_text) != 1:
                    dev_r2 += r2_score(batch_label.detach().cpu().numpy(),pre.detach().cpu().numpy()) * len(batch_text)
                # if bi == 2495:
                #     print(f'bi:{bi},\npred:{pre},\nlabel:{batch_label}')
            avg_loss = dev_loss / len(dev_text)
            print(f"eval_MAE:{avg_loss:.10f}")
            print(f'eval_RMSE:{np.sqrt(dev_rmse / len(dev_text))}')
            print(f'eval_r2:{dev_r2 / len(dev_text)}')
            dev_mae_list.append(avg_loss)
            dev_rmse_list.append(dev_rmse / len(dev_text))
            dev_r2_list.append(dev_r2 / len(dev_text))

        loss_ = avg_loss
        if e > 60 and loss_ < best_loss:
                best_loss = loss_
                torch.save(model, "bert_250k_logP_best_model.pt")

    best_model = torch.load('bert_250k_logP_best_model.pt')
    best_model.to(device)
    best_model.eval()
    print('TESTING')
    with torch.no_grad():
        test_mae = 0
        test_rmse = 0
        test_r2 = 0
        for bi, (batch_text, batch_label,batch_attention_mask) in tqdm(enumerate(test_dataloader, start=1)):
            batch_text = batch_text.to(device)
            batch_label = batch_label.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            loss, pre = best_model.forward(batch_text, batch_label,batch_attention_mask)
            test_rmse += mse_loss(pre * max_min, batch_label * max_min).detach().cpu().numpy()  * len(batch_text)
            test_r2 += r2_score(batch_label.detach().cpu().numpy(),pre.detach().cpu().numpy()) * len(batch_text)
            test_mae += loss.detach().cpu().numpy() * len(batch_text) * max_min
            # if bi == 2495:
            #     print(f'bi:{bi},\npred:{pre},\nlabel:{batch_label}')
        avg_test_loss = test_mae / len(test_text)
        print(f"test_MAE:{avg_test_loss:.10f}")
        print(f'test_RMAE:{np.sqrt(test_rmse / len(test_text))}')
        print(f'test_r2:{test_r2/ len(test_text)}')
    result_data = {'train_mae':trian_mae_list,'dev_mae':dev_mae_list,'train_rmse':train_rmse_list,
                   'dev_rmse':dev_rmse_list,'train_r2':train_r2_list,'dev_r2':dev_r2_list}
    result_data = pd.DataFrame(result_data)
    result_data.to_csv('bert_250k_logP_result_data.csv',index=False)