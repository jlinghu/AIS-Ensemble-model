from transformers import BertModel,BertTokenizer,XLNetModel
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import r2_score
import numpy as np
class asmilesDataset(Dataset):
    def __init__(self,text,label,tokenizer,max_len=96):
        self.all_text = text
        self.all_label = label
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        
        result_string = self.all_text[index]

        batch_inputs = self.tokenizer(result_string,add_special_tokens=True,
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
        self.tanh = nn.Sigmoid()
    
    def forward(self,batch_text,batch_label,batch_attention_mask):
        bert_out0 = self.bert.forward(batch_text,attention_mask=batch_attention_mask,return_dict=False)
        bert_out0 = self.tanh(bert_out0[0])
        x1,x = self.lstm(bert_out0)
        x = self.linear(x[0])
        x = x.squeeze(2)
        x = self.linear1(x.T) 
        loss = self.loss_fun(x.squeeze(1),batch_label/500)

        return loss * 500,x.squeeze(1) *500

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    model_file = '/home/ubuntu/bert'
    tokenizer = BertTokenizer.from_pretrained(model_file, use_fast=True)
    bertModel = XLNetModel.from_pretrained('/home/ubuntu/xlnet')
    
    setup_seed(42)
    dev_text = pd.read_csv('/home/ubuntu/UN_data/310k_MolWt/310k_MolWt_dev.csv')['smiles'].values.tolist()
    dev_label = pd.read_csv('/home/ubuntu/UN_data/310k_MolWt/310k_MolWt_dev.csv')['MolWt'].values.tolist()
    test_text = pd.read_csv('/home/ubuntu/UN_data/310k_MolWt/310k_MolWt_test.csv')['smiles'].values.tolist()
    test_label = pd.read_csv('/home/ubuntu/UN_data/310k_MolWt/310k_MolWt_test.csv')['MolWt'].values.tolist()
    train_batch_size = 16
    mse_loss = nn.MSELoss()
    lr = 0.00001
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    dev_dataset = asmilesDataset(dev_text, dev_label,tokenizer)
    dev_dataloader = DataLoader(dev_dataset, batch_size=16, shuffle=False)

    test_dataset = asmilesDataset(test_text, test_label, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = asmilesModel(bertModel).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr)
    best_model = torch.load('/home/ubuntu/ais_xlnet/xlnet_310k_MolWt_best_model.pt')
    best_model.to(device)
    pre_list = []
    pre_label_list = []
    test_list = []
    test_label_list = []
    best_model.eval()
    with torch.no_grad():
        dev_loss = 0
        dev_rmse = 0
        dev_r2 = 0
        for bi, (batch_text, batch_label,batch_attention_mask) in tqdm(enumerate(dev_dataloader, start=1)):
            batch_text = batch_text.to(device)
            batch_label = batch_label.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            loss, pre = best_model.forward(batch_text, batch_label,batch_attention_mask)
            dev_loss += loss.detach().cpu().numpy() * len(batch_text)
            dev_rmse += mse_loss(pre,batch_label).detach().cpu().numpy() * len(batch_text)
            pre_list.append(pre.detach().cpu().numpy())
            pre_label_list.append(batch_label.detach().cpu().numpy()) 
            if len(batch_text) != 1:
                dev_r2 += r2_score(batch_label.detach().cpu().numpy(),pre.detach().cpu().numpy()) * len(batch_text)
        avg_loss = dev_loss / len(dev_text)
        print(f"eval_MAE:{avg_loss:.10f}")
        print(f'eval_RMSE:{np.sqrt(dev_rmse / len(dev_text))}')
        print(f'eval_r2:{dev_r2 / len(dev_text)}')
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
            test_rmse += mse_loss(pre, batch_label).detach().cpu().numpy() * len(batch_text)
            if len(batch_label) != 1:
                test_r2 += r2_score(batch_label.detach().cpu().numpy(),pre.detach().cpu().numpy()) * len(batch_text)
            test_mae += loss.detach().cpu().numpy() * len(batch_text)
            test_list.append(pre.detach().cpu().numpy())
            test_label_list.append((batch_label).detach().cpu().numpy())
        avg_test_loss = test_mae / len(test_text)
        print(f"test_MAE:{avg_test_loss:.10f}")
        print(f'test_RMSE:{np.sqrt(test_rmse / len(test_text))}')
        print(f'test_r2:{test_r2/ len(test_text)}')
    dev_pre, dev_label, test_pre, test_label = [], [], [], []
    for e_pre,e_label,t_pre,t_label in zip(pre_list,pre_label_list,test_list,test_label_list):
        for i,j,k,q in zip(e_pre,e_label,t_pre,t_label):
            dev_pre.append(i)
            dev_label.append(j)
            test_pre.append(k)
            test_label.append(q)
    pre_list_pd = pd.DataFrame({'dev_pre':dev_pre,'dev_label':dev_label,'test_pre':test_pre,'test_label':test_label})
    pre_list_pd.to_csv('XLNet_310k_logP_pre.csv',index=False) 