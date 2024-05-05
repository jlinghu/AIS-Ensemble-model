from rdkit.Chem import MolFromSmiles, MolToSmiles, CanonSmiles
import torch
import re
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
NONE_PHYSICAL_CHARACTERS = (
    '.', ':', '-', '=', '#', '(', ')', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '\\', '/', '%')

def encode(smiles, with_atomMap=False):
    """ Transforms given SMILES into Atom-in-SMILES (AiS) tokens. By default, it first canonicalizes the input SMILES.
    In order to get AiS tokens with the same order of SMILES, the input SMILES should be provided with atom map number.

    parameters:
        smiles: str, SMILES
        with_atomMap: if true, it returns AiS with the same order of SMILES.
                      Useful for randomized SMILES, or SMILES augmentation.

    return:
        str, AiS tokens with white space separated.
    """
    smiles_list = smiles.split('.')
    atomInSmiles = []

    for smiles in smiles_list:
        if with_atomMap:
            mol = MolFromSmiles(smiles)
            if mol is None: return
        else:
            tmp = MolFromSmiles(CanonSmiles(smiles))
            if tmp is None: return
            for atom in tmp.GetAtoms():
                atom.SetAtomMapNum(atom.GetIdx())
            smiles = MolToSmiles(tmp)
            mol = MolFromSmiles(smiles)
        atomID_sma = {};
        atoms = set()
        for atom in mol.GetAtoms():
            atoms.add(atom.GetSmarts())
            try:
                atomId = atom.GetPropsAsDict()['molAtomMapNumber']
            except:
                atomId = 0
            atom_symbol = atom.GetSymbol()
            chiral_tag = atom.GetChiralTag().name
            charge = atom.GetFormalCharge()
            nHs = atom.GetTotalNumHs()

            if atom.GetIsAromatic():
                atom_symbol = atom_symbol.lower()
            if chiral_tag == 'CHI_TETRAHEDRAL_CCW':
                if nHs:
                    symbol = f"[{atom_symbol}@H]"
                else:
                    symbol = f"[{atom_symbol}@]"
            elif chiral_tag == "CHI_TETRAHEDRAL_CW":
                if nHs:
                    symbol = f"[{atom_symbol}@@H]"
                else:
                    symbol = f"[{atom_symbol}@@]"
            else:
                symbol = atom_symbol
                if nHs:
                    symbol += 'H'
                    if nHs > 1:
                        symbol += '%d' % nHs
            if charge > 0:
                symbol = f"[{symbol}+{charge}]" if charge > 1 else f"[{symbol}+]"
            elif charge < 0:
                symbol = f"[{symbol}{charge}]" if charge < -1 else f"[{symbol}-]"
            ring = 'R' if atom.IsInRing() else '!R'
            neighbs = ''.join(sorted([i.GetSymbol() for i in atom.GetNeighbors()]))
            atomID_sma[atomId] = f'[{symbol};{ring};{neighbs}]'

        ais = []
        for token in smiles_tokenizer(smiles):
            if token in NONE_PHYSICAL_CHARACTERS:
                symbol = token
            else:
                try:
                    atom, atomId = token[1:-1].split(':')
                except:
                    atom, atomId = token, 0
                symbol = atomID_sma[int(atomId)]
            ais.append(symbol)

        atomInSmiles.append(' '.join(ais))

    return ' . '.join(atomInSmiles)

def smiles_tokenizer(smi):
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    return tokens #' '.join(tokens)

def get_ais():
    df = pd.read_csv('/home/ubuntu/250k_zinc.csv')
    label = []
    smiles = []
    for row in range(len(df)):
        smiles.append(df['smiles'][row])
        label.append(df['qed'][row]) #MolWt
    all_atomInSMILES = []
    for s in smiles:
        all_atomInSMILES.append(encode(s).split( ))
    vocab = {"PAD": 0, "UNK": 1}
    for i in range(len(all_atomInSMILES)):
        for j in range(len(all_atomInSMILES[i])):
            if all_atomInSMILES[i][j] not in vocab:
                vocab[all_atomInSMILES[i][j]] = len(vocab)

    return all_atomInSMILES,label,vocab


class TextDataset(Dataset):
    def __init__(self, all_text, all_lable):
        self.all_text = all_text
        self.all_lable = all_lable

    def __getitem__(self, index):
        global word_2_index
        text = self.all_text[index]
        text_index = [word_2_index[i] for i in text]
        label = self.all_lable[index]
        text_len = len(text)
        return text_index, label, text_len

    def process_batch_batch(self, data):
        global word_2_index
        batch_text = []
        batch_label = []
        batch_len = []

        for d in data:
            batch_text.append(d[0])
            batch_label.append(d[1])
            batch_len.append(d[2])
        batch_max_len = 96

        batch_text = [i + [0] * (batch_max_len - len(i)) for i in batch_text]

        return torch.tensor(batch_text), torch.tensor(batch_label)

    def __len__(self):
        return len(self.all_text)

class Model(nn.Module):
    def __init__(self, corpus_len, embedding_num, hidden_num, class_num):
        super().__init__()
        self.embedding = nn.Embedding(corpus_len, embedding_num)
        self.rnn = nn.LSTM(embedding_num, hidden_num,batch_first=True,num_layers=1,bidirectional=True)
        self.loss_fun = nn.L1Loss()
        self.linear = nn.Linear(2,1)
        self.classifier = nn.Linear(hidden_num, class_num)
       
    def forward(self, x, label=None):  # batch * sent_len
                
        x_emb = self.embedding(x)  # x_emb : batch * sent_len * emb_num
        # x_emb = self.att(x_emb)
        
        t, o = self.rnn(x_emb)  # t : batch * 1 * hidden_num    o: batch * sent_len * hidden_num

        pre = self.classifier(o[0])

        pre = pre.squeeze(2)
        # pre = pre.squeeze(1)
        pre = self.linear(pre.T)


        if label is not None:
            label = label.unsqueeze(1)
            loss = self.loss_fun(pre, label)
            return loss,pre.squeeze(1)
        else:
            return None

def same_seeds(seed):
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    torch.backends.cudnn.benchmark = False  # GPU、网络结构固定，可设置为True
    torch.backends.cudnn.deterministic = True  # 固定网络结构



if __name__ == "__main__":
    same_seeds(42)
    text, label, word_2_index = get_ais()

    train_text, temp_text, train_label, temp_label = train_test_split(text, label, test_size=0.2, random_state=42)
    dev_text, test_text, dev_label, test_label = train_test_split(temp_text, temp_label, test_size=0.5, random_state=42)

    train_batch_size = 10
    embedding_num = 256
    hidden_num = 100
    epoch = 100
    lr = 0.0001
    class_num = 1
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    train_batch_size = 16
    mse_loss = nn.MSELoss()
    trian_mae_list = []
    train_rmse_list = []
    train_r2_list = []
    dev_mae_list = []
    dev_rmse_list = []
    dev_r2_list = []

    epoch = 100
    lr = 0.0001
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    train_dataset = TextDataset(train_text, train_label)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False,
                                  collate_fn=train_dataset.process_batch_batch)
    
    dev_dataset = TextDataset(dev_text, dev_label)
    dev_dataloader = DataLoader(dev_dataset, batch_size=16, shuffle=False, collate_fn=dev_dataset.process_batch_batch)


    test_dataset = TextDataset(test_text, test_label)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=test_dataset.process_batch_batch)


    model = Model(len(word_2_index), embedding_num, hidden_num, class_num).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr)
    best_loss = 0.5
    # scheduler = torch.optim.lr_scheduler.StepLR(opt,step_size=3,gamma=0.1)
    for e in range(epoch):
        print(e,"*" * 100)
        
        model.train()
        train_mae = 0
        train_rmse = 0
        train_r2 = 0
        for bi, (batch_text, batch_label) in tqdm(enumerate(train_dataloader, start=1)):
            batch_text = batch_text.to(device)
            batch_label = batch_label.to(device)
            loss, pre = model.forward(batch_text, batch_label)
            loss.backward()
            train_mae += loss.detach().cpu().numpy() * len(batch_text)
            train_rmse += mse_loss(pre,batch_label).detach().cpu().numpy() * len(batch_text)
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
            for bi, (batch_text, batch_label) in tqdm(enumerate(dev_dataloader, start=1)):
                batch_text = batch_text.to(device)
                batch_label = batch_label.to(device)
                loss, pre = model.forward(batch_text, batch_label)
                dev_loss += loss.detach().cpu().numpy() * len(batch_text)
                dev_rmse += mse_loss(pre, batch_label).detach().cpu().numpy() * len(batch_text)
                if len(batch_text) != 1:
                    dev_r2 += r2_score(batch_label.detach().cpu().numpy(),pre.detach().cpu().numpy()) * len(batch_text)
                # if bi == 1937:
                #     print(f'bi:{bi},\npred:{pre},\nlabel:{batch_label}')
            avg_loss = dev_loss / len(dev_text)
            print(f"eval_MAE:{avg_loss:.10f}")
            print(f'eval_RMSE:{np.sqrt(dev_rmse / len(dev_text))}')
            print(f'eval_r2:{dev_r2 / len(dev_text)}')
            dev_mae_list.append(avg_loss)
            dev_rmse_list.append(np.sqrt(dev_rmse / len(dev_text)))
            dev_r2_list.append(dev_r2 / len(dev_text))

        loss_ = avg_loss
        if e > 90 and loss_ < best_loss:
                best_loss = loss_
                torch.save(model, "LSTM_250k_qed_best_model.pt")
          

    best_model = torch.load('LSTM_250k_qed_best_model.pt')
    best_model.to(device)
    best_model.eval()
    print('TESTING')
    with torch.no_grad():
        test_mae = 0
        test_rmse = 0
        test_r2 = 0
        for bi, (batch_text, batch_label) in tqdm(enumerate(test_dataloader, start=1)):
            batch_text = batch_text.to(device)
            batch_label = batch_label.to(device)
            loss, pre = best_model.forward(batch_text, batch_label)
            test_rmse += mse_loss(pre, batch_label).detach().cpu().numpy() * len(batch_text)
            test_r2 += r2_score(batch_label.detach().cpu().numpy(),pre.detach().cpu().numpy()) * len(batch_text)
            test_mae += loss.detach().cpu().numpy() * len(batch_text)
            # if bi == 2495:
            #     print(f'bi:{bi},\npred:{pre},\nlabel:{batch_label}')
        avg_test_loss = test_mae / len(test_text)
        print(f"test_MAE:{avg_test_loss:.10f}")
        print(f'test_RMAE:{np.sqrt(test_rmse / len(test_text))}')
        print(f'test_r2:{test_r2/ len(test_text)}')
    result_data = {'train_mae':trian_mae_list,'dev_mae':dev_mae_list,'train_rmse':train_rmse_list,
                   'dev_rmse':dev_rmse_list,'train_r2':train_r2_list,'dev_r2':dev_r2_list}
    result_data = pd.DataFrame(result_data)
    result_data.to_csv('LSTM_250k_qed_result_data.csv',index=False)

