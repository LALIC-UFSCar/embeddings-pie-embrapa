import os
import sys
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import RobertaModel, RobertaTokenizer

class Settings:
    batch_size=16
    max_len=350
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 318

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True

set_seed(Settings.seed)

class TrainValidDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.text = df["textos"].values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        texts = self.text[idx]
        tokenized = self.tokenizer.encode_plus(texts, truncation=True, add_special_tokens=True, max_length=self.max_len, padding="max_length")
        ids = tokenized["input_ids"]
        mask = tokenized["attention_mask"]
        return {
            "ids": torch.LongTensor(ids),
            "mask": torch.LongTensor(mask),
        }

class CommonLitRoBERTa(nn.Module):
    def __init__(self, pretrained_path):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(pretrained_path)

    def forward(self, ids, mask):
        output = self.roberta(ids, attention_mask=mask)
        return output

model = CommonLitRoBERTa("roberta-base")
model.to(Settings.device)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# GET PARAMETERS ******************************************************************************
qtde_parametros = len(sys.argv)

if (qtde_parametros >= 2):
  # read folder name in which the texts are in
  pasta = sys.argv[1]

else:
  print("Erro de sintaxe!\n")
  print("Comande: python3 embeddings_dinamicas.py <pasta-de-arquivos-txt-corpus>/")
  print("\tExemplo: python3 embeddings_dinamicas.py /home/corpus/")
  sys.exit()
#fim if

try:
  textos = os.listdir(pasta)
except:
  print("Nome de pasta inválida")
  sys.exit()

files_dic = {"textos": textos}
df_train = pd.DataFrame(files_dic)

def train_tokenizer(df_train, tokenizer):
    # load dataset and train data
    train_dataset = TrainValidDataset(df_train, tokenizer, Settings.max_len)
    train_loader = DataLoader(train_dataset, batch_size=Settings.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # inicialize batch
    batch = next(iter(train_loader))

    # set ids and masks
    ids = batch["ids"].to(Settings.device)
    mask = batch["mask"].to(Settings.device)
    print(ids.shape)
    print(mask.shape)

    # create model with ids and masks
    output = model(ids, mask)
    '''
    print(output)

    last_hidden_state = output[0]
    print("last_hidden_state shape:", last_hidden_state.shape)


    pooler_output = output[1]
    print(pooler_output)

    if(pooler_output.hasattr(shape)):
        print("pooler_output shape:", pooler_output.shape)

    cls_embeddings = last_hidden_state[:, 0, :].detach()
    print("cls_embeddings shape:", cls_embeddings.shape)
    print(cls_embeddings)
    pd.DataFrame(cls_embeddings.numpy()).head()

    print(last_hidden_state.shape)
    pooled_embeddings = last_hidden_state.detach().mean(dim=1)
    print("shape:", pooled_embeddings.shape)
    print("")
    print(pooled_embeddings)
    pd.DataFrame(pooled_embeddings.numpy()).head()
    '''

train_tokenizer(df_train, tokenizer)

from transformers import RobertaTokenizerFast

tokenizer_fast = RobertaTokenizerFast.from_pretrained("roberta-base")
tokenizer_fast

train_tokenizer(df_train, tokenizer_fast)

from transformers import AutoTokenizer, RobertaForCausalLM, AutoConfig
import torch

tokenizer_causallm = AutoTokenizer.from_pretrained("roberta-base")
config = AutoConfig.from_pretrained("roberta-base")
config.is_decoder = True
model = RobertaForCausalLM.from_pretrained("roberta-base", config=config)
model.to(Settings.device)

train_tokenizer(df_train, tokenizer_causallm)