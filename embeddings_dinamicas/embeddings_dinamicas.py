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
import tqdm

# GET PARAMETERS ******************************************************************************
opcoes_modelos = ['roberta-base', 'roberta-base-tokenizer-fast', 'roberta-base-causal-ml', 'bertimbau-base', 'bertimbau-large']

qtde_parametros = len(sys.argv)
config_modelo_escolhido = "roberta-base"
if (qtde_parametros >= 2):
    # read folder name in which the texts are in
    pasta = sys.argv[1]

    if qtde_parametros >= 3:
        if sys.argv[2] in opcoes_modelos:
            config_modelo_escolhido = sys.argv[2]
        else:
            print("Erro no nome de configuração do modelo!\n")
            print("As opções de configuração de modelo são:")
            print("\troberta-base | roberta-base-tokenizer-fast | roberta-base-causal-ml | bertimbau-base | bertimbau-large")
            print("Comande: python3 embeddings_dinamicas.py <pasta-de-arquivos-txt-corpus>/ | <opcao-config-modelo>")
            print("\tExemplo: python3 embeddings_dinamicas.py /home/corpus/ roberta-base")
            sys.exit()
else:
    print("Erro de sintaxe!\n")
    print("Comande: python3 embeddings_dinamicas.py <pasta-de-arquivos-txt-corpus>/ | <opcao-config-modelo>")
    print("\tExemplo: python3 embeddings_dinamicas.py /home/corpus/")
    print("\tExemplo: python3 embeddings_dinamicas.py /home/corpus/ roberta-base")
    print("\tExemplo: python3 embeddings_dinamicas.py /home/corpus/ roberta-base-tokenizer-fast")
    print("\tExemplo: python3 embeddings_dinamicas.py /home/corpus/ roberta-base-causal-ml")
    print("\tExemplo: python3 embeddings_dinamicas.py /home/corpus/ bertimbau-base")
    print("\tExemplo: python3 embeddings_dinamicas.py /home/corpus/ bertimbau-large")
    sys.exit()

try:
    textos = os.listdir(pasta)
except:
    print("Nome de pasta inválida")
    sys.exit()

textos_com_caminhos = []
for t in textos:
    textos_com_caminhos.append(pasta+t)
textos = textos_com_caminhos

# CONFIGURATIONS ******************************************************************************
class Settings:
    batch_size= 8
    max_len= 512
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
    def __init__(self, textos, tokenizer, max_len):
        self.texts = textos
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        print(idx)
        text = self.texts[idx]
        batch_tokenized = self.tokenizer.encode_plus(text, truncation=True, add_special_tokens=True, max_length=self.max_len, padding="max_length")

        labels = torch.LongTensor(batch_tokenized["input_ids"])
        mask = torch.LongTensor(batch_tokenized["attention_mask"])

        # make copy of labels tensor, this will be input_ids
        input_ids = labels.detach().clone()
        # create random array of floats with equal dims to input_ids
        rand = torch.rand(input_ids.shape)
        # mask random 15% where token is not 0 [PAD], 1 [CLS], or 2 [SEP]
        mask_arr = (rand < .15) * (input_ids != 0) * (input_ids != 1) * (input_ids != 2)
        # loop through each row in input_ids tensor (cannot do in parallel)
        for i in range(input_ids.shape[0]):
            # get indices of mask positions from mask array
            selection = torch.flatten(mask_arr[i].nonzero()).tolist()
            # mask input_ids
            input_ids[i, selection] = 3  # our custom [MASK] token == 3
        
        return {
            "input_ids": input_ids,
            "attention_mask": mask,
            "labels": labels
        }
    
class Dataset(Dataset):
    def __init__(self, encodings):
        # store encodings internally
        self.encodings = encodings

    def __len__(self):
        # return the number of samples
        print(self.encodings)
        return self.encodings['input_ids'].shape[0]

    def __getitem__(self, i):
        # return dictionary of input_ids, attention_mask, and labels for index i
        return {
            key: tensor[i] for key,
            tensor in self.encodings.items()
        }



class CommonLitRoBERTa(nn.Module):
    def __init__(self, pretrained_path):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(pretrained_path)

    def forward(self, ids, mask):
        output = self.roberta(ids, attention_mask=mask)
        return output

def train_tokenizer(textos, tokenizer):
    # load dataset and train data
    encodings = TrainValidDataset(textos, tokenizer, Settings.max_len)
    print(encodings["input_ids"])

    dataset = Dataset(encodings)
    loader = DataLoader(dataset, batch_size=Settings.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # inicialize batch
    batch = next(iter(loader))

    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    epochs = 2

    for epoch in range(epochs):
        # setup loop with TQDM and dataloader
        loop = tqdm(loader, leave=True)
        for batch in loop:
            # initialize calculated gradients (from prev step)
            optim.zero_grad()
            # pull all tensor batches required for training
            input_ids = batch['input_ids'].to(Settings.device)
            attention_mask = batch['attention_mask'].to(Settings.device)
            labels = batch['labels'].to(Settings.device)
            # process
            outputs = model(input_ids, attention_mask=attention_mask,
                            labels=labels)
            # extract loss
            loss = outputs.loss
            # calculate loss for every parameter that needs grad update
            loss.backward()
            # update parameters
            optim.step()
            # print relevant info to progress bar
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())

    return model
    

# RUN MODELS ******************************************************************************
match config_modelo_escolhido:
    case 'roberta-base':
        # RoBERTa base model
        model = CommonLitRoBERTa("roberta-base")
        model.to(Settings.device)
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        roberta_model = train_tokenizer(textos, tokenizer)
        torch.save(roberta_model, "roberta-model.ckpt")

    case 'roberta-base-tokenizer-fast':
        #RoBERTa Fast Tokenizer
        from transformers import RobertaTokenizerFast
        tokenizer_fast = RobertaTokenizerFast.from_pretrained("roberta-base")
        roberta_model = train_tokenizer(textos, tokenizer_fast)
        torch.save(roberta_model, torch.optim.AdamWd("roberta-base"))
        config.is_decoder = True
        model = RobertaForCausalLM.from_pretrained("roberta-base", config=config)
        model.to(Settings.device)
        roberta_model = train_tokenizer(textos, tokenizer_causallm)
        torch.save(roberta_model, "roberta-base-causal-ml-model.ckpt")

    case 'bertimbau-base':
        # BERT Base
        from transformers import AutoModel, AutoTokenizer
        tokenizer_bertimbau_base = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
        model = AutoModel.from_pretrained('neuralmind/bert-base-portuguese-cased')
        model.to(Settings.device)
        bertimbau_model = train_tokenizer(textos, tokenizer_bertimbau_base)
        torch.save(bertimbau_model, "bertimbau-base-model.ckpt")

    case 'bertimbau-large':
        # BERT Large
        from transformers import AutoModel, AutoTokenizer
        tokenizer_bertimbau_large = AutoTokenizer.from_pretrained('neuralmind/bert-large-portuguese-cased')
        model = AutoModel.from_pretrained('neuralmind/bert-large-portuguese-cased')
        model.to(Settings.device)
        bertimbau_model = train_tokenizer(textos, tokenizer_bertimbau_large)
        torch.save(bertimbau_model, "bertimbau-large-model.ckpt")