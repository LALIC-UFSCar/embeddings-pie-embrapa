import os
import sys
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaForMaskedLM, RobertaTokenizer, RobertaConfig
from tqdm.auto import tqdm

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
# CONFIGURATIONS ******************************************************************************

# initialize the tokenizer using the tokenizer we initialized and saved to file
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

content = []
for t in textos_com_caminhos[:100]:
    with open(t, 'r', encoding='utf-8') as fp:
        content.append(fp.read())

batch = tokenizer(content, max_length=512, padding='max_length', truncation=True)
print(len(batch))

labels = torch.tensor(batch["input_ids"])
mask = torch.tensor(batch["attention_mask"])

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

print(input_ids.shape)

print(input_ids[0][:200])

encodings = {'input_ids': input_ids, 'attention_mask': mask, 'labels': labels}

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        # store encodings internally
        self.encodings = encodings

    def __len__(self):
        # return the number of samples
        return self.encodings['input_ids'].shape[0]

    def __getitem__(self, i):
        # return dictionary of input_ids, attention_mask, and labels for index i
        return {key: tensor[i] for key, tensor in self.encodings.items()}

dataset = Dataset(encodings)

loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

config = RobertaConfig(
    vocab_size=50_265,  # we align this to the tokenizer vocab_size
    max_position_embeddings=514,
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1
)

model = RobertaForMaskedLM(config)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# and move our model over to the selected device
model.to(device)

# activate training mode
model.train()
# initialize optimizer
optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

epochs = 2

for epoch in range(epochs):
    # setup loop with TQDM and dataloader
    loop = tqdm(loader, leave=True)
    for batch in loop:
        # initialize calculated gradients (from prev step)
        optim.zero_grad()
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        # process
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        # extract loss
        loss = outputs.loss
        # calculate loss for every parameter that needs grad update
        loss.backward()
        # update parameters
        optim.step()
        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())



