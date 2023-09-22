import os
import sys
import numpy as np
import random
import torch
from transformers import RobertaConfig, RobertaForMaskedLM, RobertaTokenizer, RobertaTokenizerFast, pipeline
from transformers import pipeline, BertModel, BertConfig, BertTokenizer, BertForPreTraining
from tqdm.auto import tqdm

# GET PARAMETERS ******************************************************************************
opcoes_modelos = ['roberta-base', 'roberta-base-tokenizer-fast', 'roberta-base-causal-ml', 'bertimbau-base', 'bertimbau-large', 'bertimbau-automatic']

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
            print("\troberta-base | roberta-base-tokenizer-fast | roberta-base-causal-ml | bertimbau-base | bertimbau-large | bertimbau-automatic")
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
    print("\tExemplo: python3 embeddings_dinamicas.py /home/corpus/ bertimbau-automatic")
    sys.exit()

try:
    textos = os.listdir(pasta)
except:
    print("Nome de pasta inválida")
    sys.exit()

textos_com_caminhos = []
for t in textos:
    textos_com_caminhos.append(pasta+t)

text_content = []
for t in textos_com_caminhos:
    with open(t, 'r', encoding='utf-8') as fp:
        text_content.append(fp.read())

# CONFIGURATIONS ******************************************************************************
class Settings:
    batch_size= int(sys.argv[3]) if qtde_parametros >= 4 else 8
    epochs = int(sys.argv[4]) if qtde_parametros >= 4 else 8
    max_len= 512
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 318

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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True

set_seed(Settings.seed)

def tokenize_texts(textos, tokenizer):
    batch = tokenizer(textos, max_length=Settings.max_len, padding='max_length', truncation=True)
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

    return {'input_ids': input_ids, 'attention_mask': mask, 'labels': labels}

def train_model(textos, tokenizer, model, model_save_path):
    encodings = tokenize_texts(textos, tokenizer)
    dataset = Dataset(encodings)
    loader = torch.utils.data.DataLoader(dataset, batch_size=Settings.batch_size, shuffle=True)

    # and move our model over to the selected device
    model.to(Settings.device)
    # activate training mode
    model.train()
    # initialize optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    epochs = Settings.epochs

    min_loss = 100
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

            if loss.item() < min_loss:
                min_loss = loss.item()
                model.save_pretrained(model_save_path)

# RUN MODELS ******************************************************************************
match config_modelo_escolhido:
    case 'roberta-base':
        # RoBERTa base model
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        config = RobertaConfig(
            vocab_size=50_265,  # we align this to the tokenizer vocab_size
            max_position_embeddings=514,
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=6,
            type_vocab_size=1
        )
        model = RobertaForMaskedLM.from_pretrained("roberta-base", config=config)
        train_model(textos, tokenizer, model, './roberta-base-roberta-masked-ml')

    case 'roberta-base-tokenizer-fast':
        #RoBERTa Fast Tokenizer
        tokenizer_fast = RobertaTokenizerFast.from_pretrained("roberta-base")
        config = RobertaConfig(
            vocab_size=50_265,  # we align this to the tokenizer vocab_size
            max_position_embeddings=514,
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=6,
            type_vocab_size=1
        )
        model = RobertaForMaskedLM.from_pretrained("roberta-base", config=config)
        train_model(textos, tokenizer_fast, model, './roberta-base-roberta-masked-ml-fast')

    case 'bertimbau-base':
        # BERT Base
        tokenizer_bertimbau_base = RobertaTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
        model = RobertaForMaskedLM.from_pretrained('neuralmind/bert-base-portuguese-cased')
        train_model(textos, tokenizer_bertimbau_base, model, './bertimbau-base-roberta-marked-ml')

    case 'bertimbau-large':
        # BERT Large
        tokenizer_bertimbau_large = RobertaTokenizer.from_pretrained('neuralmind/bert-large-portuguese-cased')
        model = RobertaForMaskedLM.from_pretrained('neuralmind/bert-large-portuguese-cased')
        train_model(textos, tokenizer_bertimbau_large, model, './bertimbau-large-roberta-marked-ml')

    #PARA TESTAR
    case 'bertimbau-automatic':
        # Auto Pipeline from Transformers
        config = BertConfig(
            vocab_size=50_265,  # we align this to the tokenizer vocab_size
            max_position_embeddings=514,
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=6,
            type_vocab_size=1
        )
        model = BertForPreTraining.from_pretrained('neuralmind/bert-large-portuguese-cased')
        tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-large-portuguese-cased', config = config)
        pipe = pipeline('fill-mask', model=model, tokenizer=tokenizer)

        pipe('Tinha uma [MASK] no meio do caminho.')