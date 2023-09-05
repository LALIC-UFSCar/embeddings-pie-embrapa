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

files_dic = {"textos": textos}
df_train = pd.DataFrame(files_dic)

# CONFIGURATIONS ******************************************************************************
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
    return output
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

# RUN MODELS ******************************************************************************
match config_modelo_escolhido:
    case 'roberta-base':
        # RoBERTa base model
        model = CommonLitRoBERTa("roberta-base")
        model.to(Settings.device)
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        roberta_model = train_tokenizer(df_train, tokenizer)
        torch.save(roberta_model, "roberta-model.ckpt")

    case 'roberta-base-tokenizer-fast':
        #RoBERTa Fast Tokenizer
        from transformers import RobertaTokenizerFast
        tokenizer_fast = RobertaTokenizerFast.from_pretrained("roberta-base")
        roberta_model = train_tokenizer(df_train, tokenizer_fast)
        torch.save(roberta_model, "roberta-base-model.ckpt")

    case 'roberta-base-causal-ml':
        #RoBERTa for Causal ML
        from transformers import AutoTokenizer, RobertaForCausalLM, AutoConfig
        tokenizer_causallm = AutoTokenizer.from_pretrained("roberta-base")
        config = AutoConfig.from_pretrained("roberta-base")
        config.is_decoder = True
        model = RobertaForCausalLM.from_pretrained("roberta-base", config=config)
        model.to(Settings.device)
        roberta_model = train_tokenizer(df_train, tokenizer_causallm)
        torch.save(roberta_model, "roberta-base-causal-ml-model.ckpt")

    case 'bertimbau-base':
        # BERT Base
        from transformers import AutoModel, AutoTokenizer
        tokenizer_bertimbau_base = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
        model = AutoModel.from_pretrained('neuralmind/bert-base-portuguese-cased')
        model.to(Settings.device)
        bertimbau_model = train_tokenizer(df_train, tokenizer_bertimbau_base)
        torch.save(bertimbau_model, "bertimbau-base-model.ckpt")

    case 'bertimbau-large':
        # BERT Large
        from transformers import AutoModel, AutoTokenizer
        tokenizer_bertimbau_large = AutoTokenizer.from_pretrained('neuralmind/bert-large-portuguese-cased')
        model = AutoModel.from_pretrained('neuralmind/bert-large-portuguese-cased')
        model.to(Settings.device)
        bertimbau_model = train_tokenizer(df_train, tokenizer_bertimbau_large)
        torch.save(bertimbau_model, "bertimbau-large-model.ckpt")