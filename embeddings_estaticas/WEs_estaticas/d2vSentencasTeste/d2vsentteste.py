# Teste de similaridade de sentenças usando embeddings geradas pelo Doc2Vec

print('Importando bibliotecas...')
import os
import sys
from tqdm import tqdm 
from nltk import sent_tokenize
import nltk
from nltk.lm.preprocessing import flatten
from gensim.models.doc2vec import Doc2Vec
nltk.download('punkt')
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('portuguese')

path_projetos = sys.argv[1]
print(path_projetos)
try:
    projetos = os.listdir(path_projetos)
    projetos.sort()
except:
    print('Path inexistente - não foi possível localizar os textos')
    raise

print('Lendo todas as sentenças do corpus usado para treinamento...')
# Lista de armazenamento das sentenças originais
sentencas  = []
# Dicionário para armazenar qual sentença pertence a qual arquivo
arquivos_sentencas = dict()
j = 0
# Para cada arquivo de texto
for filename in tqdm(projetos):
    f = open(os.path.join(path_projetos,filename), encoding='utf8')
    projeto = f.read()
    # sent_tokenize() separa as frases do texto
    frases = sent_tokenize(projeto)
    # Para cada frase
    for frase in frases:
      sentencas.append(frase)
      arquivos_sentencas[j] = filename
      j = j + 1

d2v_model_name = sys.argv[2]
try:
    d2v_model_sentencas = Doc2Vec.load(d2v_model_name)
except:
    print('Modelo inexistente - não foi possível localizar o modelo')
    raise

# Função que mostra os resultados das 10 sentenças mais similares a sentença de input
def doc2vec_most_similar(modelo, embedding, documentos, frase_original, dicionario_idx_sents):
    print("\nDocumento original: \"" + frase_original + "\"\n")
    for tupla in modelo.most_similar(embedding):
        print("Documento encontrado: \"" + documentos[tupla[0]] + "\"")
        print("Idx:", tupla[0])
        print("Nome do arquivo:", dicionario_idx_sents[tupla[0]])
        print("Distância:", 1-tupla[1], "\n")

# Pré-processando a frase da mesma maneira que é feita no treinamento
projeto = sys.argv[3]
projeto_tok = []
frases = sent_tokenize(projeto)
for frase in frases:
    tokens = nltk.word_tokenize(frase)  
    processada = [w.lower() for w in tokens if not w.lower() in stopwords and w.isalnum() or '_' in w]
    if processada:
        projeto_tok.append(processada)

# Calculando a embedding da frase de input por meio da função infer_vector do Doc2Vec
projeto_tok_embedding = d2v_model_sentencas.infer_vector(list(flatten(projeto_tok)))

# Recuperando as 10 sentenças mais similares a de input
doc2vec_most_similar(d2v_model_sentencas.dv, projeto_tok_embedding, sentencas, sys.argv[3], arquivos_sentencas)