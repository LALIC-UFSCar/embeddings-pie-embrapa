# Teste de similaridade de documentos inteiros usando embeddings geradas pelo Doc2Vec

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

print('Lendo todos os documentos do corpus usado para treinamento...')
# Lista de armazenamento dos documentos originais
projetos_inteiros = []
# Associa cada índice a um nome de arquivo
arquivos_indices = dict()
# Para cada arquivo de texto
for i, filename in tqdm(enumerate(tqdm(projetos))):
  arquivos_indices[i] = filename
  f = open(os.path.join(path_projetos,filename), encoding='utf8')
  projeto = f.read()
  projetos_inteiros.append(projeto)

d2v_model_name = sys.argv[2]
try:
    d2v_model_inteiros = Doc2Vec.load(d2v_model_name)
except:
    print('Modelo inexistente - não foi possível localizar o modelo')
    raise

# Função que mostra os resultados dos 10 documentoos mais similares ao texto de input
def doc2vec_most_similar(modelo, embedding, documentos, frase_original, dicionario_idx_docs):
    print("\nDocumento original: \"" + frase_original + "\"\n")
    for tupla in modelo.most_similar(embedding):
        print("Documento encontrado: \"" + documentos[tupla[0]] + "\"")
        print("Idx:", tupla[0])
        print("Nome do arquivo:", dicionario_idx_docs[tupla[0]])
        print("Distância:", 1-tupla[1], "\n")

# Pré-processando o texto da mesma maneira que é feita no treinamento
projeto = sys.argv[3]
projeto_tok = []
frases = sent_tokenize(projeto)
for frase in frases:
    tokens = nltk.word_tokenize(frase)  
    processada = [w.lower() for w in tokens if not w.lower() in stopwords and w.isalnum() or '_' in w]
    if processada:
        projeto_tok.append(processada)

# Calculando a embedding do texto de input por meio da função infer_vector do Doc2Vec
projeto_tok_embedding = d2v_model_inteiros.infer_vector(list(flatten(projeto_tok)))

# Recuperando os 10 documentos mais similares ao texto de input
doc2vec_most_similar(d2v_model_inteiros.dv, projeto_tok_embedding, projetos_inteiros, sys.argv[3], arquivos_indices)