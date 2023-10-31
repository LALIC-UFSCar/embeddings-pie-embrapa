# Teste de similaridade de sentenças usando embeddings geradas pela média de embeddings de palavras

print('Importando bibliotecas...')
import os
import sys
from tqdm import tqdm 
from nltk import sent_tokenize
import nltk
import numpy as np
from scipy import spatial
from gensim.models import Word2Vec
import pickle
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

model_name = sys.argv[2]
try:
    embeddings = Word2Vec.load(model_name + '.model')
except:
    print('Modelo inexistente - não foi possível localizar o modelo')
    raise

print('Lendo todas as sentenças do corpus usado para treinamento das embeddings das palavras...')
# Lista de armazenamento das sentenças originais
sentencas_original = []
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
      sentencas_original.append(frase)
      arquivos_sentencas[j] = filename
      j = j + 1

def avg_feature_vector(sentence, model, num_features, index2word_set):
    #Inicializamos um vetor com zeros
    feature_vec = np.zeros((num_features, ), dtype='float32')
    #Divindo a sentença em palavras
    words = sentence.split()

    n_words = 0
    for word in words:
        #Se a palavra estiver no vocabulário
        if word in index2word_set:
            n_words += 1
            #Somamos o vetor da palavra ao vetor da sentença
            feature_vec = np.add(feature_vec, model[word])
    if (n_words > 0):
        #Dividimos o vetor da sentença pela quantidade de palavras que ele possui
        feature_vec = np.divide(feature_vec, n_words)
    else:
        return None
    
    #Retornamos o vetor da sentença
    return feature_vec

# Carregando as embeddings das sentenças
with open(sys.argv[3], "rb") as fp:
    sentence_vectors = pickle.load(fp)

# Função que calcula a similaridade cosseno de dois vetores de maneira segura
# (caso um dos vetores seja nulo, por exemplo)
def safe_cosine_similarity(vector1, vector2):
    try:
        similarity = 1 - spatial.distance.cosine(vector1, vector2)
        return similarity
    except:
        return -sys.maxsize

# Função que encontra os top N vetores mais similares (pela medida cosseno)
# a um vetor de entrada input_vector
def find_most_similar_vectors(input_vector, vector_list, N):
    # Calcula as similaridades cosseno entre o vetor de entrada e todos os vetores da lista
    similarities = [safe_cosine_similarity(input_vector, vector) for vector in vector_list]
    # Obtém os índices dos N maiores valores de similaridade
    top_indices = np.argsort(similarities)[-N:]
    # Cria uma lista de tuplas (distância, índice)
    most_similar_tuples = [(1 - similarities[i], i) for i in top_indices]

    return most_similar_tuples

# Função que encapsula as outras e mostra os resultados das 10 sentenças mais similares a sentença de input
def afv_most_similar(embeddings, embedding_input, documentos, frase_original, dicionario_idx_sents):
    print("\nDocumento original: \"" + frase_original + "\"\n")
    for vector, index in find_most_similar_vectors(embedding_input, embeddings, 10)[::-1]:
        print("Documento encontrado: \"" + documentos[index] + "\"")
        print("Idx:", index)
        print("Nome do arquivo:", dicionario_idx_sents[index])
        print("Distância:", vector, "\n")

# Pré-processando a frase da mesma maneira que é feita no treinamento
projeto = sys.argv[4]
projeto_tok = []
frases = sent_tokenize(projeto)
for frase in frases:
    tokens = nltk.word_tokenize(frase)  
    processada = [w.lower() for w in tokens if not w.lower() in stopwords and w.isalnum() or '_' in w]
    if processada:
        projeto_tok.append(processada)
# A frase processada é armazenada em uma única string
palavras = [palavra for lista in projeto_tok for palavra in lista]
sentenca_final = ' '.join(palavras)

# Calculando a embedding da frase de input por meio do mesmo método de média
projeto_tok_embedding = avg_feature_vector(sentenca_final, embeddings.wv, 300, embeddings.wv.index_to_key)

# Recuperando as 10 sentenças mais similares a de input
afv_most_similar(sentence_vectors, projeto_tok_embedding, sentencas_original, sys.argv[4], arquivos_sentencas)