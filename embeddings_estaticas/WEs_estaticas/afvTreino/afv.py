# Gerando Word Embeddings Estáticas para sentenças usando a média das embeddings das palavras que as compõem

print('Importando bibliotecas...')
import os
import sys
from tqdm import tqdm 
from nltk import sent_tokenize
import nltk
import numpy as np
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
# Lista de armazenamento de sentenças
sentencas  = []
# Para cada arquivo de texto
for filename in tqdm(projetos):
    f = open(os.path.join(path_projetos,filename), encoding='utf8')
    projeto = f.read()
    # sent_tokenize() separa as frases do texto
    frases = sent_tokenize(projeto)
    # Para cada frase
    for frase in frases:
      # word_tokenize() recupera os tokens em uma frase
      tokens = nltk.word_tokenize(frase)  
      # Para cada token, o mesmo é armazenado em sua forma minúscula caso não seja stopword
      # e seja alfanumérico. Ademais, se há "_" no token significa que este é um n-grama do
      # domínio e, portanto, é também armazenado
      processada = [w.lower() for w in tokens if not w.lower() in stopwords and w.isalnum() or '_' in w]
      # Por fim, a sentença processada é adicionada à lista de sentenças
      sentencas.append(processada)

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

# ['esta é a primeira sentenca', 'já esta é a segunda', 'e assim em diante']
sentencas_unidas = [' '.join(frase) for frase in sentencas]

print('\nCalculando os vetores das sentenças a partir das embeddings das palavras...')
sentence_vectors = [
    avg_feature_vector(i, embeddings.wv, 300, embeddings.wv.index_to_key)
    for i in tqdm(sentencas_unidas)
]

print('\nSalvando as embeddings calculadas das sentenças...')
with open(sys.argv[3], "wb") as fp:
    pickle.dump(sentence_vectors, fp)

print('\n--FIM DE EXECUÇÃO--')