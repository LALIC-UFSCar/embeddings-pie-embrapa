# Gerando Word Embeddings Estáticas para sentenças usando Doc2Vec

print('Importando bibliotecas...')
import os
import time
import sys
from tqdm import tqdm 
from nltk import sent_tokenize
import nltk
from multiprocessing import Pool
import multiprocessing
nltk.download('punkt')
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('portuguese')

# Recuperando os nomes dos arquivos que contém os textos dos projetos (se estiverem
# anotados, termos do domínio são destacados com <>, sendo que, se o termo é composto 
# de mais de uma palavra, as mesmas são unidas com _ ):

path_projetos = sys.argv[1]
print(path_projetos)
try:
    projetos = os.listdir(path_projetos)
    projetos.sort()
except:
    print('Path inexistente - não foi possível localizar os textos')
    raise
print('Número de arquivos encontrados:', len(projetos))

print('Lendo e tokenizando todas as sentenças do corpus...')
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
print('No total há', len(sentencas), 'sentenças no córpus ' + path_projetos + '.')

# Podemos contar quantos tokens e types há nessas sentenças:
from nltk.lm.preprocessing import flatten
tokens = list(flatten(sentencas))
print('Número de tokens:', len(list(flatten(sentencas))))
print('Número de types :', len(list(set(tokens))))


print('Gerando WEs estáticas para sentenças com Doc2Vec...')
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(sentencas)]
start_time = time.time()
d2v_model_sentencas = Doc2Vec(documents, vector_size=300, min_count=int(sys.argv[2]), window=int(sys.argv[3]), workers=multiprocessing.cpu_count())
print('Tempo para geração das embeddings de sentenças com o Doc2Vec:', round((time.time() - start_time),2), 'segundos')

# Salvando o modelo:
d2v_model_name = sys.argv[4]
d2v_model_sentencas.save(d2v_model_name)

print('--FIM DE EXECUÇÃO--')