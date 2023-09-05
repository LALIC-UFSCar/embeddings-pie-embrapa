# Gerando Word Embeddings Estáticas para o córpus de texto dos projetos da Embrapa

print('Importando bibliotecas...')
import os
import gensim
import time
from tqdm import tqdm 
from nltk import sent_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('portuguese')

# Recuperando os nomes dos arquivos que contém os textos anotados dos projetos 
# (termos do domínio são destacados com <>, sendo que, se o termo é composto 
# de mais de uma palavra, as mesmas são unidas com _ ):

print('Digite o path para os textos:')
path_projetos = input()
try:
    projetos = os.listdir(path_projetos)
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


print('Gerando WEs estáticas com o Word2Vec...')
from gensim.models import Word2Vec

start_time = time.time()

w2v_embeddings = Word2Vec(sentencas, 
                          min_count=3,     # Ignora palavras que ocorrem menos de 3 vezes
                          vector_size=300, # Dimensão das embeddings
                          workers=8,       # Número de processadores (paralelização)
                          window=5,        # Tamanho da janela de contexto, em palavras, usada durante o treinamento
                          epochs=30)       # Número de épocas de treinamento
print('Tempo para geração das embeddings com Word2Vec:', round((time.time() - start_time),2), 'segundos')
print('Dimensão do vocabulário obtido:', len(w2v_embeddings.wv))

# Salvando o modelo:
print('Digite o nome do modelo para salvar:')
w2v_model_name = input()
w2v_embeddings.save(w2v_model_name + '.model')

print('Gerando WEs estáticas com o FastText...')
from gensim.models.fasttext import FastText

start_time = time.time()

ft_embeddings = FastText(sentencas, 
                         min_count=3,     # Ignora palavras que ocorrem menos de 3 vezes
                         vector_size=300, # Dimensão das embeddings
                         workers=8,       # Número de processadores (paralelização)
                         window=5,        # Tamanho da janela de contexto, em palavras, usada durante o treinamento
                         epochs=10)       # Número de épocas de treinamento
print('Tempo para geração das embeddings com FastText:', round((time.time() - start_time),2), 'segundos')
print('Dimensão do vocabulário obtido:', len(ft_embeddings.wv))

# Observa-se que a dimensão do vocabulário é a mesma obtida pelo Word2Vec dado que 
# foram ignoradas palavras que ocorrem menos de 3 vezes no corpus em ambos os casos, 
# sendo, assim, geradas embeddings para as demais palavras.

# Salvando o modelo:
print('Digite o nome do modelo para salvar:')
ft_model_name = input()
ft_embeddings.save(ft_model_name + '.model')

print('--FIM DE EXECUÇÃO--')