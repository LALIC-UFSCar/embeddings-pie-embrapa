# Teste de similaridade de palavras usando embeddings geradas pelo Word2Vec

import sys
from gensim.models import Word2Vec

model_name = sys.argv[1]
try:
    embeddings = Word2Vec.load(model_name + '.model')
except:
    print('Modelo inexistente - não foi possível localizar o modelo')
    raise

print("\n10 palavras mais similares a \"" + sys.argv[2] + "\" segundo o modelo Word2Vec:\n")
i = 1
for palavra, similaridade in embeddings.wv.most_similar(sys.argv[2]):
    print(str(i) + ". \'" + palavra + "\', " + "distância: " + str(1-similaridade))
    i = i + 1
print("")