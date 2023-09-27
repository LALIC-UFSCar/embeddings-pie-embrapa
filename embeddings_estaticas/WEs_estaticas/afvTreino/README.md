Código para geração de *word embeddings* estáticas de sentenças de um córpus usando a média das embeddings das palavras que as compõem. Tais embeddings podem ter sido geradas por qualquer método, como o Word2Vec ou FastText. 

Para rodar execute:
```bash
python afv.py <pasta-de-arquivos-txt-corpus> <nome-do-modelo> <nome-do-arquivo-de-embeddings>
```
Onde:
- `pasta-de-arquivos-txt-corpus`: Path para pasta que contém o córpus
- `nome-do-modelo`: Nome do modelo (Word2Vec, FastText...) salvo de embeddings de palavras
- `nome-do-arquivo-de-embeddings`: Nome do arquivo que irá conter as embeddings das sentenças do córpus
