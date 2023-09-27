Código para buscar as 10 sentenças mais similares a uma sentença de entrada usando as *embeddings* geradas pelo método da média das embeddings das palavras (average feature vector). 

Para rodar execute:
```bash
python afvteste.py <pasta-de-arquivos-txt-corpus> <nome-do-modelo> <nome-do-arquivo-de-embeddings> <sentença>
```
Onde:
- `pasta-de-arquivos-txt-corpus`: Path para pasta que contém o córpus
- `nome-do-modelo`: Nome do modelo (Word2Vec, FastText...) salvo de embeddings de palavras usado para gerar as embeddings das sentenças
- `nome-do-arquivo-de-embeddings`: Nome do arquivo que contém as embeddings das sentenças do córpus
- `sentença`: A sentença que se deseja buscar as 10 mais similares no córpus
