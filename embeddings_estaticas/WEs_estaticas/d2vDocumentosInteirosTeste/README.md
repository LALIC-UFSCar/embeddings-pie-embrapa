Código para buscar os 10 documentos mais similares a um texto de entrada usando as *embeddings* geradas pelo Doc2Vec. 

Para rodar execute:
```bash
python d2vdocsteste.py <pasta-de-arquivos-txt-corpus> <nome-do-modelo> <texto>
```
Onde:
- `pasta-de-arquivos-txt-corpus`: Path para pasta que contém o córpus
- `nome-do-modelo`: Nome do modelo Doc2Vec de documentos salvo
- `texto`: O texto que se deseja buscar os 10 documentos mais similares no córpus
