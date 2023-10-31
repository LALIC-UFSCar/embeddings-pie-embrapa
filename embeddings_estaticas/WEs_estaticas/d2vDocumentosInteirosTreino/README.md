Código para geração de *word embeddings* estáticas de documentos por meio do método Doc2Vec.

Para rodar execute:
```bash
python d2vdocs.py <pasta-de-arquivos-txt-corpus> <min-count> <janela-de-contexto> <nome-do-modelo>
```
Onde:
- `pasta-de-arquivos-txt-corpus`: Path para pasta que contém o córpus
- `min-count`: O modelo não irá gerar embeddings para palavras que ocorrem menos de min-count vezes no córpus
- `janela-de-contexto`: Tamanho da janela de contexto, em palavras, usada durante o treinamento
- `nome-do-modelo`: Nome do arquivo do modelo que será salvo