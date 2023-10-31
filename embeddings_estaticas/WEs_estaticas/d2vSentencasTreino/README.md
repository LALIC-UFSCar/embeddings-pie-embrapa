Código para geração de *word embeddings* estáticas de sentenças por meio do método Doc2Vec.

Para rodar execute:
```bash
python d2vsent.py <pasta-de-arquivos-txt-corpus> <min_count> <janela_de_contexto> <nome_do_modelo>
```
Onde:
- `pasta-de-arquivos-txt-corpus`: Path para pasta que contém o córpus
- `min_count`: O modelo não irá gerar embeddings para palavras que ocorrem menos de min_count vezes no córpus
- `janela_de_contexto`: Tamanho da janela de contexto, em palavras, usada durante o treinamento
- `nome_do_modelo`: Nome do arquivo do modelo que será salvo