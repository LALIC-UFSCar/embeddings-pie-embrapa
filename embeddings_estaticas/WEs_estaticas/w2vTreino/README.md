Código para geração de *word embeddings* estáticas a nível de palavra por meio do método Word2Vec.

Para rodar execute:
```bash
python w2v.py <pasta-de-arquivos-txt-corpus> <min_count> <janela_de_contexto> <epochs> <nome_do_modelo>
```
Onde:
- `pasta-de-arquivos-txt-corpus`: Path para pasta que contém o córpus
- `min-count`: O modelo não irá gerar embeddings para palavras que ocorrem menos de min-count vezes no córpus
- `janela-de-contexto`: Tamanho da janela de contexto, em palavras, usada durante o treinamento
- `epochs`: Número de épocas de treinamento
- `nome-do-modelo`: Nome do arquivo .model do modelo que será salvo