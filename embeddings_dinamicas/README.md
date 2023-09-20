# Embeddings Dinâmicas PIE
Este projeto tem o objetivo de gerar embeddings dinâmicas e um modelo linguístico a partir dos textos de projeto da Embapa.

## Pré-requisitos
Antes de executar o projeto, certifique-se de ter os seguintes requisitos atendidos:

- Python 3 instalado
- python3-venv (opcional)

## Instalação
1. Clone este repositório para o seu ambiente local:

``` shell
git clone https://github.com/LALIC-UFSCar/embeddings-pie-embrapa.git
```

2. Crie um ambiente virtual (opcional)

``` shell
python3 -m venv embeddings
source embeddings/bin/activate
```

3. Instale as dependências do projeto:

``` shell
pip install -r requirements.txt
```

## Uso
Para utilizar o código de geração de embeddings é preciso somente rodar o seguinte comando indicando a pasta com os textos a serem analisados:
1. Certifique-se de ter uma pasta com os textos que deseja-se analisar.
2. Habilite o ambiente virtual (opcional):

``` shell
cd <pasta-projeto> && source embeddings/bin/activate
```

3. Execute o seguinte comando:

``` shell
python3 embeddings_dinamicas.py <pasta-de-arquivos-txt-corpus>
```

Opcionalmente, podem ser passados mais dois parâmetros:

``` shell
python3 pipeline_categorizacao.py <pasta-de-arquivos-txt-corpus> <opcao-config-modelo>
```

onde `<opcao-config-modelo>` indica a configuração do modelo e tokenizer usado para gerar as embeddings e treinar o modelo linguístico.
As opções são: `roberta-base`, `roberta-base-tokenizer-fast`, `roberta-base-causal-ml`, `bertimbau-base`, `bertimbau-large`.
Se nenhuma opção é indicada, é usada como padrão a `roberta-base`.
