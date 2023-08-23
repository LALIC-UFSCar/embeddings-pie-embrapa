# Embeddings Dinâmicas PIE
Este projeto tem o objetivo de gerar embeddings dinâmicas dos textos de projeto da Embapa.

## Pré-requisitos
Antes de executar o projeto, certifique-se de ter os seguintes requisitos atendidos:

- Python 3 instalado

## Instalação
1. Clone este repositório para o seu ambiente local:

``` shell
git clone https://github.com/LALIC-UFSCar/embeddings-pie-embrapa.git
```

2. Crie um ambiente virtual (opcional)

``` shell
python3 -m venv pipeline
source pipeline/bin/activate
```

3. Instale as dependências do projeto:

``` shell
pip install -r requirements.txt
```

## Uso
Para utilizar o código de geração de embeddings é preciso somente rodar o seguinte comando indicando a pasta com os textos a serem analisados:
1. Certifique-se de ter uma pasta com os textos que deseja-se analisar.
2. Execute o seguinte comando:

``` shell
python3 embeddings_dinamicas.py /home/corpus
```