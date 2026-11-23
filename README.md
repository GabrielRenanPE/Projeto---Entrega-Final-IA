# Classificação dos resultados do Santa Cruz (2015–2025)

É possível distinguir vitórias, empates e derrotas do Santa Cruz antes de a bola rolar? Neste projeto da disciplina de **Inteligência Artificial da UABJ–UFRPE**, comparamos seis classificadores usando informações disponíveis antes do jogo, como mando de campo, adversário e competição. O placar serve apenas para conferir o resultado de cada partida.

[Artigo em PDF](Artigo_Santa_Cruz.pdf) · [Notebook no Google Colab](https://colab.research.google.com/github/GabrielRenanPE/Projeto---Entrega-Final-IA/blob/main/Experimento_Santa_Cruz.ipynb) · [Dataset no Kaggle](https://www.kaggle.com/datasets/gabrielrenanpe/dados-do-time-santa-cruz-futebol-clube-2015-2025)

## Como executar no Linux

Você precisa de Git e Python 3.12 com suporte a ambientes virtuais. A planilha já está no repositório.

**1. Baixe o projeto e entre na pasta.**

```bash
git clone https://github.com/GabrielRenanPE/Projeto---Entrega-Final-IA.git
cd Projeto---Entrega-Final-IA
```

**2. Crie um ambiente virtual e instale as dependências.**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**3. Prepare os dados e execute o experimento.** No repositório, a planilha está na raiz. Copie-a para a pasta que o programa lê e execute os dois scripts em ordem.

```bash
mkdir -p dados
cp Dados_Santa.xlsx dados/
python organizar_planilha.py
python experimento.py
```

O primeiro script converte a planilha em `dados/jogos_organizados.csv`. O segundo faz a limpeza, treina os modelos e mostra no terminal as métricas e a matriz de confusão. Se uma janela com um gráfico abrir, feche-a para continuar. Para executar em um terminal sem interface gráfica, use `MPLBACKEND=Agg python experimento.py`.

**4. Confira os resultados.** O programa cria a pasta `resultados/`. Para visualizar no terminal a média e o desvio padrão de cada configuração:

```bash
cat resultados/resumo.csv
```

Para repetir o experimento em outro dia, entre na pasta do projeto, ative o ambiente com `source .venv/bin/activate` e execute `python experimento.py`.

## Como executar no Google Colab

1. Abra o [notebook no Colab](https://colab.research.google.com/github/GabrielRenanPE/Projeto---Entrega-Final-IA/blob/main/Experimento_Santa_Cruz.ipynb).
2. Insira uma **célula de código no início** e execute os comandos abaixo. Eles baixam o CSV do repositório diretamente para a pasta esperada pelo notebook.

   ```python
   !mkdir -p dados
   !curl -fsSL https://raw.githubusercontent.com/GabrielRenanPE/Projeto---Entrega-Final-IA/main/jogos_organizados.csv -o dados/jogos_organizados.csv
   ```

3. No menu **Ambiente de execução**, escolha **Executar tudo**. Com o CSV nessa pasta, o notebook pula a solicitação de upload do ZIP.
4. Acompanhe as tabelas e os gráficos abaixo das células. Para baixar as saídas, abra **Arquivos** na barra lateral e entre em `resultados/`.

## Dados e avaliação

Das 470 partidas da base, duas foram excluídas por divergência entre o placar e o resultado informado. Restaram **468 jogos** do ponto de vista do Santa Cruz: 181 vitórias, 143 empates e 144 derrotas.

Avaliamos Árvore de Decisão, KNN, Naive Bayes, Regressão Logística, SVM e Rede Neural MLP. Cada algoritmo teve três configurações testadas nos mesmos dez grupos de partidas (*10-fold cross-validation* estratificada). Em cada rodada, nove grupos foram usados no treino e um no teste.

| Critério | Melhor configuração | Resultado médio |
| --- | --- | ---: |
| F1 macro | Regressão Logística, `C = 1` | 42,0% |
| Acurácia | Árvore de Decisão, profundidade 2 | 45,3% |

A árvore acertou mais jogos no total, mas errou todos os 143 empates. A regressão logística reconheceu 34 deles. O F1 macro dá o mesmo peso às três classes e ajuda a enxergar essa diferença.

## Arquivos gerados

| Arquivo | O que consultar |
| --- | --- |
| `dados/jogos_processados.csv` | As 468 partidas usadas no experimento |
| `resultados/resumo.csv` | Média e desvio padrão por modelo e configuração |
| `resultados/metricas_por_fold.csv` | Acurácia, precisão, recall e F1 de cada rodada |
| `resultados/previsoes.csv` | Resultado real e previsto por partida |
| `resultados/comparacao.pdf` e `resultados/mando.pdf` | Comparação dos modelos e distribuição dos resultados em casa e fora |

## Equipe

Gabriel Renan Silva Martins Guimarães · Anthonny Lucas Santos da Silva · Eliel dos Santos Rocha

**Professor:** Prof. Dr. Anderson Pinheiro Cavalcanti · **Instituição:** Unidade Acadêmica de Belo Jardim, Universidade Federal Rural de Pernambuco.
