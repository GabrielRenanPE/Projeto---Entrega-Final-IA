Classificação dos resultados do Santa Cruz (2015–2025)
Projeto da disciplina de Inteligência Artificial da UABJ–UFRPE. Comparamos seis algoritmos para classificar partidas do Santa Cruz em vitória, empate ou derrota, usando informações disponíveis antes do jogo, como mando de campo, adversário e competição.
Experimento
A base reúne 470 partidas. Duas foram excluídas por divergência entre o placar e o resultado informado. As 468 restantes incluem 181 vitórias, 143 empates e 144 derrotas. Base original no Kaggle.

Testamos Árvore de Decisão, KNN, Naive Bayes, Regressão Logística, SVM e Rede Neural MLP. Cada algoritmo teve três configurações avaliadas por validação cruzada estratificada com 10 folds.

A Regressão Logística com C = 1 obteve o maior F1 macro médio (42,0%). A Árvore de Decisão com profundidade 2 teve a maior acurácia média (45,3%), mas errou todos os empates.
Como executar
No Google Colab, abra o notebook e execute as células em ordem. Quando solicitado, envie o ZIP completo do projeto. [Experimento_Santa_Cruz.ipynb](Experimento_Santa_Cruz.ipynb)

No computador, descompacte o projeto e execute os comandos na pasta que contém o arquivo requirements.txt, usando Python 3.12.
python -m pip install -r requirements.txt
python organizar_planilha.py
python experimento.py
As bases original e pré-processada ficam em dados/. As métricas por fold, as previsões e os gráficos ficam em resultados/. [Artigo em PDF](Artigo_Santa_Cruz.pdf)
Equipe
Gabriel Renan Silva Martins Guimarães, Anthonny Lucas Santos da Silva e Eliel dos Santos Rocha.

Professor: Prof. Dr. Anderson Pinheiro Cavalcanti.
