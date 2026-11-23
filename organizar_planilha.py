from datetime import datetime, time
import openpyxl
import pandas as pd

planilha = openpyxl.load_workbook("dados/Dados_Santa.xlsx", data_only=True)
# Padroniza os cabeçalhos das diferentes temporadas.
nomes = {
    "data": "Data", "hora": "Horario", "horario": "Horario",
    "resultado": "Resultado", "fora&casa": "Fora_Casa", "casa&fora": "Fora_Casa",
    "time casa": "Time_Casa", "time fora": "Time_Fora", "casa": "Time_Casa",
    "fora": "Time_Fora", "placar": "Placar", "campeonato": "Campeonato",
    "fases da competição": "Fase", "fase": "Fase", "adversario": "Adversario",
    "coluna 1": "Coluna_1_original"
}

jogos = []
for aba in planilha:
    # Percorre as tabelas de cada aba.
    for tabela in aba.tables.values():
        linhas = aba[tabela.ref]
        cabecalho = [nomes[coluna.name.lower()] for coluna in tabela.tableColumns]
        for linha in linhas[1:]:
            jogo = dict(zip(cabecalho, [celula.value for celula in linha]))
            # Ignora linhas sem partida.
            if not jogo["Data"]:
                continue
            jogo["Ano"] = int(aba.title[-4:])
            jogo["Aba_original"] = aba.title
            jogo["Tabela_original"] = tabela.name
            jogo["Linha_original"] = linha[0].row
            if isinstance(jogo["Data"], datetime):
                jogo["Data"] = jogo["Data"].strftime("%d/%m/%Y")
            if isinstance(jogo.get("Horario"), time):
                jogo["Horario"] = jogo["Horario"].strftime("%H:%M")
            for coluna in ["Placar", "Campeonato"]:
                valor = jogo.get(coluna)
                # Recupera placares que o Excel converteu em datas.
                if isinstance(valor, datetime):
                    jogo[coluna] = f"{valor.day}-{valor.month}"
            for coluna, valor in jogo.items():
                if isinstance(valor, str):
                    jogo[coluna] = valor.strip()
            jogos.append(jogo)

# Mantém as 15 colunas da base organizada.
colunas = ["Ano", "Aba_original", "Tabela_original", "Linha_original", "Resultado",
           "Data", "Horario", "Fora_Casa", "Adversario", "Time_Casa", "Time_Fora",
           "Placar", "Campeonato", "Fase", "Coluna_1_original"]
base = pd.DataFrame(jogos)[colunas]
base.to_csv("dados/jogos_organizados.csv", index=False)
