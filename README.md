# Análise de Painel Tributário e Inovação

Este projeto executa um pipeline econométrico em painel firma-ano para investigar a relação entre **agressividade tributária** e **inovação corporativa**, com foco em modelos de efeitos fixos e testes de robustez.

De forma resumida, o script `code/run_analysis.py` pode:
- Ler e limpar as bases (`dados.xlsx` e `relatorios-rf.xlsx`);
- Construir o painel final com defasagens, winsorização e proxies tributárias (ABTD, ETRC, GAAPETR);
- Estimar modelos FE (firma e ano), com moderações por fiscalização e setor;
- Gerar saídas principais de regressão, estatísticas descritivas e fluxo amostral;
- Rodar diagnósticos econométricos em arquivos separados (ex.: Spearman, VIF, Hausman e robustez avançada como DK/Wooldridge/Pesaran CD/Jarque-Bera, quando habilitados nas flags).

# cmds
source .venv/bin/activate   
python3 code/run_analysis.py
