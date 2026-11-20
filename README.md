# Detecção de Fraude em Cartões de Crédito

Projeto de Machine Learning aplicado à identificação de transações fraudulentas em um cenário de extremo desbalanceamento de classes, utilizando validação estratificada, ajuste de limiar de decisão e explicabilidade com SHAP.

## 1. O Problema e o Impacto do Desbalanceamento
A base de dados contém transações reais realizadas por titulares de cartões europeus, onde apenas **0,17%** são fraudes (`Class = 1`) e **99,83%** são operações legítimas (`Class = 0`). 
Nesse cenário, a **acurácia engana**: um modelo ingênuo que classifica todas as transações como normais atinge 99,83% de acurácia, mas deixa passar 100% das fraudes. Por isso, a avaliação foi direcionada exclusivamente para a classe minoritária (`1`), priorizando:
- **Recall:** proporção de fraudes reais que o modelo conseguiu capturar (evitando prejuízo financeiro direto).
- **Precisão (Precision):** proporção de alertas de fraude que eram realmente fraudes (evitando bloquear cartões de clientes legítimos).
- **F1-Score e PR-AUC:** equilíbrio entre captura de fraudes e controle de falsos positivos.

## 2. Preparação dos Dados
- **Limpeza:** Remoção de [X] linhas duplicadas identificadas na base original para evitar viés na avaliação.
- **Engenharia de Recursos:** 
  - Transformação logarítmica (`np.log1p`) na coluna `Amount` (`log_amount`) para reduzir a assimetria de valores extremos.
  - Extração da hora do dia (`Hour`) a partir da coluna `Time`, revelando aumento proporcional de fraudes durante a madrugada (0h às 6h).
- **Prevenção de Vazamento de Dados (*Data Leakage*):** Divisão estratificada (`train_test_split` com `stratify=y`, 80% treino / 20% teste) realizada **antes** da padronização (`StandardScaler`), que foi encapsulada dentro de um `Pipeline`.

## 3. Comparação entre Modelos (Classe 1 - Fraude)

| Modelo | Precisão | Recall | F1-Score | PR-AUC |
| :--- | :---: | :---: | :---: | :---: |
| Regressão Logística (Baseline) | [0.XX] | [0.XX] | [0.XX] | [0.XX] |
| Random Forest (`class_weight='balanced'`) | [0.XX] | [0.XX] | [0.XX] | [0.XX] |
| XGBoost (`scale_pos_weight`) | [0.XX] | [0.XX] | [0.XX] | [0.XX] |
| **[Modelo Vencedor + GridSearch + Limiar 0.3]** | **[0.XX]** | **[0.XX]** | **[0.XX]** | **[0.XX]** |

## 4. Limiar de Decisão e Explicabilidade (SHAP)
- **Ajuste de Limiar (`0.5` $\rightarrow$ `0.3`):** Reduzir o corte de probabilidade aumentou o Recall de `[XX%]` para `[XX%]`, capturando `[X]` fraudes adicionais no conjunto de teste com um aumento controlado nos falsos positivos.
- **Achados do SHAP:** A análise global (*Summary Plot*) e local (*Waterfall Plot*) mostrou que as componentes principais **[ex: V14, V17, V12 e V4]** exercem o maior peso na decisão do modelo, onde valores fortemente negativos em `V14` e `V17` empurram a probabilidade diretamente para a classe de fraude.

## 5. Evoluções em Relação à Aula Base
- Criação da variável temporal `Hour` para capturar padrões comportamentais de madrugada.
- Remoção explícita de registros duplicados antes da divisão treino/teste.
- [Adicione aqui outra melhoria que você fez, ex: comparação com SMOTE/Undersampling ou cálculo do valor financeiro total recuperado].
