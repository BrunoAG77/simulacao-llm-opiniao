# Percepção dos Brasileiros sobre o Racismo no Brasil

> Simulação de Opinião Pública com LLMs — Pesquisa CESOP/UNICAMP  
> **Universidade Presbiteriana Mackenzie**

## Integrantes do Grupo

| N° USP | Nome |
|--------|------|
| 10417318 | Bruno Antico Galin |
| 10391119 | Cleverson Pereira da Silva |
| 10417877 | Eduardo Takashi Missaka |
| 10418552 | Ismael de Sousa Silva |
| 10410862 | Vitor Alves Pereira |

---

## Visão Geral

Este projeto simula respostas de questionários de opinião pública utilizando um modelo de linguagem de grande escala (LLM) e compara os resultados com os dados reais da pesquisa **04829** do CESOP (Centro de Estudos de Opinião Pública — UNICAMP), cujo tema é a percepção dos brasileiros sobre o racismo no Brasil.

A abordagem segue o artigo de referência *Simulating Public Opinion: Comparing Distributional and Individual-Level Predictions from LLMs and Random Forests*, avaliando acurácia, distribuição das respostas e importância relativa das variáveis preditoras.

---

## Parte 1 — Inicialização e Leitura

### Dependências

```python
!pip install pyreadstat
```

```python
import pandas as pd
import pyreadstat
import torch
import re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_score,
                             recall_score, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from transformers import AutoTokenizer, AutoModelForCausalLM
```

### Leitura dos Dados

```python
df = pd.read_spss('04829.SAV')
print(df.columns)
```

---

### Questões Simuladas

As quatro questões abaixo foram selecionadas como alvo para simulação pelo LLM.

---

#### P01 — Principal fator gerador de desigualdades no Brasil (RU)

> *"Dentre as questões listadas nessa cartela, na sua opinião, qual é a que mais contribui para gerar as desigualdades no Brasil?"*

| Código | Resposta |
|--------|----------|
| 01 | Raça/Cor/Etnia |
| 02 | Classe social |
| 03 | Gênero ou sexo |
| 04 | Local de moradia |
| 05 | Deficiência |
| 06 | Orientação sexual |
| 07 | Local de origem/onde nasceu |
| 97 | Nenhuma destas/Outras |
| 98 | Não sabe |
| 99 | Não respondeu |

```python
df['P1'].replace({
    'Raça/Cor/Etnia': 1,
    'Classe social': 2,
    'Gênero ou sexo': 3,
    'Local de moradia (cidade, região da cidade, bairro, comunidade)': 4,
    'Deficiência (auditiva, visual, intelectual e motora)': 5,
    'Orientação sexual (heterossexual, homossexual)': 6,
    'Local de origem/onde nasceu (país ou região do país)': 7,
    'Nenhuma destas/Outras (sem especificar)': 97,
    'Não sabe/ Não respondeu': 9899,
}, inplace=True)
```

---

#### P02 — Conforto ao declarar raça/cor/etnia (RU)

> *"Agora gostaria de saber como o(a) sr(a) se sente ao ter que responder qual sua raça/cor/etnia. Você diria que se sente:"*

| Código | Resposta |
|--------|----------|
| 01 | Muito confortável |
| 02 | Confortável |
| 03 | Desconfortável |
| 04 | Muito desconfortável |
| 98 | Não sabe |
| 99 | Não respondeu |

```python
df['P2'].replace({
    'Muito confortável': 1,
    'Confortável': 2,
    'Desconfortável': 3,
    'Muito desconfortável': 4,
    'Não sabe/ Não respondeu': 9899,
}, inplace=True)
```

---

#### P03 — Facilidade em definir raça/cor/etnia (RU)

> *"Para o(a) sr(a), definir a sua raça/cor/etnia é muito fácil, fácil, difícil ou muito difícil?"*

| Código | Resposta |
|--------|----------|
| 01 | Muito fácil |
| 02 | Fácil |
| 03 | Difícil |
| 04 | Muito difícil |
| 98 | Não sabe |
| 99 | Não respondeu |

```python
df['P3'].replace({
    'Muito fácil': 1,
    'Fácil': 2,
    'Difícil': 3,
    'Muito difícil': 4,
    'Não sabe/ Não respondeu': 9899,
}, inplace=True)
```

---

#### P04 — Importância de declarar raça/cor/etnia (RU)

> *"Na sua opinião, declarar a sua raça/cor/etnia é muito, pouco ou nada importante?"*

| Código | Resposta |
|--------|----------|
| 01 | Muito importante |
| 02 | Pouco importante |
| 03 | Nada importante |
| 98 | Não sabe |
| 99 | Não respondeu |

```python
df['P4'].replace({
    'Muito importante': 1,
    'Pouco importante': 2,
    'Nada importante': 3,
    'Não sabe/ Não respondeu': 9899,
}, inplace=True)
```

---

### Configuração das Variáveis

```python
targets  = ['P1', 'P2', 'P3', 'P4']
features = ['SEXO', 'IDADE', 'RACA', 'ESCOLARIDADE',
            'UF', 'MUNI', 'REGIAO', 'RELIGIAO']

# True  → executa o LLM (requer GPU)
# False → carrega resultados pré-gerados de arquivo pickle
executar_llm = False
```

---

## Parte 2 — Modelo LLM

### Preparação do DataFrame

```python
dfmodel = df[features + targets]
```

### Funções Auxiliares

```python
def extrair_numero(resposta):
    """Extrai o número inteiro da resposta gerada pelo modelo."""
    match = re.search(r'\d+', resposta)
    return int(match.group()) if match else None
```

### Construção do Prompt

```python
def send_input(lin, pergunta):
    """Monta o prompt com o perfil do respondente e a pergunta."""
    opcoes_txt = "\n".join([f"{k} {v}" for k, v in pergunta["opcoes"].items()])
    return f"""
Perfil:
Idade: {lin['IDADE']}
Sexo: {lin['SEXO']}
Raça: {lin['RACA']}

Pergunta:
{pergunta["texto"]}

Opções:
{opcoes_txt}

Responda apenas com um número:
"""
```

> **Nota sobre o prompt:** O perfil inclui idade, sexo e raça do respondente. As opções são enumeradas e o modelo é instruído a retornar **somente o número** correspondente à alternativa escolhida, facilitando o parsing da resposta.

### Geração em Lote (Batch Inference)

```python
def gerar(prompts, batch_size=8):
    """Envia prompts em lotes para o modelo e retorna as respostas decodificadas."""
    respostas = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, padding_side='left').to(device)
        model.to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=2,
            do_sample=True,
            temperature=0.3,
            top_p=0.8,
            pad_token_id=tokenizer.eos_token_id
        )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for p, d in zip(batch, decoded):
            respostas.append(d[len(p):].strip())
    return respostas
```

### Execução por Pergunta

```python
def executar(df, pergunta, nsamples=200):
    """Amostra respondentes, gera prompts e retorna previsões do LLM."""
    dfsample = df.sample(nsamples)
    prompts  = [send_input(lin, pergunta) for lin in dfsample.to_dict("records")]
    respostas = gerar(prompts, batch_size=8)
    y_llm = [extrair_numero(r) for r in respostas]
    return dfsample, y_llm
```

### Loop Principal de Simulação

O modelo **TinyLlama-1.1B-Chat-v1.0** é executado em **3 repetições**, com **200 amostras cada** (~10% do dataset original). Os resultados são salvos em pickle para reaproveitamento sem re-execução.

```python
metricasllm = []

if executar_llm:
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    for rep in range(3):
        print(f"\n=====Repetição {rep}=====")
        resultados = {}

        for id, pergunta in perguntas.items():
            dfsample, y_llm = executar(dfmodel, pergunta, nsamples=200)
            resultados[id]  = {"dfsample": dfsample, "y_llm": y_llm}

            y_real = resultados[id]["dfsample"][id].reset_index(drop=True)
            y_llm  = pd.Series(resultados[id]["y_llm"]).reset_index(drop=True)

            opcoes_nonull = list(perguntas[id]["opcoes"].keys())
            y_llm = y_llm.apply(lambda x: x if x in opcoes_nonull else None)
            mask  = y_real.notnull() & y_llm.notnull()
            y_real, y_llm = y_real[mask], y_llm[mask]

            metricasllm.append({
                "Repetição": rep + 1,
                "Pergunta":  id,
                "YReal":     y_real.value_counts(normalize=True),
                "YLLM":      y_llm.value_counts(normalize=True),
                "Acurácia":  accuracy_score(y_real, y_llm),
                "Precisão":  precision_score(y_real, y_llm, average="weighted"),
                "Recall":    recall_score(y_real, y_llm, average="weighted"),
                "F1":        f1_score(y_real, y_llm, average="weighted"),
                "Report":    classification_report(y_real, y_llm),
            })

        with open("resultados.pkl",  "wb") as f: pickle.dump(resultados, f)
        with open("metricasllm.pkl", "wb") as f: pickle.dump(metricasllm, f)
else:
    with open("resultados.pkl",  "rb") as f: resultados  = pickle.load(f)
    with open("metricasllm.pkl", "rb") as f: metricasllm = pickle.load(f)

dfmetricas = pd.DataFrame(metricasllm)
```

### Métricas Médias do LLM

```python
dfmetricas = dfmetricas.groupby("Pergunta")[["Acurácia","Precisão","Recall","F1"]].mean().reset_index()
```

### Visualizações — LLM

```python
# Acurácia por pergunta
sns.barplot(x="Pergunta", y="Acurácia", data=dfmetricas, color="red")
plt.title("Barplot - Acurácia do LLM por pergunta")
plt.show()

# Boxplot de Precisão, Recall e F1
sns.boxplot(data=dfmetricas[["Precisão","Recall","F1"]])
plt.title("Boxplot - Precisão, Recall e F1-Score do LLM por pergunta")
plt.ylabel("Percentual")
plt.show()

# Mapa de calor de métricas
metricas_calor = dfmetricas.set_index("Pergunta")[["Acurácia","Precisão","Recall","F1"]]
sns.heatmap(metricas_calor, annot=True)
plt.title("Mapa de Calor - Métricas do LLM por pergunta")
plt.show()

# Comparação distribuição real vs LLM
for id in resultados:
    dist_y_real = resultados[id]["dfsample"][id].value_counts(normalize=True).sort_index()
    dist_y_llm  = pd.Series(resultados[id]["y_llm"]).value_counts(normalize=True).sort_index()
    dist = pd.DataFrame({
        "Pergunta": dist_y_real.index,
        "Real": dist_y_real.values,
        "LLM":  dist_y_llm.reindex(dist_y_real.index, fill_value=0).values,
    })
    longdist = dist.melt(id_vars="Pergunta", var_name="Origem", value_name="Distribuição")
    sns.barplot(x="Pergunta", y="Distribuição", data=longdist, hue="Origem", errorbar=None)
    plt.xlabel(f"{id}")
    plt.show()
```

---

## Parte 3 — Modelo de Aprendizado Supervisionado (Random Forest)

### Treinamento e Avaliação

```python
rfmetricas = []

for t in targets:
    print(f"\n====={t}=====")

    dfrf = dfmodel.copy()
    dfrf[t] = pd.to_numeric(dfrf[t])
    dfrf = dfrf[dfrf[t] != 9899]           # remove "Não sabe/Não respondeu"
    dfrf[features] = dfrf[features].astype(str).fillna("Desconhecido")
    dfrf = dfrf.dropna(subset=[t])

    x = pd.get_dummies(dfrf[features])     # one-hot encoding
    y = dfrf[t]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    rfmodel = RandomForestClassifier(n_estimators=100, random_state=42)
    rfmodel.fit(x_train, y_train)
    y_pred = rfmodel.predict(x_test)

    rfmetricas.append({
        "Pergunta": t,
        "Acurácia": accuracy_score(y_test, y_pred),
        "Precisão": precision_score(y_test, y_pred, average="weighted"),
        "Recall":   recall_score(y_test, y_pred, average="weighted"),
        "F1":       f1_score(y_test, y_pred, average="weighted"),
        "Report":   classification_report(y_test, y_pred),
    })

dfrfmetricas = pd.DataFrame(rfmetricas)
```

### Visualizações — Random Forest

```python
# Acurácia por pergunta
sns.barplot(x="Pergunta", y="Acurácia", data=dfrfmetricas, color="red", errorbar=None)
plt.title("Barplot - Acurácia do RandomForest por pergunta")
plt.show()

# Boxplot de Precisão, Recall e F1
sns.boxplot(data=dfrfmetricas[["Precisão","Recall","F1"]])
plt.title("Boxplot - Precisão, Recall e F1-Score do RandomForest por pergunta")
plt.ylabel("Percentual")
plt.show()

# Mapa de calor
metricasrf_calor = dfrfmetricas.set_index("Pergunta")[["Acurácia","Precisão","Recall","F1"]]
sns.heatmap(metricasrf_calor, annot=True)
plt.title("Mapa de Calor - Métricas do RandomForest por pergunta")
plt.show()
```

### Importância das Variáveis (Explicabilidade)

```python
dfimportance = pd.DataFrame({
    "Feature":    x.columns,
    "Importância": rfmodel.feature_importances_,
})

# Top 10 features (com dummies)
dfimportance.sort_values("Importância", ascending=False).head(10)

# Agrupamento por variável original (sem dummies)
dfimportance["Original"] = dfimportance["Feature"].str.split("_").str[0]
importance_nodummies = (
    dfimportance.groupby("Original")["Importância"]
    .sum().reset_index()
    .sort_values("Importância", ascending=False)
)
importance_nodummies
```

---

## Comparação Final — LLM vs. Random Forest

```python
llmxrf = pd.DataFrame({
    "Pergunta": dfrfmetricas["Pergunta"],
    "LLM": dfmetricas["Acurácia"],
    "RF":  dfrfmetricas["Acurácia"],
})

longlxr = llmxrf.melt(id_vars="Pergunta", value_vars=["LLM","RF"],
                       var_name="Modelo", value_name="Acurácia")
sns.barplot(x="Pergunta", y="Acurácia", data=longlxr, hue="Modelo", errorbar=None)
plt.title("Gráfico Final - Acurácia do LLM vs RandomForest")
plt.show()
```

---

## Resumo Técnico

| Aspecto | Detalhe |
|---------|---------|
| **Pesquisa** | CESOP 04829 — Percepção dos brasileiros sobre racismo |
| **Modelo LLM** | TinyLlama/TinyLlama-1.1B-Chat-v1.0 (open-source) |
| **Modelo Supervisionado** | RandomForestClassifier (sklearn, 100 estimadores) |
| **Questões simuladas** | P1 (8 alternativas), P2, P3, P4 (4 alternativas cada) |
| **Features utilizadas** | SEXO, IDADE, RAÇA, ESCOLARIDADE, UF, MUNICÍPIO, REGIÃO, RELIGIÃO |
| **Amostras por rodada** | 200 respondentes (~10% do dataset) |
| **Repetições** | 3 (validação cruzada manual) |
| **Métricas avaliadas** | Acurácia, Precisão, Recall, F1-Score, Classification Report |
| **Execução** | 100% open-source, compatível com Google Colab (GPU recomendada para LLM) |

---

---

## Reprodutibilidade e Dados

### Arquivos Disponibilizados

Este projeto é **100% reprodutível** e open-source. Os seguintes artefatos estão inclusos:

| Arquivo | Tamanho | Descrição |
|---------|---------|-----------|
| `04829.SAV` | 373 KB | Dataset original (SPSS) — 2.085 respondentes, 8 features, 4 perguntas alvo |
| `metricasllm.pkl` | 16 KB | Métricas consolidadas do LLM (12 registros: 3 repetições × 4 questões) |
| `resultados.pkl` | 136 KB | Respostas simuladas e dados dos respondentes (200 amostras por questão) |

### Como Reproduzir

1. **Carregar o dataset:**
   ```python
   import pyreadstat
   df, meta = pyreadstat.read_sav('04829.SAV')
   ```

2. **Carregar métricas já calculadas:**
   ```python
   import pickle
   with open('metricasllm.pkl', 'rb') as f:
       metricasllm = pickle.load(f)
   dfmetricas = pd.DataFrame(metricasllm)
   ```

3. **Carregar resultados das simulações:**
   ```python
   with open('resultados.pkl', 'rb') as f:
       resultados = pickle.load(f)
   # resultados contém: P1, P2, P3, P4
   # cada um com 'dfsample' e 'y_llm'
   ```

### Características do Dataset

- **Respondentes**: 2.085 registros
- **Features**: SEXO, IDADE, RAÇA, ESCOLARIDADE, UF, MUNICÍPIO, REGIÃO, RELIGIÃO
- **Questões simuladas**: P1, P2, P3, P4 (múltipla escolha, 3–8 alternativas)
- **Amostras por simulação**: 200 respondentes (~10% do total)
- **Repetições LLM**: 3 (validação cruzada)
- **Total de previsões LLM**: 800 (4 questões × 200 amostras)

### Estrutura dos Pickles

**metricasllm.pkl:**
```
[
  {
    'Repetição': 1,
    'Pergunta': 'P1',
    'YReal': pd.Series,           # distribuição real
    'YLLM': pd.Series,            # distribuição predita
    'Acurácia': float,
    'Precisão': float,
    'Recall': float,
    'F1': float,
    'Report': str                 # classification_report completo
  },
  ...  # 11 registros mais
]
```

**resultados.pkl:**
```
{
  'P1': {
    'dfsample': DataFrame,        # 200 respondentes com features
    'y_llm': list                 # 200 previsões (inteiros)
  },
  'P2': {...},
  'P3': {...},
  'P4': {...}
}
```

### Ambiente de Execução

✅ **Totalmente compatível com Google Colab**  
✅ **100% open-source** (sem API keys privadas)  
✅ **GPU recomendada** para execução do LLM (ou configure `executar_llm = False` para usar resultados pré-salvos)  
✅ **Dependências instaláveis via pip**: `pandas`, `pyreadstat`, `torch`, `transformers`, `scikit-learn`, `seaborn`, `matplotlib`

---

## Referências

- **Cesop/UNICAMP.** *Pesquisa 04829 — Percepção dos brasileiros sobre o racismo no Brasil.* https://www.cesop.unicamp.br/
- **Argyle, L. P. et al.** *Simulating Public Opinion: Comparing Distributional and Individual-Level Predictions from LLMs and Random Forests.* (artigo de referência do projeto)
- **Zhang, P. et al.** *TinyLlama: An Open-Source Small Language Model.* 2024. https://github.com/jzhang38/TinyLlama
- **Pedregosa, F. et al.** *Scikit-learn: Machine Learning in Python.* JMLR, 2011. https://scikit-learn.org/
- **Falcon, R. & Campos, A.** *Hugging Face Transformers: State-of-the-art Natural Language Processing.* https://huggingface.co/docs/transformers/
