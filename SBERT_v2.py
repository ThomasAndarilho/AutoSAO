# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 18:02:59 2023

@author: UQBU
"""

import pandas as pd
from sentence_transformers import SentenceTransformer

path = r"dados.csv"
df = pd.read_csv(path, sep= "|")
df.drop(["Unnamed: 0"], axis=1, inplace = True)
df.dropna(subset = ["TELO_TX_LINHA"], inplace = True)
#df.fillna("", inplace = True)

model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

lista = []
for indice, linha in df.iterrows():
    print(indice, end = "|")
    texto = linha[2]
    # Sentences are encoded by calling model.encode()
    embedding = model.encode(texto).tolist()
    om = linha[0]
    embedding.append(om)
    lista.append(embedding)

df_lista = pd.DataFrame(lista)
# Seta o número da om como índice da tabela de embeddings
df_lista.set_index(512, inplace = True)
# Seta o número da om como índice da tabela de embeddings
df.set_index("NumOrdem", inplace = True)
df_join = df.join(df_lista)
df_join.to_excel("sbert_v2.xlsx")    

