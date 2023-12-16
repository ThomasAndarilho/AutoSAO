# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 18:02:59 2023

@author: UQBU
"""

import numpy as np
import pandas as pd
import os
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')  # Baixa os recursos necessários para a tokenização, caso ainda não tenha sido feito

def get_embedding(texto):
    embedding = []
    tokens = word_tokenize(texto)
    for token in tokens:
        try:
            word_embedding = model.wv.get_vector(token.lower())
            embedding.append(word_embedding)
        except KeyError:
            pass
    media_embedding = np.mean(embedding, axis = 0)
    return media_embedding.tolist()     

# Definir o caminho absoluto para o diretório onde o arquivo está localizado
directory_path = r"C:\Users\UQBU\OneDrive - PETROBRAS\Documents\Particular\Doutorado\PPGI 2022\GPT\Projeto final\OM\Github\Petroles\Petrovec_Hibrido_Word2vec"

# Construir o caminho relativo para o arquivo
vectors_path = os.path.join(directory_path, "Publico-Hibrido(Completo+NILC).txt.model.wv.vectors.npy")
syn1neg_path = os.path.join(directory_path, "Publico-Hibrido(Completo+NILC).txt.model.trainables.syn1neg.npy")
model_path = os.path.join(directory_path, "Publico-Hibrido(Completo+NILC).txt.model")

# Carregar o arquivo
word_vectors = np.load(vectors_path)
syn1neg = np.load(syn1neg_path)

# Carregar o modelo principal
model = Word2Vec.load(model_path)

# Definir os vetores de palavras no modelo
model.wv.vectors = word_vectors
#model.trainables.syn1neg = syn1neg

path = r"dados.csv"
df = pd.read_csv(path, sep= "|")
df.drop(["Unnamed: 0"], axis=1, inplace = True)
df.dropna(subset = ["TELO_TX_LINHA"], inplace = True)
#df.fillna("", inplace = True)

lista = []
for indice, linha in df.iterrows():
    print(indice, end = "|")
    texto = linha[2]
    embedding = get_embedding(texto)
    # Se der erro por muitas requisições
    om = linha[0]
    embedding.append(om)
    lista.append(embedding)

df_lista = pd.DataFrame(lista)
# Seta o número da om como índice da tabela de embeddings
df_lista.set_index(100, inplace = True)
# Seta o número da om como índice da tabela de embeddings
df.set_index("NumOrdem", inplace = True)
#df.at[indice, "gpt"] = embedding
df_join = df.join(df_lista)
df_join.to_excel("petroles.xlsx")    
#df["gpt"] = lista

