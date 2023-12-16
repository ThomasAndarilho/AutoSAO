# -*- coding: utf-8 -*-
"""
Código para obter os embeddings usando o BERTimbau

Created on Mon Nov 20 19:56:50 2023
@author: UQBU
"""

    
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel

# Carregue o tokenizer e o modelo BERTimbau pré-treinado
tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
modelo = AutoModel.from_pretrained('neuralmind/bert-base-portuguese-cased')

  
path = r"dados.csv"
df = pd.read_csv(path, sep= "|")
df.drop(["Unnamed: 0"], axis=1, inplace = True)
df.fillna("", inplace = True)

lista = []
for indice, linha in df.iterrows():
    print(indice, end = "|")
#    print(texto)
    texto = linha[2]
    frases = [texto]
    # Tokenize as frases
    tokens = tokenizer.batch_encode_plus(frases, padding=True, truncation=True, max_length=100, return_tensors='pt',)
    # Obtenha os embeddings das frases
    with torch.no_grad():
        embeddings_bert = modelo(**tokens)['last_hidden_state']
    for frase, embedding in zip(frases, embeddings_bert):
        print("Frase:", frase)
        sentence_embedding = torch.mean(embedding, dim=0).tolist()
    om = linha[0]
    sentence_embedding.append(om)
    lista.append(sentence_embedding)

df_lista = pd.DataFrame(lista)
# Seta o número da om como índice da tabela de embeddings
df_lista.set_index(768, inplace = True)
# Seta o número da om como índice da tabela de embeddings
df.set_index("NumOrdem", inplace = True)
#df.at[indice, "gpt"] = embedding
df_join = df.join(df_lista)
df_join.to_excel("BERT.xlsx")    

AutoTokenizer.from_pretrained.batch_encode_plus()