# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 14:02:33 2023

@author: UQBU

https://codigo-externo.petrobras.com.br/cd-petrobras/model-ops/-/blob/main/hub-modelos-ia/exemplo-consumo-modelo/Python%20OpenAI.ipynb

"""

import os
import openai  # pip install openai
from dotenv import load_dotenv  # pip install python-dotenv
from configparser import ConfigParser, ExtendedInterpolation

config = ConfigParser(interpolation=ExtendedInterpolation())
config.read('config.ini', 'UTF-8')

load_dotenv()

os.environ['REQUESTS_CA_BUNDLE'] = './petrobras-ca-root.pem'

openai.api_type = 'azure'
openai.api_base = config['OPENAI']['OPENAI_API_BASE']
openai.api_version = config['OPENAI']['OPENAI_API_VERSION']
openai.api_key = 'not-used'

# Configuração do prefixo do APIM, que é diferente do prefixo da Azure (não documentada)
from openai.api_resources.abstract import APIResource
APIResource.azure_api_prefix = config['OPENAI']['AZURE_OPENAI_PREFIX']

# Configura headers
headers = {
    'api-key': config['OPENAI']['OPENAI_API_KEY'],
    'Content-Type': 'application/json',
    'Cache-Control': 'no-cache'
}

def get_embedding(text, engine, headers, **kwargs):

    try:
        response = openai.Embedding.create(
            # Contact your team admin to get the name of your engine or model deployment.  
            # This is the name that they used when they created the model deployment.
            input=text,
            engine=engine,
            headers=headers,
            **kwargs
        )
    
        embeddings = response['data'][0]['embedding']
        return embeddings

    except openai.error.APIError as e:
        # Handle API error here, e.g. retry or log
        print(f"OpenAI API returned an API Error: {e}")

    except openai.error.AuthenticationError as e:
        # Handle Authentication error here, e.g. invalid API key
        print(f"OpenAI API returned an Authentication Error: {e}")

    except openai.error.APIConnectionError as e:
        # Handle connection error here
        print(f"Failed to connect to OpenAI API: {e}")

    except openai.error.InvalidRequestError as e:
        # Handle connection error here
        print(f"Invalid Request Error: {e}")

    except openai.error.RateLimitError as e:
        # Handle rate limit error
        print(f"OpenAI API request exceeded rate limit: {e}")

    except openai.error.ServiceUnavailableError as e:
        # Handle Service Unavailable error
        print(f"Service Unavailable: {e}")

    except openai.error.Timeout as e:
        # Handle request timeout
        print(f"Request timed out: {e}")

import pandas as pd
#import numpy as np
import time

path = r"dados.csv"
df = pd.read_csv(path, sep= "|")
df.drop(["Unnamed: 0"], axis=1, inplace = True)
df.fillna("", inplace = True)
#df["gpt"] = np.nan
#df = df.head()

lista = []
for indice, linha in df.iterrows():
    print(indice, end = "|")
#    print(texto)
    texto = linha[2]
    embedding = get_embedding(text = texto,
                              engine=config['OPENAI']['EMBEDDINGS_MODEL'],
                              headers=headers)
    # Se der erro por muitas requisições
    if embedding == None:
        time.sleep(5)
        embedding = get_embedding(text = texto,
                                  engine=config['OPENAI']['EMBEDDINGS_MODEL'],
                                  headers=headers)        
    om = linha[0]
    embedding.append(om)
    lista.append(embedding)

df_lista = pd.DataFrame(lista)
# Seta o número da om como índice da tabela de embeddings
df_lista.set_index(1536, inplace = True)
# Seta o número da om como índice da tabela de embeddings
df.set_index("NumOrdem", inplace = True)
#df.at[indice, "gpt"] = embedding
df_join = df.join(df_lista)
df_join.to_excel("gpt.xlsx")    
#df["gpt"] = lista