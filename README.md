# AutoSAO
Automatização de análise de textos longos de ordens de manutenção SAP com uso de diversas técnicas de embedding diferentes.

Neste repositório estão contidos códigos associados ao [artigo](https://github.com/ThomasAndarilho/AutoSAO/blob/main/Artigo.pdf), que contém a comparação do desempenho de diversas técnicas de embedding na tarefa de classificação automática de textos.
A seguir serão apresentados os scripts de processamento de texto utilizados para gerar os dados que servirão de base para as simulações contidas no [notebook principal](https://github.com/ThomasAndarilho/AutoSAO/blob/main/main.ipynb)

Este projeto visa automatizar a classificação de textos com treinamento supervisionado de modelos de inteligência artificial. Existem diversas técnicas consolidadas de classificação de dados numéricos e essas técnicas podem ser utilizadas na classificação uma vez que estes sejam convertidos para vetores de números. A tarefa de conversão de textos para vetores não é trivial e existem diversas metodologias para fazer isso desde a contagem de palavras até técnicas baseadas na arquitetura Transformer, como um serviço de embedding criado pela OpenAI, que também foi utilizado no presente trabalho. No repositório estão contidos os códigos utilizados para gerar o embedding de textos de ordens de manutenção para sua classificação.

## Índice

- [TF-IDF](#tf-idf)
- [GPT](#gpt)
- [BERTimbau](#bertimbau)
- [SBERT](#sbert)
- [Petrolês](#petroles)
- [Simulações](#main)

### TF-IDF

Para realizar o cálculo da matriz TF-IDF, foi utilizado o script [TF-IDF.py](https://github.com/ThomasAndarilho/AutoSAO/blob/main/TF-IDF.py).
Esse script utiliza como entrada o arquivo dados.csv, em particular a coluna de texto longo e realiza diversas técnicas de NLP, a saber: eliminação de nulos, tokenização, remoção de *stopwords*, *stemming*. Ao final é utilizada a classe [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) da biblioteca sklearn para criar a matriz a partir dos dados tratados.
Para utilizar o script, é necessário apontar o caminho do arquivo de dados na linha 12.
O script gera como saída os seguintes arquivos com conjuntos de treino e teste: TF-IDF_X_train.xlsx, TF-IDF_X_test.xlsx, TF-IDF_y_train.xlsx e TF-IDF_y_test.xlsx.

### GPT

As ferramentas de NLP da OpenAI que utilizam a arquitetura Transformers representam um grande avanço na área de processamento de linguagem natural, no entanto o uso de informações corporativas dentro dessa ferramenta pode gerar vazamento de dados (como o relatado na [Forbes](https://www.forbes.com/sites/siladityaray/2023/05/02/samsung-bans-chatgpt-and-other-chatbots-for-employees-after-sensitive-code-leak/?sh=643fc8526078)) uma vez que o conteúdo das mensagens pode ser utilizado para o treinamento de versões futuras do modelo.
Com objetivo de evitar esse risco, a PETROBRAS, em parceria com a Microsoft, criou um serviço interno com tecnologia OpenAI chamado [ChatPetrobras](https://www.linkedin.com/posts/fernando-castelloes-436878107_chatpetrobras-ia-generativa-acess%C3%ADvel-a-activity-7113709745651298304-lWug?trk=public_profile_like_view). Dentro desses serviços foram disponibilizadas diversas APIs, inclusive a de embedding mais avançada da OpenAI, o [text-embedding-ada-002](https://openai.com/blog/new-and-improved-embedding-model).
Para realizar esse embedding foi utilizado o script [GPT.py](https://github.com/ThomasAndarilho/AutoSAO/blob/main/GPT.py).
Para utilizar o script, é necessário estar conectado à intranet PETROBRAS e são necessários os arquivos "config.ini" e "petrobras-ca-root.pem", que foram omitidos do presente repositório por conter informações sensíveis. Também é necessário apontar o caminho do arquivo de dados na linha 86.
O script gera como saída o arquivo gpt.xlsx.

### BERTimbau

O BERT (*Bidirectional Encoder Representations from Transformers*), da Google é um exemplo de engenho baseado na arquiterura Transformers. Esse modelo foi treinado e disponibilizado pela Google, no entanto essa versão disponibilizada foi treinada no idioma inglês. Para permitir aplicações no inioma Português brasileiro, a empresa [NeuralMind.ai](https://neuralmind.ai/bert/) fez um *fine tunning* do BERT. Esse modelo foi carinhosamente apelidado de BERTimbau, e disponibilizado sob o código "bert-base-portuguese-cased" através da biblioteca "transformers" do [Hugging Face](https://huggingface.co/neuralmind/bert-base-portuguese-cased).
Foram gerados embeddings com esse modelo através do script [BERTimbau.py](https://github.com/ThomasAndarilho/AutoSAO/blob/main/BERTimbau.py).
Para utilizar o script, é necessário apontar o caminho do arquivo de dados na linha 19. Um fato relevante sobre esse modelo, é que ele gera um embedding por token. Para obter o embedding da sentença foi tirada a média dos vetores de embedding através da função torch.mean() da biblioteca PyTorch
O script gera como saída o arquivo BERT.xlsx.

### SBERT

Também foi utilizada uma versão do BERT modificada para o embedding de sentenças, o [SBERT](https://www.sbert.net/). Dentro da biblioteca sentence_transformers foram utilizados dois modelos poliglotas diferentes aplicados nos scripts [SBERT_v1.py](https://github.com/ThomasAndarilho/AutoSAO/blob/main/SBERT_v1.py) e [SBERT_v2.py](https://github.com/ThomasAndarilho/AutoSAO/blob/main/SBERT_v2.py).
Para utilizar os scripts, é necessário apontar o caminho do arquivo de dados na linha 11.
Os scripts geras como saídas os arquivos sbert_v1.xlsx e sbert_v2.xlsx.

### Petrolês

O Petrolês é uma iniciativa de colaboração interinstitucional liderada pelo Centro de Pesquisas e Desenvolvimento da Petrobras (CENPES), em parceria com PUC-Rio, UFRGS e PUC-RS, e visa incentivar pesquisas nas áreas de Processamento de Linguagem Natural e Linguística Computacional aplicadas ao domínio de O&G. Em sua [página pública](https://petroles.puc-rio.ai/) são disponibilizados diversos modelos treinados a partir dos corpora especializados nos domínios de Óleo e Gás (O&G). Foram utilizados três modelos do tipo Word2vec em três scripts distintos, a saber:
- O script [Petroles_H_100.py](https://github.com/ThomasAndarilho/AutoSAO/blob/main/Petroles_H_100.py) utiliza o modelo [Petrovec-híbrido (Word2vec) 100](https://petroles.puc-rio.ai/files/embeddings/Petrovec_Hibrido_Word2vec.zip) e gera como saída o arquivo petroles.xlsx.
- O script [Petroles_O&G_100.py](https://github.com/ThomasAndarilho/AutoSAO/blob/main/Petroles_O%26G_100.py) utiliza o modelo [Petrovec-O&G (Word2vec) 100](https://petroles.puc-rio.ai/files/embeddings/Petrovec_OeG_Word2vec.zip) e gera como saída o arquivo petroles_o&g_100.xlsx.
- O script [Petroles_O&G_300.py](https://github.com/ThomasAndarilho/AutoSAO/blob/main/Petroles_O%26G_300.py) utiliza o modelo [Petrovec-O&G (Word2vec) 300](https://petroles.puc-rio.ai/files/embeddings/PetroVEC_OeG_Word2vec_300d.zip) e gera como saída o arquivo petroles_o&g_300.xlsx.
Para utilizar cada um dos scripts, é necessário baixar e descompactar os modelos disponíveis na [página pública](https://petroles.puc-rio.ai/) e apontar o diretório onde os arquivos descompactados estão na linha 30 dos scripts. Também é necessário apontar caminho do arquivo de dados na linha 48.

### Simulações

Os dados gerados pelo processamento de textos com os scripts supracitados são utilizados para treinar três tipos diferentes de modelos: Random Forest (RF), Redes Neurais (do inglês "Neural Networks" - NN), e XGBoost.
Os códigos utilizados para essas simulações estão contidos no notebook [main.ipynb](https://github.com/ThomasAndarilho/AutoSAO/blob/main/main.ipynb).
Para executar com sucesso o código contido nesse notebook, é necessário que ele esteja na mesma pasta que os arquivos .xlsx gerados por cada um dos scripts supracitados.
As interpretações, conclusões e referências estão contidas no arquivo [Artigo.pdf](https://github.com/ThomasAndarilho/AutoSAO/blob/main/Artigo.pdf)

Resultados

| Image Name           | Remission Map                               |
| :------------------: | :-----------------------------------------: |
| i7705600_-338380.png | ![gpt](https://github.com/ThomasAndarilho/AutoSAO/assets/65094666/07e09a08-b1fd-49e0-980b-174e5655e081) |
| i7726110_-353570.png | ![Remission Map](data/i7726110_-353570.png) |



| Image Name           | Remission Map                               | Ground Truth                                   | Road Map Probabilities                     | Road Map Classes                               |
| :------------------: | :-----------------------------------------: | :--------------------------------------------: | :----------------------------------------: | :--------------------------------------------: |
| i7705600_-338380.png | ![Remission Map](data/i7705600_-338380.png) | ![Ground Truth](data/i7705600_-338380_svg.png) | ![Road Map](data/r7705600_-338380_map.png) | ![Road Map](data/r7705600_-338380_map_1_6.png) |
| i7726110_-353570.png | ![Remission Map](data/i7726110_-353570.png) | ![Ground Truth](data/i7726110_-353570_svg.png) | ![Road Map](data/r7726110_-353570_map.png) | ![Road Map](data/r7726110_-353570_map_1_6.png) |

