# AutoSAO
Automatização de análise de textos longos de ordens de manutenção SAP com uso de diversas técnicas de embedding diferentes.

Neste repositório estão contidos códigos associados ao [artigo](https://github.com/ThomasAndarilho/AutoSAO/blob/main/Artigo.pdf), que contém a comparação do desempenho de diversas técnicas de embedding na tarefa de classificação automática de textos.
A seguir serão apresentados os scripts de processamento de texto utilizados para gerar os dados que servirão de base para as simulações contidas no [notebook principal](https://github.com/ThomasAndarilho/AutoSAO/blob/main/main.ipynb)

Este projeto visa automatizar a classificação de textos com treinamento supervisionado de modelos de inteligência artificial. Existem diversas técnicas consolidadas de classificação de dados numéricos e essas técnicas podem ser utilizadas na classificação uma vez que estes sejam convertidos para vetores de números. A tarefa de conversão de textos para vetores não é trivial e existem diversas metodologias para fazer isso desde a contagem de palavras até técnicas baseadas na arquitetura Transformer, como um serviço de embedding criado pela OpenAI, que também foi utilizado no presente trabalho. No repositório estão contidos os códigos utilizados para gerar o embedding de textos de ordens de manutenção para sua classificação.

## Índice

- [TF-IDF](#tf-idf)
- [GPT](#gpt)
- [BERTimbau](#bertimbau)

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

### Rascunho
