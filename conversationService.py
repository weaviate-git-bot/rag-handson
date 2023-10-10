import unicodedata
from dotenv import load_dotenv
from genai import PromptPattern
import loggingService
from genai.model import Credentials, Model
from genai.schemas import GenerateParams
import os
from sentence_transformers import SentenceTransformer
import weaviate

load_dotenv()
logger = loggingService.get_logger()

class_name = os.getenv("WEVIATE_CLASS", 'Livros')
path = os.getenv("DATA_PATH", "data")

weaviate_url = os.getenv("WEAVIATE_URL", 'http://127.0.0.1:8080')
client = weaviate.Client(
    url=weaviate_url,
)

api_key = os.getenv("GENAI_KEY", None)
api_endpoint = os.getenv("GENAI_API", 'https://workbench-api.res.ibm.com')
model_name = os.getenv('MODEL_NAME', 'bigscience/mt0-xxl')
creds = Credentials(api_key, api_endpoint=api_endpoint)
params = GenerateParams(
    decoding_method="greedy",
    max_new_tokens=100,
    min_new_tokens=1,
    stream=False,
    temperature=0.8,
    top_k=50,
    # top_p=1,
)
model = Model(model=model_name, credentials=creds, params=params)

# depois extrair para arquivo
pt1 = """Responda a pergunta a seguir de forma sucinta usando o contexto fornecido. Caso não tenha certeza da resposta siceramente diga que não possui informações suficientes sobre esse tema.

{{context}}

Pergunta: {{question}}
Resposta:"""

prompt = PromptPattern.from_str(pt1)

def get_context(query: str, certainty= 0.8, limit = 4) -> str:
  """_summary_

  Args:
      query (str): _description_
      certainty (float, optional): _description_. Defaults to 0.8.
      limit (int, optional): _description_. Defaults to 4.
  """
  
  result = (client.query
  .get('Livros', ["content", "source", "page"])
  .with_additional(["certainty", "distance"]) # note that certainty is only supported if distance==cosine
  .with_near_text({'concepts': query, 'certainty': 0.6})
  .with_limit(limit)
  .do()
  )
  
  # print(result)
  
  retorno = ''
  class_name = 'Livros'
  
  if len(result['data']['Get'][class_name]) == 0:
    return retorno
  retorno = result['data']['Get'][class_name][0]['content']
  
  for contexto in result['data']['Get'][class_name][1:]:
    retorno += unicodedata.normalize("NFKD", f"\n{contexto['content']}")
  
  return retorno

def get_llm_response(question: str, prompt = prompt) -> str:
  resposta = ''
  
  contexto = get_context(question)
  prompt.sub('context', contexto).sub('question', question)
  
  logger.info(prompt)
  logger.info('-' * 39)

  respostas =model.generate([prompt])
  retorno = ''
  
  for resposta in respostas:
    retorno += str(resposta.generated_text)
  
  return retorno
  

if __name__ == '__main__':
  print(get_llm_response('por que arthur dent deitou na lama?'))
  # print(get_context('por que arthur dent deitou na lama?'))

  # print(client.query
  #   .aggregate("Livros")
  #   .with_fields("meta { count }")
  #   .do())
