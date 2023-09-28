from dotenv import load_dotenv
from conversationService import get_embedding
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import loggingService
import os
import weaviate

load_dotenv()
logger = loggingService.get_logger()

apikey = os.getenv("GEANAI_KEY", None)
class_name = os.getenv("WEVIATE_CLASS", 'LivrosVectorizer')
path = os.getenv("DATA_PATH", "data")
weaviate_url = os.getenv("WEAVIATE_URL", 'http://127.0.0.1:8080')

client = weaviate.Client(
    url=weaviate_url,
)

pages = []
pdf_loader = PyPDFDirectoryLoader(path)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50, separators=["\n\n", "\n", " ", "", "\n \n", "\n \n \n","\n\n\n"])
documents = pdf_loader.load_and_split(text_splitter=text_splitter)
print(len(documents))
print(documents[0])

def pdf_text_splitter(pdf_text) -> dict:
  retorno = {'content': '', 'source': '', 'page': 0}
  
  retorno['content'] = getattr(pdf_text, 'page_content')
  retorno['souce'] = getattr(pdf_text, 'metadata')['source']
  retorno['page'] = getattr(pdf_text, 'metadata')['page']
  
  return retorno

for doc in documents:
  # logger.debug(pdf_text_splitter(doc))
  # print(pdf_text_splitter(doc))
  pages.append(pdf_text_splitter(doc))
  
print(pages[0])

i = 0
client.batch.configure(batch_size=10)  # Configure batch
with client.batch as batch:
    
  for page in pages:
    # logger.info(f"importing question: {i+1}")
    print(f"importing question: {i+1}")
    i = i+1
    
    properties = {
      "content": page["content"],
      "page": str(page["page"]),
      "source": page["source"],
    }
    
    batch.add_data_object(properties, 'LivrosVectorizer',)

result = (client.query
  .get('LivrosVectorizer', ["content", "source", "page"])
  .with_additional(["certainty", "distance"]) # note that certainty is only supported if distance==cosine
  .with_near_text('o que significa casmurro')
  .with_limit(4)
  .do())

print(result['data']['Get'][class_name])

retorno = ''
if len(result['data']['Get'][class_name]) > 0:
  retorno = result['data']['Get'][class_name][0]['content']

  for contexto in result['data']['Get'][class_name][1:]:
    retorno += f"\n{contexto['content']}"

logger.info(retorno)