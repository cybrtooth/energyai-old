import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader, TextLoader

model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# create function for vector storing into persist db using loaders
def create_or_add_to_vector_store(loader):
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    vector_store = Chroma.from_documents(texts, embeddings, collection_metadata={"hnsw:space": "cosine"}, persist_directory="stores/energy_cosine")
    print(f'Vector store created using loader: {loader} \n\n')

# for root, dirs, files in os.walk('energy_docs'):
#     for file in files:
#         if file.endswith('pdf'):
#             loader = PDFMinerLoader(os.path.join(root,file))

# more efficient way 
# create loaders
loader_list = []
energy_pdf_docs_loader = DirectoryLoader('energy_docs', glob='**/*.pdf', show_progress=True, loader_cls=PyPDFLoader)
loader_list.append(energy_pdf_docs_loader)
# text_crawl_loader = DirectoryLoader(
#     'energyeducationcrawl/searchoutput/',
#     glob='**/*.txt', 
#     show_progress=True,
#     loader_cls=TextLoader
# )
#loader_list.append(text_crawl_loader)

# ee_docs = text_crawl_loader.load()
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# print('splitting docs... \n\n')
# texts = text_splitter.split_documents(ee_docs)

# vector_store = Chroma.from_documents(texts, embeddings, collection_metadata={"hnsw:space": "cosine"}, persist_directory="stores/energy_cosine")

print('created ee docs vector store')
# loop through loaders and create or add to vector store from them 
for loader in loader_list:
    create_or_add_to_vector_store(loader)


# loader = PyPDFLoader("pet.pdf")
print("Finished creating vector stores.......")

