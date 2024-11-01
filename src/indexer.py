import pandas as pd
import torch
import time
from transformers import AutoTokenizer, AutoModel
from langchain_core.documents import Document
from langchain_elasticsearch import ElasticsearchStore


class CustomEmbeddingModel:

    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed_text(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

    def embed_query(self, query):
        return self.embed_text(query)

    def embed_documents(self, documents):
        return [self.embed_text(doc) for doc in documents]


class Indexer:

    def __init__(self, data_path="data/data.xlsx", embedding_model=None):
        self.data = pd.read_excel(data_path)
        self.embedding_model = embedding_model if embedding_model else CustomEmbeddingModel()

    def perform_indexing(self):
        self.preprocess_data()
        self.prepare_docs_for_indexing()
        self.index_docs()

    def preprocess_data(self):
        self.data['weather_climate_desc'] = self.data.apply(
            lambda row: f"Weather is {row['weather']} and Climate is {row['climate']}", axis=1)

    def prepare_docs_for_indexing(self):
        self.docs = []
        for i, row in self.data.iterrows():
            doc = Document(page_content=row['weather_climate_desc'],
                           metadata={'city': row['city'], 'temperature': row['temperature'],
                                     'weather': row['weather'], 'climate': row['climate']})
            self.docs.append(doc)

    def index_docs(self, index_name='weather_rag'):
        ElasticsearchStore.from_documents(self.docs, self.embedding_model,
                                          index_name=index_name, es_url="http://localhost:9200")


if __name__ == "__main__":
    t1 = time.time()
    indexer = Indexer()
    indexer.perform_indexing()
    t2 = time.time()
    print(f"Time taken for indexing: {round(t2-t1, 3)} sec")
