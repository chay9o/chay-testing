import requests


class JINA:
    def __init__(self, api_key: str, model: str = "jina-reranker-v2-base-multilingual"):
        self.__header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        self.__model = model

    def rerank(self, query, docs, top_n: int = 10):
        url = "https://api.jina.ai/v1/rerank"

        data_obj = {
            "model": self.__model,
            "query": query,
            "documents": docs,
            "top_n": top_n
        }

        response = requests.post(url, headers=self.__header, json=data_obj)

        return response.json()

    def embed(self, batch: list):
        try:
            url = 'https://api.jina.ai/v1/embeddings'

            data_obj = {
                'input': batch,
                'model': 'jina-embeddings-v2-base-es'
            }

            response = requests.post(url, headers=self.__header, json=data_obj)

            return response.json()
        except Exception as e:
            print(e)
            return "Something went wrong"




