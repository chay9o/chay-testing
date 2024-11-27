import cohere
import weaviate
from django.conf import settings
from dotenv import load_dotenv
import base64
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

load_dotenv()

class ClientCredentials:

    def __init__(self):
        try:
            self.cohere_client = cohere.Client(settings.COHERE_API_KEY)
            self.__auth_config = weaviate.auth.AuthApiKey(api_key=settings.WEAVIATE_API_KEY)
            self.weaviate_client = weaviate.connect_to_local(
                host="173.208.218.180",
                port=8080,
                grpc_port=50051
                #additional_headers={"X-API-KEY": "hsdnfd7y3n87ry28gd989m82372t1e8hsey78t3291de"}
            )
            if self.weaviate_client.is_ready():
                print("Weaviate connection established successfully.")
            else:
                print("Weaviate is not ready.")
        except Exception as e:
            print("Error connecting to Weaviate:", e)
                
