import cohere
import weaviate
from django.conf import settings
from dotenv import load_dotenv
from weaviate.classes.init import Auth
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
            

            weaviate_api_key = "hsdnfd7y3n87ry28gd989m82372t1e8hsey78t3291de"

            self.weaviate_client = weaviate.connect_to_custom(
                http_host="173.208.218.180",
                http_port=8080,
                http_secure=False,
                grpc_host="173.208.218.180",
                grpc_port=50051,
                grpc_secure=False,
                headers={
                    "X-API-KEY": "chay@gmail.com"
                }
            )
            if self.weaviate_client.is_ready():
                print("Weaviate connection established successfully.")
            else:
                print("Weaviate is not ready.")
        except Exception as e:
            print("Error connecting to Weaviate:", e)
                
