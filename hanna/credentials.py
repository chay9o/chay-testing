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
                http_host="w4.strategicfuture.ai",
                http_port=, 
                http_secure=True,  # Use HTTPS for secure connection
                grpc_host="w4.strategicfuture.ai",
                grpc_port=50051,
                grpc_secure=False,  # If gRPC is not configured for HTTPS, leave it False
                headers={
                    "X-API-KEY": "jane@doe.com"
                }
            )
            if self.weaviate_client.is_ready():
                print("Weaviate connection established successfully.")
            else:
                print("Weaviate is not ready.")
        except Exception as e:
            print("Error connecting to Weaviate:", e)
                
