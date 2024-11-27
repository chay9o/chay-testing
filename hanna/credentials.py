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
        self.cohere_client = cohere.Client(settings.COHERE_API_KEY)
        self.__auth_config = weaviate.auth.AuthApiKey(api_key=settings.WEAVIATE_API_KEY)
        self.weaviate_client = weaviate.connect_to_local(
            host=settings.WEAVIATE_HOST,
            port=settings.WEAVIATE_PORT,
            grpc_port=settings.WEAVIATE_GRPC_PORT,
            additional_headers={"X-API-KEY": settings.WEAVIATE_API_KEY}
        )
