import cohere
import weaviate
from django.conf import settings
from dotenv import load_dotenv
from weaviate.classes.init import Auth
from weaviate import Client
from weaviate.auth import AuthClientPassword 
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
            auth_client = AuthClientPassword(
                username="chay.kusumanchi@strategicfuture.ai",  # Replace with your WCS email
                password="Chaitanya@2244",         # Replace with your WCS password
                openid_url="https://auth.wcs.api.weaviate.io/auth/realms/SeMI",  # OIDC URL
            )

            self.weaviate_client = weaviate.connect_to_weaviate_cloud(
                cluster_url=weaviate_url,   # Replace with your Weaviate Cloud URL
                auth_credentials=Auth.client_password(
                    username="chay.kusumanchi@strategicfuture.ai",  # Your Weaviate Cloud username
                    password="Chaitanya@2244"  # Your Weaviate Cloud password
                )
            )
            if self.weaviate_client.is_ready():
                print("Weaviate connection established successfully.")
            else:
                print("Weaviate is not ready.")
        except Exception as e:
            print("Error connecting to Weaviate:", e)
                
