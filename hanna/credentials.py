import cohere
import weaviate
from django.conf import settings
from dotenv import load_dotenv
from weaviate.classes.init import Auth
from weaviate import Client, AuthClientPassword
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
            auth = AuthClientPassword(
                username="<your-username>",
                password="<your-password>",
                client_id="wcs",
                openid_url="https://auth.wcs.api.weaviate.io/auth/realms/SeMI",
            )

            self.weaviate_client = Client(
                url="https://w4.strategicfuture.ai",  # Your Weaviate endpoint
                auth_client_secret=auth,
                additional_headers={
                    "X-OpenID-User": "chay.kusumanchi@strategicfuture.ai"  # Optional: Include extra headers if required
                }
            )
            if self.weaviate_client.is_ready():
                print("Weaviate connection established successfully.")
            else:
                print("Weaviate is not ready.")
        except Exception as e:
            print("Error connecting to Weaviate:", e)
                
