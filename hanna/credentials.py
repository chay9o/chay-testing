import cohere
import weaviate
from weaviate.classes.init import Auth
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
        # self.__auth_config = weaviate.auth.AuthApiKey(api_key=settings.WEAVIATE_API_KEY)
        # self.weaviate_client = weaviate.Client(
        #     url=settings.WEAVIATE_URL,
        #     additional_headers={"X-Cohere-Api-Key": settings.COHERE_API_KEY},
        #     auth_client_secret=self.__auth_config
        # )

        self.__auth_config = Auth.api_key(settings.WEAVIATE_API_KEY)
        self.weaviate_client = weaviate.connect_to_wcs(
            skip_init_checks=True,
            cluster_url=settings.WEAVIATE_URL,
            auth_credentials=self.__auth_config,
            headers={
                "X-Cohere-Api-Key": settings.COHERE_API_KEY
            }
        )

