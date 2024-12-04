import weaviate
import cohere
import warnings
from requests.auth import HTTPBasicAuth
from weaviate.classes.init import Auth
from django.conf import settings
import base64

warnings.filterwarnings("ignore")


class BasicAuthManager:
    def __init__(self, username, password):
        """
        Initialize Basic Authentication details.
        :param username: Username for authentication.
        :param password: Password for authentication.
        """
        self.username = username
        self.password = password

    def get_authorization_header(self):
        """
        Generate the Authorization header for Basic Authentication.
        :return: Authorization header value as a string.
        """
        auth_string = f"{self.username}:{self.password}"
        auth_encoded = base64.b64encode(auth_string.encode()).decode()
        return f"Basic {auth_encoded}"


class ClientCredentials:
    def __init__(self):
        """
        Initialize the client and connect to Weaviate using Basic Authentication.
        """
        try:
            # Basic Authentication Configuration
            auth_manager = BasicAuthManager(
                username="weaviate1",  # Replace with your username
                password="AdminMobika11$$W"  # Replace with your password
            )

            # Generate Authorization header
            authorization_header = auth_manager.get_authorization_header()
            self.cohere_client = cohere.Client(settings.COHERE_API_KEY)

            # Connect to Weaviate
            self.weaviate_client = weaviate.connect_to_weaviate_cloud(
                cluster_url="https://fxyipvexrfmhljjhohkuhw.c0.us-west3.gcp.weaviate.cloud",
                auth_credentials=Auth.api_key("2qQA9k9vsrjugU1zH2mJ4aMpxJC6ujKSobRK"),
                headers={
                    "X-Cohere-Api-Key": settings.COHERE_API_KEY
                }
            )

            # Test connection
            if self.weaviate_client.is_ready():
                print("Weaviate connection established successfully.")
            else:
                print("Weaviate is not ready.")
        except Exception as e:
            print(f"Error connecting to Weaviate: {e}")
