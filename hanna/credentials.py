import cohere
import weaviate
from django.conf import settings
from dotenv import load_dotenv
import warnings
import time
import requests

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

load_dotenv()

class OIDCAuthManager:
    def __init__(self, client_id, client_secret, token_url):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url
        self.access_token = None
        self.token_expiry = 0

    def fetch_tokens(self):
        response = requests.post(
            self.token_url,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
        )
        if response.status_code == 200:
            tokens = response.json()
            self.access_token = tokens["access_token"]
            self.refresh_token = tokens.get("refresh_token")
            self.token_expiry = time.time() + tokens["expires_in"] - 60  # Refresh 1 min before expiry
            print("Access token fetched successfully.")
        else:
            raise Exception(f"Failed to fetch tokens: {response.json()}")

    def get_access_token(self):
        if time.time() > self.token_expiry:
            if self.refresh_token:
                self.refresh_access_token()
            else:
                self.fetch_tokens()
        return self.access_token

class ClientCredentials:
    def __init__(self):
        try:
            # Cohere Client
            self.cohere_client = cohere.Client(settings.COHERE_API_KEY)

            # OIDC Authentication Manager
            self.oidc_manager = OIDCAuthManager(
                client_id="C42VtjQDJG1FXx8CpnwnIhYA8h2bGZcL",  # Replace with your actual client ID
                client_secret="U3vInO_R1zKTHXIS59LQ0nwogoqBWFTlHOXPJ-hTSTUKi2fQvu-To2TZBcUFhaqs", 
                token_url="https://dev-tqyuws3babyfc0v5.us.auth0.com/oauth/token",  # Updated token URL
            )
            self.oidc_manager.fetch_tokens()

            # Weaviate Client
            self.weaviate_client = weaviate.connect_to_custom(
                http_host="w4.strategicfuture.ai",  # Replace with your actual domain
                http_port="8080",  # HTTP port
                http_secure=True,  # Use HTTPS for secure connection
                grpc_host="w4.strategicfuture.ai",  # Replace with your actual domain
                grpc_port=50051,  # GRPC port
                grpc_secure=False,  # Set True if GRPC uses HTTPS
                headers={
                    "Authorization": f"Bearer {self.oidc_manager.get_access_token()}"
                }
            )

            # Check Weaviate Connection
            if self.weaviate_client.is_ready():
                print("Weaviate connection established successfully.")
            else:
                print("Weaviate is not ready.")
        except Exception as e:
            print("Error connecting to Weaviate:", e)
