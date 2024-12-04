import weaviate
import warnings
from requests.auth import HTTPBasicAuth
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

            # Connect to Weaviate
            self.weaviate_client = weaviate.connect_to_custom(
                http_host="w4.strategicfuture.ai",  # Replace with your actual domain
                http_port="8082",  # HTTP port
                http_secure=True,  # Use HTTPS for secure connection
                grpc_host="w4.strategicfuture.ai",  # Replace with your actual domain
                grpc_port=50051,  # GRPC port
                grpc_secure=False,  # Set True if GRPC uses HTTPS
                headers={
                    "Authorization": authorization_header  # Use Basic Auth
                },
            )

            # Test connection
            if self.weaviate_client.is_ready():
                print("Weaviate connection established successfully.")
            else:
                print("Weaviate is not ready.")
        except Exception as e:
            print(f"Error connecting to Weaviate: {e}")
