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
            # OIDC token (replace `<your_oidc_token>` with the actual access token you received)
            oidc_token = "eyJhbGciOiJIUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJlY2Y3M2Y1Zi0yNDVhLTQyYjgtYWJkNy1kNDNjYTdiMGJhMmQifQ.eyJleHAiOjE3MzI4MzM0ODAsImlhdCI6MTczMjgzMTY4MCwianRpIjoiNDU1YjQ3ZjktZjMxZi00Yzk0LTk3MDAtYjRkOTNiNDhmYzMwIiwiaXNzIjoiaHR0cHM6Ly9hdXRoLndjcy5hcGkud2VhdmlhdGUuaW8vYXV0aC9yZWFsbXMvU2VNSSIsImF1ZCI6Imh0dHBzOi8vYXV0aC53Y3MuYXBpLndlYXZpYXRlLmlvL2F1dGgvcmVhbG1zL1NlTUkiLCJzdWIiOiI2ZmFlMzYxYy0xYjdjLTQxMjctYjYyMi0zYTJhMzYyMGNkOTciLCJ0eXAiOiJSZWZyZXNoIiwiYXpwIjoid2NzIiwic2Vzc2lvbl9zdGF0ZSI6Ijg0NjUyMTA4LWNjYjctNGZmYy1hNGEwLWFjNGQyMDQ2Mzk4OCIsInNjb3BlIjoicHJvZmlsZSBvcGVuaWQgZW1haWwiLCJzaWQiOiI4NDY1MjEwOC1jY2I3LTRmZmMtYTRhMC1hYzRkMjA0NjM5ODgifQ.XWO5cQU8zkBa3PLu0O-2OArcmTXFgJ6r0kWmYHSgFvY"  # Add your full token here

            self.weaviate_client = weaviate.connect_to_custom(
                http_host="w4.strategicfuture.ai",
                http_port="8082", # Placeholder value; won't be actively used due to HTTPS
                http_secure=True,  # Use HTTPS for secure connection
                grpc_host="w4.strategicfuture.ai",
                grpc_port=50051,
                grpc_secure=False,  # If gRPC is not configured for HTTPS, leave it False
                headers={
                    "Authorization": f"Bearer {oidc_token}"  # Pass the OIDC token in the Authorization header
                }
            )
            if self.weaviate_client.is_ready():
                print("Weaviate connection established successfully.")
            else:
                print("Weaviate is not ready.")
        except Exception as e:
            print("Error connecting to Weaviate:", e)
                
