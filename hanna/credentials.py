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
            oidc_token = "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJUOU1HUG1WOTZna1ljbFhqQjl1ZG52ZktKZlJlU0lpSTRrS3FjQi1SWTlJIn0.eyJleHAiOjE3MzI4MzI1ODAsImlhdCI6MTczMjgzMTY4MCwianRpIjoiNDgyMTRmYzItYjc0NC00YzQzLTg2NDYtNDRiODZlOGQ0MWI2IiwiaXNzIjoiaHR0cHM6Ly9hdXRoLndjcy5hcGkud2VhdmlhdGUuaW8vYXV0aC9yZWFsbXMvU2VNSSIsImF1ZCI6WyJ3Y3MiLCJhY2NvdW50Il0sInN1YiI6IjZmYWUzNjFjLTFiN2MtNDEyNy1iNjIyLTNhMmEzNjIwY2Q5NyIsInR5cCI6IkJlYXJlciIsImF6cCI6IndjcyIsInNlc3Npb25fc3RhdGUiOiI4NDY1MjEwOC1jY2I3LTRmZmMtYTRhMC1hYzRkMjA0NjM5ODgiLCJhbGxvd2VkLW9yaWdpbnMiOlsiaHR0cDovL2NvbnNvbGUuc2VtaS50ZWNobm9sb2d5IiwiaHR0cHM6Ly9jb25zb2xlLndlYXZpYXRlLmlvLyoiLCJodHRwczovL2NvbnNvbGUud2VhdmlhdGUuaW8iLCJodHRwOi8vY29uc29sZS5zZW1pLnRlY2hub2xvZ3kvKiIsImh0dHBzOi8vYWNjZW50dXJlMDAxLmRlbW8uc2VtaS50ZWNobm9sb2d5IiwiaHR0cDovL3BsYXlncm91bmQuc2VtaS50ZWNobm9sb2d5LyoiLCJodHRwczovL2NvbnNvbGUuc2VtaS50ZWNobm9sb2d5Il0sInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJkZWZhdWx0LXJvbGVzLXNlbWkiLCJvZmZsaW5lX2FjY2VzcyIsInVtYV9hdXRob3JpemF0aW9uIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwic2NvcGUiOiJwcm9maWxlIG9wZW5pZCBlbWFpbCIsInNpZCI6Ijg0NjUyMTA4LWNjYjctNGZmYy1hNGEwLWFjNGQyMDQ2Mzk4OCIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJuYW1lIjoibi5hLiBuLmEuIiwicHJlZmVycmVkX3VzZXJuYW1lIjoiY2hheS5rdXN1bWFuY2hpQHN0cmF0ZWdpY2Z1dHVyZS5haSIsImdpdmVuX25hbWUiOiJuLmEuIiwiZmFtaWx5X25hbWUiOiJuLmEuIiwiZW1haWwiOiJjaGF5Lmt1c3VtYW5jaGlAc3RyYXRlZ2ljZnV0dXJlLmFpIn0.kNhlHWj2o5yTy5stmfEwoXAY-ER00ThSwgKZ1nBiUA_IvzAunDCsvNaiV7En8Pe4-5zbVUwr2cuQYzrTOYdZKLoq0F3z5Dv-z2T2KxWY8bZQd99VVd6nOo6XqpPDhq8QXzAKh5C_Z1EZOik1JOPvJZQaUBHH1aenWXv_H_UqPNLQp78qSyjpJB7ExS_tHxoVZhgG3MPeKB_8nTys8KNpXxw3ID4P0rAXf1fTtdJ_kXVxRdKGWHEE5Z462wyPEjOAoJN-5ccKyqAOqc1xsZIGHag88iVUKY_GQqUrfAHD0WYIP-klG8hnryrZw8W2IEa-6pvHZs8HloZWvLueH2QcyA","expires_in":900,"refresh_expires_in":1800,"refresh_token":"eyJhbGciOiJIUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJlY2Y3M2Y1Zi0yNDVhLTQyYjgtYWJkNy1kNDNjYTdiMGJhMmQifQ.eyJleHAiOjE3MzI4MzM0ODAsImlhdCI6MTczMjgzMTY4MCwianRpIjoiNDU1YjQ3ZjktZjMxZi00Yzk0LTk3MDAtYjRkOTNiNDhmYzMwIiwiaXNzIjoiaHR0cHM6Ly9hdXRoLndjcy5hcGkud2VhdmlhdGUuaW8vYXV0aC9yZWFsbXMvU2VNSSIsImF1ZCI6Imh0dHBzOi8vYXV0aC53Y3MuYXBpLndlYXZpYXRlLmlvL2F1dGgvcmVhbG1zL1NlTUkiLCJzdWIiOiI2ZmFlMzYxYy0xYjdjLTQxMjctYjYyMi0zYTJhMzYyMGNkOTciLCJ0eXAiOiJSZWZyZXNoIiwiYXpwIjoid2NzIiwic2Vzc2lvbl9zdGF0ZSI6Ijg0NjUyMTA4LWNjYjctNGZmYy1hNGEwLWFjNGQyMDQ2Mzk4OCIsInNjb3BlIjoicHJvZmlsZSBvcGVuaWQgZW1haWwiLCJzaWQiOiI4NDY1MjEwOC1jY2I3LTRmZmMtYTRhMC1hYzRkMjA0NjM5ODgifQ.XWO5cQU8zkBa3PLu0O-2OArcmTXFgJ6r0kWmYHSgFvY"  # Add your full token here

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
                
