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
            # Add your full token here

            oidc_token = "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJUOU1HUG1WOTZna1ljbFhqQjl1ZG52ZktKZlJlU0lpSTRrS3FjQi1SWTlJIn0.eyJleHAiOjE3MzI4MzQwNzYsImlhdCI6MTczMjgzMzE3NiwianRpIjoiZWVkMWMxZjUtMzkwYy00YjAzLWIxOWQtMmUzMGU3MDVhOThmIiwiaXNzIjoiaHR0cHM6Ly9hdXRoLndjcy5hcGkud2VhdmlhdGUuaW8vYXV0aC9yZWFsbXMvU2VNSSIsImF1ZCI6WyJ3Y3MiLCJhY2NvdW50Il0sInN1YiI6IjZmYWUzNjFjLTFiN2MtNDEyNy1iNjIyLTNhMmEzNjIwY2Q5NyIsInR5cCI6IkJlYXJlciIsImF6cCI6IndjcyIsInNlc3Npb25fc3RhdGUiOiI1ZTQzYWE1YS0xODM5LTQ2ZTUtODFjYy1lZDA4YWI5ZjQwYzUiLCJhbGxvd2VkLW9yaWdpbnMiOlsiaHR0cDovL2NvbnNvbGUuc2VtaS50ZWNobm9sb2d5IiwiaHR0cHM6Ly9jb25zb2xlLndlYXZpYXRlLmlvLyoiLCJodHRwczovL2NvbnNvbGUud2VhdmlhdGUuaW8iLCJodHRwOi8vY29uc29sZS5zZW1pLnRlY2hub2xvZ3kvKiIsImh0dHBzOi8vYWNjZW50dXJlMDAxLmRlbW8uc2VtaS50ZWNobm9sb2d5IiwiaHR0cDovL3BsYXlncm91bmQuc2VtaS50ZWNobm9sb2d5LyoiLCJodHRwczovL2NvbnNvbGUuc2VtaS50ZWNobm9sb2d5Il0sInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJkZWZhdWx0LXJvbGVzLXNlbWkiLCJvZmZsaW5lX2FjY2VzcyIsInVtYV9hdXRob3JpemF0aW9uIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwic2NvcGUiOiJwcm9maWxlIG9wZW5pZCBlbWFpbCIsInNpZCI6IjVlNDNhYTVhLTE4MzktNDZlNS04MWNjLWVkMDhhYjlmNDBjNSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJuYW1lIjoibi5hLiBuLmEuIiwicHJlZmVycmVkX3VzZXJuYW1lIjoiY2hheS5rdXN1bWFuY2hpQHN0cmF0ZWdpY2Z1dHVyZS5haSIsImdpdmVuX25hbWUiOiJuLmEuIiwiZmFtaWx5X25hbWUiOiJuLmEuIiwiZW1haWwiOiJjaGF5Lmt1c3VtYW5jaGlAc3RyYXRlZ2ljZnV0dXJlLmFpIn0.XA-VTEX33YMjivhiYrYBNkYD2PlpbM2ryqZ_l2qUslHyMICHjqPV7FVHZzDwFbsA6KCukbJr6Skc0N00xz03_dZrArs6_UbOtpoHs0KZ799ayGOBz5QR-bbGDQp8mpOxrmP850ggerdYI2bIU5AXliYTTL1HxyxDBCYdVZPUqWsQbxtOJQGst6w020ic6eA9fOf7dAX6hw3ffiLes4XwK-amskQVAhZbCbIIOHFF6tL6KDxbXDkUi9IA5DrxfLocIo7tk41RZKe4aKR8HYa7z1ABrZ3uT1YeM-gYLnldhrCnPd-3OREpAR8GO0Im_O6aRCvDSLNImQd3Vd4vcBYz6w"
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
                
