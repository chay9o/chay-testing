import weaviate
import jwt
import datetime
import warnings

warnings.filterwarnings("ignore")


class JWTAuthManager:
    def __init__(self, secret, issuer, audience, user_id, name):
        """
        Initialize JWT authentication details.
        :param secret: JWT secret key (from Docker Compose).
        :param issuer: JWT issuer URL (from Docker Compose).
        :param audience: JWT audience (from Docker Compose).
        :param user_id: User ID to include in the token payload.
        :param name: User name to include in the token payload.
        """
        self.secret = secret
        self.issuer = issuer
        self.audience = audience
        self.user_id = user_id
        self.name = name

    def generate_long_term_token(self, validity_days=3650):
        """
        Generate a long-term JWT token.
        :param validity_days: Validity period in days (default: 10 years).
        :return: JWT token string.
        """
        payload = {
            "sub": self.user_id,
            "name": self.name,
            "iat": datetime.datetime.utcnow(),
            "exp": datetime.datetime.utcnow() + datetime.timedelta(days=validity_days),
            "aud": self.audience,
            "iss": self.issuer,
        }
        token = jwt.encode(payload, self.secret, algorithm="HS256")
        return token


class ClientCredentials:
    def __init__(self):
        """
        Initialize the client and connect to Weaviate using JWT authentication.
        """
        try:
            # JWT Configuration
            jwt_manager = JWTAuthManager(
                secret = "6f8a482f3b8b6c9aaf1e5d7a02a7c5e6f4d18eaa1224ec6c9b342f7c8d3fa09e", 
                issuer="https://auth.wcs.api.weaviate.io",  # Replace with your JWT issuer
                audience="weaviate",  # Replace with your JWT audience
                user_id="chay.kusumanchi@strategicfuture.ai",  # User identifier
                name="Chay Kusumanchi"  # User name
            )

            # Generate long-term JWT
            jwt_token = jwt_manager.generate_long_term_token()
            print("Generated JWT Token:", jwt_token)

            # Connect to Weaviate using the JWT token
            self.weaviate_client = weaviate.connect_to_custom(
                http_host="w4.strategicfuture.ai",
                grpc_host="w4.strategicfuture.ai",
                grpc_port=50051,
                grpc_secure=False,
                headers={
                    "Authorization": f"Bearer {jwt_token}"
                },
            )

            # Test connection
            if self.weaviate_client.is_ready():
                print("Weaviate connection established successfully.")
            else:
                print("Weaviate is not ready.")
        except Exception as e:
            print(f"Error connecting to Weaviate: {e}")
