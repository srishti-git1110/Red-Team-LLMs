import dotenv
from pydantic import Field
from pydantic_settings import BaseSettings


dotenv.load_dotenv()

class APIConfig(BaseSettings):
    hf_access_token: str = Field(..., env="hf_access_token")

config = APIConfig()