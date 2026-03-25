from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    admin_token: str = "change-me"
    rbi_internal_secret: str = "change-me-internal"
    docker_network: str = "rbi_net"
    session_image: str = "rbi-browser-session"


settings = Settings()
