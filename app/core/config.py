import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class TelegramConfig:
    token: str
    parse_mode: str

@dataclass
class DatabaseConfig:
    url: str
    echo: bool
    pool_size: int
    max_overflow: int

@dataclass
class RedisConfig:
    url: str

@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float
    timeout: float
    max_token: int
    thinking_effort: str

@dataclass
class CrewAIConfig:
    verbose: bool
    memory: bool
    output_file: str = ""
    embedder: str = ""
    knowledge: str = ""


@dataclass
class MCPConfig:
    enabled: bool
    timeout: float

@dataclass
class AppConfig:
    name: str
    environment: str
    debug: bool
    telegram: TelegramConfig
    database: DatabaseConfig
    redis: RedisConfig
    llm: LLMConfig
    crewcfg: CrewAIConfig
    mcp: MCPConfig
    max_memory_items: int
    assistant_history_limit: int

class Settings:
    def load() -> AppConfig:
        return AppConfig(
            name = os.getenv("APP_NAME"),
            environment= os.getenv("ENV"),
            debug = os.getenv("DEBUG"),
            telegram=TelegramConfig(
                token=os.getenv("TELEGRAM_TOKEN"),
                parse_mode="HTML"
            ),
            database=DatabaseConfig(
                url=os.getenv("DATABASE_URL"),
                echo=os.getenv("ECHO"),
                pool_size=10,
                max_overflow=20
            ),
            redis=RedisConfig(
                url=os.getenv("REDIS_URL")
            ),
            llm=LLMConfig(
                base_url=os.getenv("VLLM_URL"),
                api_key=os.getenv("VLLM_API_KEY"),
                model=os.getenv("VLLM_MODEL"),
                temperature=0.5,
                max_token=2048,
                thinking_effort="mid",
                timeout=180
            ),
            crewcfg=CrewAIConfig(
                verbose=False,
                memory=False,
            ),
            mcp=MCPConfig(
                enabled=True,
                timeout=360
            ),
            max_memory_items=int(os.getenv("MAX_MEMORY_ITEMS", "20")),
            assistant_history_limit=int(os.getenv("ASSISTANT_HISTORY_LIMIT", "20")),
        )

config = Settings.load()
