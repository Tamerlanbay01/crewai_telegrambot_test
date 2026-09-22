import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class TelegramConfig:
    token: str
    parse_mode: str | None

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
class EmbeddingConfig:
    base_url: str
    api_key: str
    model: str
    timeout: float
    dimension: int


@dataclass
class QdrantConfig:
    url: str
    api_key: str
    collection_name: str
    timeout: float


@dataclass
class S3Config:
    endpoint: str
    access_key: str
    secret_key: str
    region: str
    skill_bucket: str
    use_ssl: bool
    max_skill_md_bytes: int
    max_skill_package_bytes: int
    max_skill_file_bytes: int


@dataclass
class SandboxConfig:
    python_image: str = "python:3.12.7-slim"
    timeout_seconds: int = 30
    memory_mb: int = 256
    cpus: float = 0.5
    pids_limit: int = 64
    max_stdout_bytes: int = 256 * 1024
    max_stderr_bytes: int = 256 * 1024

@dataclass
class CrewAIConfig:
    verbose: bool
    memory: bool
    output_file: str = ""
    embedder: str = ""
    knowledge: str = ""
    approval_persistence_path: str = ".crewai/flow_states.db"


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
    embedding: EmbeddingConfig
    qdrant: QdrantConfig
    s3: S3Config
    sandbox: SandboxConfig
    crewcfg: CrewAIConfig
    mcp: MCPConfig
    max_memory_items: int
    memory_semantic_top_k: int
    memory_min_confidence: float
    memory_processing_batch_size: int
    memory_extraction_max_attempts: int
    memory_extraction_lease_seconds: int
    memory_extraction_retry_base_seconds: int
    assistant_history_limit: int

class Settings:
    def load() -> AppConfig:
        return AppConfig(
            name = os.getenv("APP_NAME"),
            environment= os.getenv("ENV"),
            debug = os.getenv("DEBUG"),
            telegram=TelegramConfig(
                token=os.getenv("TELEGRAM_TOKEN"),
                parse_mode=None
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
            embedding=EmbeddingConfig(
                base_url=os.getenv("EMBEDDING_BASE_URL", ""),
                api_key=os.getenv("EMBEDDING_API_KEY", ""),
                model=os.getenv("EMBEDDING_MODEL", ""),
                timeout=float(os.getenv("EMBEDDING_TIMEOUT", "30")),
                dimension=int(os.getenv("EMBEDDING_DIMENSION", "1536")),
            ),
            qdrant=QdrantConfig(
                url=os.getenv("QDRANT_URL", ""),
                api_key=os.getenv("QDRANT_API_KEY", ""),
                collection_name=os.getenv("QDRANT_MEMORY_COLLECTION", "agent_memory"),
                timeout=float(os.getenv("QDRANT_TIMEOUT", "10")),
            ),
            s3=S3Config(
                endpoint=os.getenv("S3_ENDPOINT", ""),
                access_key=os.getenv("S3_ACCESS_KEY", ""),
                secret_key=os.getenv("S3_SECRET_KEY", ""),
                region=os.getenv("S3_REGION", "us-east-1"),
                skill_bucket=os.getenv("S3_SKILL_BUCKET", "agent-skills"),
                use_ssl=os.getenv("S3_USE_SSL", "false").strip().lower()
                in {"1", "true", "yes", "on"},
                max_skill_md_bytes=int(os.getenv("MAX_SKILL_MD_BYTES", str(64 * 1024))),
                max_skill_package_bytes=int(
                    os.getenv("MAX_SKILL_PACKAGE_BYTES", str(20 * 1024 * 1024))
                ),
                max_skill_file_bytes=int(
                    os.getenv("MAX_SKILL_FILE_BYTES", str(10 * 1024 * 1024))
                ),
            ),
            sandbox=SandboxConfig(
                python_image=os.getenv(
                    "SKILL_SANDBOX_PYTHON_IMAGE", "python:3.12.7-slim"
                ),
                timeout_seconds=int(os.getenv("SKILL_SANDBOX_TIMEOUT_SECONDS", "30")),
                memory_mb=int(os.getenv("SKILL_SANDBOX_MEMORY_MB", "256")),
                cpus=float(os.getenv("SKILL_SANDBOX_CPUS", "0.5")),
                pids_limit=int(os.getenv("SKILL_SANDBOX_PIDS_LIMIT", "64")),
                max_stdout_bytes=int(
                    os.getenv("SKILL_SANDBOX_MAX_STDOUT_BYTES", str(256 * 1024))
                ),
                max_stderr_bytes=int(
                    os.getenv("SKILL_SANDBOX_MAX_STDERR_BYTES", str(256 * 1024))
                ),
            ),
            crewcfg=CrewAIConfig(
                verbose=False,
                memory=False,
                approval_persistence_path=os.getenv(
                    "CREWAI_APPROVAL_PERSISTENCE_PATH",
                    ".crewai/flow_states.db",
                ),
            ),
            mcp=MCPConfig(
                enabled=True,
                timeout=360
            ),
            max_memory_items=int(os.getenv("MAX_MEMORY_ITEMS", "20")),
            memory_semantic_top_k=int(os.getenv("MEMORY_SEMANTIC_TOP_K", "10")),
            memory_min_confidence=float(os.getenv("MEMORY_MIN_CONFIDENCE", "0.75")),
            memory_processing_batch_size=int(
                os.getenv("MEMORY_PROCESSING_BATCH_SIZE", "10")
            ),
            memory_extraction_max_attempts=int(
                os.getenv("MEMORY_EXTRACTION_MAX_ATTEMPTS", "5")
            ),
            memory_extraction_lease_seconds=int(
                os.getenv("MEMORY_EXTRACTION_LEASE_SECONDS", "300")
            ),
            memory_extraction_retry_base_seconds=int(
                os.getenv("MEMORY_EXTRACTION_RETRY_BASE_SECONDS", "30")
            ),
            assistant_history_limit=int(os.getenv("ASSISTANT_HISTORY_LIMIT", "20")),
        )

config = Settings.load()
