"""Runtime-facing Pydantic models."""
from enum import StrEnum
from uuid import UUID
from pydantic import BaseModel, Field 

class AgentRuntimeStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING_APPROVAL = "waiting_approval"
    
class AgentRuntimeRequest(BaseModel):
    run_id: UUID
    user_id: int
    agent_id: int
    message: str
    chat_id: UUID | None = None
    metadata: dict[str, object] = Field(default_factory = dict)
    
class AgentRuntimeResult(BaseModel):
    status: AgentRuntimeStatus
    content: str | None = None
    error: str | None = None
    metadata: dict[str, object] = Field(default_factory = dict)
    