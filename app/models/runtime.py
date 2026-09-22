"""Explicit, serializable contracts for backend-owned agent runtime state."""

from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from models.agent_prompt import AgentPromptVersion
from models.memory import MemoryScope
from models.permission import ActionClass, PermissionSubjectType
from models.tool import ToolIntent, ToolRequest


class AgentRuntimeStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING_APPROVAL = "waiting_approval"


class RuntimeAgentKind(StrEnum):
    PRIMARY = "primary"
    USER = "user"
    SYSTEM = "system"
    TEMPORARY = "temporary"


class DelegationType(StrEnum):
    RESPOND = "respond"
    DELEGATE_USER_AGENT = "delegate_user_agent"
    DELEGATE_SYSTEM_AGENT = "delegate_system_agent"
    CREATE_TEMPORARY_SUBAGENT = "create_temporary_subagent"


class TemporarySubagentStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"


class RuntimeAgentIdentity(BaseModel):
    subject_type: PermissionSubjectType
    subject_id: str
    kind: RuntimeAgentKind
    name: str


class RuntimeSkillDefinition(BaseModel):
    """Serializable skill instructions resolved by backend services."""

    key: str
    name: str
    description: str
    version: int
    required_permissions: list[str] = Field(default_factory=list)
    instructions: str | None = None
    constraints: list[str] = Field(default_factory=list)
    storage_uri: str | None = None
    id: UUID | None = None
    package_checksum: str | None = None
    runtime: str | None = None
    entrypoint: str | None = None
    action_class: ActionClass = ActionClass.EXECUTE


class RuntimeMemoryItem(BaseModel):
    """A single memory entry already filtered for the active runtime."""

    memory_id: UUID | None = None
    scope: MemoryScope
    key: str
    content: str


class RuntimeAgentDefinition(BaseModel):
    identity: RuntimeAgentIdentity
    role: str
    goal: str
    backstory: str | None = None
    custom_instructions: str | None = None
    prompt_version_id: UUID | None = None
    can_spawn_subagents: bool = False
    active_skills: list[RuntimeSkillDefinition] = Field(default_factory=list)
    allowed_skill_keys: list[str] = Field(default_factory=list)
    default_skill_keys: list[str] = Field(default_factory=list)


class RuntimeChatMessage(BaseModel):
    role: str
    content: str


class RuntimeBudgets(BaseModel):
    max_time: float = 180.0
    max_llm_calls: int = 24
    max_tool_calls: int = 12
    max_delegations: int = 12
    max_subagents: int = 6
    max_tokens: int = 32_000
    max_retries: int = 3
    max_cost: float | None = None
    max_subagent_depth: int = 2
    max_subagents_per_parent: int = 3
    max_agent_iterations: int = 8


class RuntimeUsage(BaseModel):
    llm_calls: int = 0
    tool_calls: int = 0
    delegations: int = 0
    subagents: int = 0
    tokens: int = 0
    retries: int = 0
    cost: float | None = None


class DelegationDecision(BaseModel):
    type: DelegationType
    target_id: str | None = None
    task_summary: str | None = None
    temporary_name: str | None = None
    temporary_role: str | None = None
    temporary_goal: str | None = None
    temporary_backstory: str | None = None
    requested_permissions: list[str] = Field(default_factory=list)


class DelegationRecord(BaseModel):
    source: RuntimeAgentIdentity
    target: RuntimeAgentIdentity
    delegation_type: DelegationType
    task_summary: str
    timestamp: datetime
    success: bool | None = None
    error: str | None = None


class RuntimeTemporarySubagent(BaseModel):
    identity: RuntimeAgentIdentity
    parent: RuntimeAgentIdentity
    role: str
    goal: str
    backstory: str | None = None
    depth: int
    status: TemporarySubagentStatus = TemporarySubagentStatus.CREATED
    permissions: list[str] = Field(default_factory=list)
    active_skills: list[RuntimeSkillDefinition] = Field(default_factory=list)
    created_at: datetime
    completed_at: datetime | None = None


class PendingApprovalCheckpoint(BaseModel):
    approval_id: UUID
    flow_id: str
    tool_request: ToolRequest


class ToolApprovalFlowState(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    request: dict[str, object] = Field(default_factory=dict)
    run_id: str = ""
    user_id: int = 0
    approval_id: str | None = None
    runtime_permission_scopes: list[str] = Field(default_factory=list)
    runtime_skill_catalog: list[RuntimeSkillDefinition] = Field(default_factory=list)


class AgentRuntimeContext(BaseModel):
    run_id: UUID
    user_id: int
    starting_agent_id: UUID
    starting_prompt: AgentPromptVersion
    starting_agent: RuntimeAgentDefinition
    active_agent: RuntimeAgentIdentity
    chat_context: list[RuntimeChatMessage] = Field(default_factory=list)
    connected_persistent_agents: list[RuntimeAgentDefinition] = Field(default_factory=list)
    available_system_agents: list[RuntimeAgentDefinition] = Field(default_factory=list)
    active_skills: list[RuntimeSkillDefinition] = Field(default_factory=list)
    memory: list[RuntimeMemoryItem] = Field(default_factory=list)
    budgets: RuntimeBudgets = Field(default_factory=RuntimeBudgets)
    usage: RuntimeUsage = Field(default_factory=RuntimeUsage)
    delegation_state: list[DelegationRecord] = Field(default_factory=list)
    delegation_stack: list[RuntimeAgentIdentity] = Field(default_factory=list)
    pending_approval: PendingApprovalCheckpoint | None = None
    temporary_subagents: list[RuntimeTemporarySubagent] = Field(default_factory=list)
    original_message: str
    current_input: str
    runtime_started_at: datetime
    agent_iterations: dict[str, int] = Field(default_factory=dict)


class RuntimeStep(BaseModel):
    decision: DelegationDecision
    content: str | None = None
    tool_intent: ToolIntent | None = None
    tokens_used: int = 0
    retries: int = 0


class AgentRuntimeRequest(BaseModel):
    run_id: UUID
    user_id: int
    agent_id: UUID
    message: str
    chat_id: UUID | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class AgentRuntimeResult(BaseModel):
    status: AgentRuntimeStatus
    content: str | None = None
    error: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    context: AgentRuntimeContext | None = None
