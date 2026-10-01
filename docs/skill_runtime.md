# Native Skills runtime

SkillService verifies tenant ownership, version, storage URI and package checksum.
SkillRuntimeResolver materializes the verified package for CrewAI native discovery.
Temporary files remain available throughout `Crew.akickoff()` and are removed on
success, failure and cancellation. Package Python files are never imported by the
CrewAI runtime; executable skills use the backend authority and sandbox.

## Production tools

AssistantService and ScheduleService select a production ToolRegistry. The same
executor is passed to native READ tools and BackendToolAuthority. FakeToolExecutor
requires explicit injection in tests or development.

Register real backend handlers with `RegisteredTool(definition, async_handler)`
and inject `ToolRegistry([...])` as `tool_executor`. The default registry has no
external connectors: `read_information`, `write_note` and `external_action` are
unavailable until a real handler is installed. Unknown tools fail closed.
`execute_skill` is reserved for BackendToolAuthority and its sandbox execution.

Native tools expose READ only, recheck skill lifecycle and permissions on every
invocation, share a tool budget and serialize access to their AsyncSession.
WRITE, EXECUTE and external side effects use ToolIntent and approval checkpoints.

## Skill lifecycle

- ACTIVE: available for new assignments and new runs.
- ARCHIVED: excluded from new assignments and resolution. Existing RUNNING or
  WAITING_APPROVAL runs may continue only with a matching persisted checkpoint pin
  for the immutable skill id, version, checksum, URI and execution metadata.
- DISABLED: revoked for all runs, including existing pins and already-created tools.

The same lifecycle rules apply to native instructions and sandbox execution after
approval. Archive retains storage; explicit administrative storage deletion can
still make a retained version unavailable. No new database enum or migration is
needed for these semantics. User-owned skills can be revoked through
`SkillService.disable_skill`; system skill status changes remain administrative.

## CrewAI compatibility

CrewAI is pinned to 1.15.22. Runtime uses its default AgentExecutor with
`akickoff()`. LLM/tool Flow nodes run in worker threads. BackendReadTool bridges
synchronous tool calls to its owning event loop with run_coroutine_threadsafe;
permissions, lifecycle checks, AsyncSession and the async handler run on that loop.
Both native function calling and ReAct pass the owner-loop integration test.

Cancellation waits for the in-flight kickoff to finish before deleting native
Skill files. This keeps files available to worker threads; cancellation can take
until the bounded current agent execution completes.

The design crew receives the allowed persistent-agent memory scopes. Runtime
prompts identify the active agent and distinguish native tool calls from backend
ToolIntent actions. Workers execute their delegated task and return results to
the primary coordinator.

Run `python scripts/live_skill_chat.py` from the project virtualenv for a real
S3 + LLM acceptance dialogue through application chat handlers. Telegram delivery
is recorded locally, SQL data uses an isolated SQLite database, and a real S3 READ
handler is explicitly registered for the test. Reports contain the dialogue,
native skill/tool events, run usage and cleanup evidence. Only the unique test
S3 package is removed afterwards.

## Local connections

Keep connection secrets in the Git-ignored `.env`. S3 uses S3_ENDPOINT,
S3_ACCESS_KEY, S3_SECRET_KEY, S3_REGION, S3_SKILL_BUCKET and S3_USE_SSL.
CrewAI LLM uses VLLM_URL, VLLM_MODEL and VLLM_API_KEY. For a custom endpoint with an
intentionally empty API key, the client uses a non-secret `not-required` transport
value required by CrewAI; it does not inherit OPENAI_API_KEY from the environment.

## Persistent Crew orchestration

Each new AgentRun snapshots a lightweight catalog of the user's ACTIVE crews:
id, name and purpose. Only the Primary sees this catalog in its prompt and can
return `RUN_CREW` with a listed id and a concrete task summary. Old checkpoints
default to an empty catalog. Crews created after a run starts become available
to the next new run.

AgentRuntimeService validates the selected id against the snapshot, then checks
current ownership/status, members and required skills through CrewService.
DynamicCrewAIRuntime runs the existing resolved crew with the invocation input
added to copies of its tasks. Stored tasks are not modified. The result becomes
the Primary's current input for synthesis within the same AgentRun.

Crew inference and READ tools share the remaining run budgets. A common LLM
guard checks each call before reaching the provider, including forced-final
and conversion calls; consumed calls and tools are retained even on failure.
Crew invocations use the existing delegation requested/completed/failed audit
events with `delegation_type=run_crew`. The new catalog and decision need no SQL
migration. Native Skills, tool permissions and approval boundaries are retained.

Run `python scripts/live_skill_chat.py --crew` for the real S3 + LLM dialogue:
create and confirm a two-member crew, select it through RUN_CREW, load its assigned
native Skill, read source data from S3, and return its report through Primary.
The same local transport, isolated SQL data and test-package cleanup apply.
