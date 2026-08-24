from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from integrations.llm.factory import create_crewai_llm


@CrewBase
class AssistantCrew:
    agents_config = "crewai/config/agents.yaml"
    tasks_config = "crewai/config/tasks.yaml"

    @agent
    def assistant(self) -> Agent:
        return Agent(
            config=self.agents_config["assistant"],
            llm=create_crewai_llm(),
            verbose=False,
            allow_delegation=False,
        )

    @task
    def respond(self) -> Task:
        return Task(
            config=self.tasks_config["respond"],
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=False,
        )