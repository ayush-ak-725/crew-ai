from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import FileWriterTool

file_writer_tool_summary = FileWriterTool(file_name='summary.txt', directory='meeting_minutes')
file_writer_tool_action_items = FileWriterTool(file_name='action_items.txt', directory='meeting_minutes')
file_writer_tool_sentiment = FileWriterTool(file_name='sentiment_analysis.txt', directory='meeting_minutes')


@CrewBase
class MeetingMinutesCrew:
    """Meeting Minutes Crew"""

    ollama_llm = LLM(
        model="ollama/qwen3:8b",
        base_url="http://localhost:11434",
        temperature=0.3,   # important: reduces hallucinated tool calls
    )

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def meeting_minutes_summarizer(self) -> Agent:
        return Agent(
            config=self.agents_config["meeting_minutes_summarizer"],  # type: ignore[index]
            tools=[file_writer_tool_summary, file_writer_tool_action_items, file_writer_tool_sentiment],
            llm=self.ollama_llm,
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def meeting_minutes_writer(self) -> Agent:
        return Agent(
            config=self.agents_config["meeting_minutes_writer"],  # type: ignore[index]
            llm=self.ollama_llm
        )


    @task
    def meeting_minutes_summary_task(self) -> Task:
        return Task(
            config=self.tasks_config["meeting_minutes_summarizer_task"],  # type: ignore[index]
        )

    @task
    def meeting_minutes_writing_task(self) -> Task:
        return Task(
            config=self.tasks_config["meeting_minutes_writer_task"],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Content Crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
