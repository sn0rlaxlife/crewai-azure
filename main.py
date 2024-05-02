import sys
import os
from dotenv import load_dotenv
from crewai import Crew, Process, Agent, Task
from tools import search_tools
from tools.search_tools import SearchTools
from langchain_openai import AzureChatOpenAI

load_dotenv()

class CrewAI():
    def __init__(self):
        self.search_tools = SearchTools()

    def run_task(self, task, query):
        if task == "search_internet":
            return self.search_tools.search_internet(query)
        elif task == "search_news":
            return self.search_tools.search_news(query)
        else:
            return "Unknown task"

crew_ai = CrewAI()

default_llm = AzureChatOpenAI(
    openai_api_version=os.environ.get("AZURE_OPENAI_VERSION", "<xxx>-<x>-<x>-<x>"),
    azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "<deployment-name>"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", "https://<deployment>.openai.azure.com/"),
    api_key=os.environ.get("AZURE_OPENAI_KEY")
)

# Create a researcher agent
researcher = Agent(
    role='Senior Researcher',
    goal='To identify the next big trend in AI and its security threats and vulnerabilities',
    verbose=True,
    llm=default_llm,
    backstory='A researcher with a background in AI and cybersecurity'
  )


# Create a task
def local_agent(self):
    search_tools = SearchTools()
    return Agent(
        role='Elite Security Researcher',
        goal='To identify the next big trend in AI and its security threats and vulnerabilities',
        backstory='A researcher with a background in AI and cybersecurity',
        verbose=True,
        tools=[
          search_tools.search_internet,
          search_tools.search_news
          ],
        llm=default_llm,
    )

# Now you can call the function with your agent
search_results = crew_ai.run_task("search_internet", "Existing and emerging threats in AI/LLMs security")
# Task for the researcher
research_task = Task(
  description='Identify the security threats and vulnerabilities in the next big trend in LLMs and propose solutions to mitigate them',
  agent=researcher,  # Assigning the task to the researcher
  llm=default_llm,  # Assigning the LLM to the task
  expected_output='A detailed report on the next big trend in AI for executives'
)


# Instantiate your crew
tech_crew = Crew(
  agents=[researcher, local_agent(self=local_agent)],  # Add the researcher and the local agent to the crew
  tasks=[research_task],
  process=Process.sequential  # Tasks will be executed one after the other
)

# Begin the task execution
tech_crew.kickoff()
