from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
import subprocess
import threading


class RunCmdInput(BaseModel):

    cmd: str = Field(description="The command to run")

    autoYes: bool = Field(
        description="Automatically answer yes to any prompts", default=True
    )


def run_command(cmd: str, autoYes: bool = True):
    """Run a command and return the output"""

    print(f"🤖 Running command: {cmd}\n")


    if autoYes:
        cmd = "yes | " + cmd

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    return result.stdout + result.stderr


run_command_tool = StructuredTool.from_function(
    func=run_command,
    name="run_command",
    description="Run a command and return the output",
    args_schema=RunCmdInput,
)
