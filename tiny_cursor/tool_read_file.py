from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

class ReadFileInput(BaseModel):
    file_path: str = Field(description="The path to the file to read")

def read_file(file_path: str) -> str:
    """Reads the content of a file and returns it as a string."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"

read_file_tool = StructuredTool.from_function(
    func=read_file,
    name="read_file",
    description="Reads the content of a file and returns it as a string. Input is a file path.",
    args_schema=ReadFileInput,
)
