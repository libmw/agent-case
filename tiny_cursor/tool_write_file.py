from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

class WriteFileInput(BaseModel):
    filename: str = Field(description="The name of the file to write to")
    content: str = Field(description="The content to write to the file")


def write_file(filename: str, content: str):
    """Write content to a file"""
    with open(filename, 'w') as f:
        f.write(content)
    return f"Wrote {len(content)} characters to {filename}"
    
write_file_tool = StructuredTool.from_function(
    func=write_file,
    name="write_file",
    description="Write content to a file",
    args_schema=WriteFileInput
)
