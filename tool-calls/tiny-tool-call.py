import sys
import os



from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

# Add the project root to sys.path to allow imports from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm import llm

# ====================== 2. 定义参数 Schema（和你之前的 BaseTool 用法一致） ======================
class ReadMarkdownArgsSchema(BaseModel):
    """读取Markdown文件的参数模型（BaseTool 必须显式定义）"""

    file_path: str = Field(
        description="markdown文件的路径（相对/绝对路径），例如 tiny-tool-call-readme.md",
        examples=["tiny-tool-call-readme.md", "./docs/test.md"],
    )


# ====================== 3. 用 BaseTool 定义工具（你熟悉的能正常工作的方式） ======================
class MarkdownSummaryTool(BaseTool):
    # 工具名称
    name: str = "read_markdown_file"
    # 工具描述（明确参数名）
    description: str = (
        "用于读取markdown文件的纯文本内容，必须传入file_path参数指定文件路径"
    )
    # 核心：绑定参数 schema（这是 BaseTool 能返回正常参数名的关键）
    args_schema: type[BaseModel] = ReadMarkdownArgsSchema

    def _run(self, file_path: str) -> str:
        """同步执行逻辑（核心工具功能）"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                md_content = f.read()
            # 可选：清理 markdown 格式（和你之前的逻辑一致）
            return f"Markdown 文件 {file_path} 内容：\n{md_content}"
        except FileNotFoundError:
            return f"错误：未找到文件 {file_path}"
        except Exception as e:
            return f"读取失败：{str(e)}"


# ====================== 4. 绑定工具并调用（和你之前的逻辑一致） ======================
# 初始化工具实例
markdown_tool = MarkdownSummaryTool()

# 绑定工具到 LLM
llm_with_tools = llm.bind_tools([markdown_tool])

# 测试指令
prompt = "请读取 tiny-tool-call-readme.md 文件的内容并总结其核心信息"

# 调用 LLM
response = llm_with_tools.invoke(prompt)

# 解析工具调用结果
if response.tool_calls:
    print("\n=== 工具调用请求 ===")
    for tool_call in response.tool_calls:  # pyright: ignore[reportUnknownMemberType]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        print(f"工具名称：{tool_name}")
        print(
            f"工具参数：{tool_args}"
        )  # 现在会正常返回 {'file_path': 'tiny-tool-call-readme.md'}

        # 执行工具
        if tool_name == "read_markdown_file":
            tool_result = markdown_tool.run(tool_args)
            print("\n=== 工具执行结果 ===")
            print(tool_result[:200] + "..." if len(tool_result) > 200 else tool_result)

            # 生成最终总结
            final_response = llm.invoke(
                f"基于以下内容总结核心信息，分点说明：\n{tool_result}"
            )
            print("\n=== 最终总结 ===")
            print(final_response.content)
else:
    print("未调用工具，直接返回：", response.content)
