import sys
import os

# Add both the parent directory (for utils) and current directory (for local modules)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.llm import llm

from run_command import (
    run_command_tool,
)  # pyright: ignore[reportImplicitRelativeImport]
from tool_read_file import (
    read_file_tool,
)  # pyright: ignore[reportImplicitRelativeImport]
from tool_write_file import (
    write_file_tool,
)  # pyright: ignore[reportImplicitRelativeImport]

tools = [run_command_tool, read_file_tool, write_file_tool]
llm_with_tools = llm.bind_tools(tools)

system_prompt = "you are a professional software engineer,you finish jobs from user's request.you have many tools to use."




# Get user input from command line
prompt = input("🤖 what can I do for you?\n")
print(f"👧 {prompt}\n")

# Initialize message history with the user's prompt
messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]

# Create a mapping of tool names to tool objects for easy lookup
tool_map = {tool.name: tool for tool in tools}

# Main loop to handle tool calls
while True:
    try:
        # Get response from LLM with current message history
        response = llm_with_tools.invoke(messages)
    except Exception as e:
        if "timeout" in str(e).lower():
            print("🤖 Request timed out. Retrying with shorter message history...")
            # Keep only the last few messages to reduce context size
            if len(messages) > 5:
                messages = messages[:2] + messages[-3:]  # Keep system + user + last 3 messages
            continue
        else:
            print(f"🤖 Error: {str(e)}")
            break

    # print("Response:", response)

    if response.content:
        print(f"🤖 {response.content}\n")

    # Add the assistant's response to message history
    assistant_message = {"role": "assistant", "content": response.content}
    if hasattr(response, "tool_calls") and response.tool_calls:
        assistant_message["tool_calls"] = response.tool_calls
    messages.append(assistant_message)

    # Check if the response has tool calls
    if hasattr(response, "tool_calls") and response.tool_calls:
        # print(f"\nDetected {len(response.tool_calls)} tool call(s)")

        # Execute each tool call and collect results
        tool_results = []
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call.get("id", f"call_{len(tool_results)}")

            print(f"🤖 Executing tool: {tool_name}")
            print(f"Arguments: {tool_args}\n")

            if tool_name in tool_map:
                try:
                    # Get the tool and invoke it with the arguments
                    tool = tool_map[tool_name]
                    result = tool.invoke(tool_args)
                    print(f"🤖 Tool result: {result}\n")

                    # Add tool result to collection
                    tool_results.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": tool_name,
                            "content": str(result),
                        }
                    )
                except Exception as e:
                    error_msg = f"Error executing tool {tool_name}: {str(e)}"
                    print(error_msg)

                    # Add error result to collection
                    tool_results.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": tool_name,
                            "content": error_msg,
                        }
                    )
            else:
                error_msg = f"🤖 Tool {tool_name} not found in available tools\n"
                print(error_msg)

                # Add error result to collection
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": error_msg,
                    }
                )

        # Add all tool results to message history
        messages.extend(tool_results)

        # Continue the loop to let the AI process the tool results
        continue
    else:
        print("🤖 Task finished!")
        # No more tool calls, break the loop
        break
