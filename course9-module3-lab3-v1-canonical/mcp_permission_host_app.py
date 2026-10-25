import json
import sys
from typing import TypedDict

import gradio as gr
from dotenv import load_dotenv
from mcp.types import TextContent, TextResourceContents
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

from mcp_permission_client_base import MCPPermissionClient


class PendingApproval(TypedDict):
    tool_name: str
    arguments: dict[str, object]


class RiskAssessment(TypedDict):
    tool: str
    risk_level: str
    permission: str
    requires_approval: bool
    description: str


class MCPPermissionHostApp(MCPPermissionClient):
    """AI host application with permission enforcement and risk assessment."""

    def __init__(self, server_script: str) -> None:
        super().__init__(server_script)
        self.llm_client = OpenAI()
        self.model = "gpt-4o-mini"
        self.conversation_history: list[ChatCompletionMessageParam] = []
        self.pending_approval: PendingApproval | None = None  # Track tool waiting for approval
        self.risk_levels: dict[str, str] = {
            "read_file": "low",
            "write_file": "medium",
            "delete_file": "high",
            "execute_command": "critical",
        }

    async def get_available_tools(self) -> list[ChatCompletionToolParam]:
        """Get all available tools in OpenAI function calling format with permission info."""
        await self.connect()

        # Get real MCP tools
        mcp_tools = await self.list_tools()

        openai_tools: list[ChatCompletionToolParam] = []

        # Add real MCP tools with permission information
        for tool in mcp_tools:
            parameters: dict[str, object] = {"type": "object", "properties": {}}

            # Add permission and risk level to description
            permission = self.permissions.get(tool.name, "ask")
            risk = self.risk_levels.get(tool.name, "medium")
            description = tool.description or f"Execute {tool.name}"
            description += f" (Permission: {permission}, Risk: {risk})"

            # Convert MCP input schema to OpenAI parameters
            schema = tool.inputSchema
            if schema:
                if "properties" in schema:
                    parameters["properties"] = schema["properties"]
                if "required" in schema and schema["required"]:
                    parameters["required"] = schema["required"]

            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": description,
                        "parameters": parameters,
                    },
                }
            )

        # Add synthetic tools for resources
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": "mcp_list_resources",
                    "description": "List all available resources from the MCP server",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        )

        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": "mcp_read_resource",
                    "description": "Read a specific resource by URI from the MCP server",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "uri": {
                                "type": "string",
                                "description": "The URI of the resource to read (for example, 'file://audit/log')",
                            }
                        },
                        "required": ["uri"],
                    },
                },
            }
        )

        # Add synthetic tools for prompts
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": "mcp_list_prompts",
                    "description": "List all available prompt templates from the MCP server",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        )

        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": "mcp_get_prompt",
                    "description": "Get a rendered prompt template from the MCP server",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The name of the prompt template",
                            },
                            "arguments": {
                                "type": "object",
                                "description": "Arguments for the prompt template",
                            },
                        },
                        "required": ["name"],
                    },
                },
            }
        )

        return openai_tools

    async def execute_tool(self, tool_name: str, arguments: dict[str, object]) -> str:
        """Execute a tool call with permission checking (real MCP tool or synthetic helper)."""
        await self.connect()

        # Handle synthetic resource tools
        if tool_name == "mcp_list_resources":
            resources = await self.list_resources()
            result = "Available resources:\n"
            for resource in resources:
                result += f"- {resource.uri}"
                if resource.name:
                    result += f" ({resource.name})"
                if resource.description:
                    result += f": {resource.description}"
                result += "\n"
            return result

        if tool_name == "mcp_read_resource":
            uri = arguments.get("uri")
            if not isinstance(uri, str) or not uri:
                return "Error: URI is required"
            try:
                contents = await self.read_resource(uri)
                if len(contents) > 0:
                    content = contents[0]
                    if isinstance(content, TextResourceContents):
                        return content.text
                    return str(content)
                return str(contents)
            except Exception as e:
                return f"Error reading resource: {str(e)}"

        # Handle synthetic prompt tools
        if tool_name == "mcp_list_prompts":
            prompts = await self.list_prompts()
            result = "Available prompts:\n"
            for prompt in prompts:
                result += f"- {prompt.name}"
                if prompt.description:
                    result += f": {prompt.description}"
                if prompt.arguments:
                    args = [arg.name for arg in prompt.arguments]
                    result += f" (args: {', '.join(args)})"
                result += "\n"
            return result

        if tool_name == "mcp_get_prompt":
            name = arguments.get("name")
            raw_prompt_args = arguments.get("arguments", {})
            if not isinstance(name, str) or not name:
                return "Error: Prompt name is required"
            prompt_args = (
                {str(key): str(value) for key, value in raw_prompt_args.items()}
                if isinstance(raw_prompt_args, dict)
                else {}
            )
            try:
                messages = await self.get_prompt(name, prompt_args)
                result = f"Prompt: {name}\n\n"
                for msg in messages:
                    content = msg.content
                    text = content.text if isinstance(content, TextContent) else str(content)
                    result += f"[{msg.role}]: {text}\n\n"
                return result
            except Exception as e:
                return f"Error getting prompt: {str(e)}"

        # Handle regular MCP tools with permission checking
        try:
            # Use inherited permission-aware tool execution
            result = await self.call_tool_with_permission(tool_name, arguments)

            # Extract text content from result
            if len(result) > 0:
                content = result[0]
                if isinstance(content, TextContent):
                    result_text = content.text
                    # Check if this is an approval request
                    if (
                        "Permission required for tool:" in result_text
                        and "Please approve this operation" in result_text
                    ):
                        # Store pending approval
                        self.pending_approval = {"tool_name": tool_name, "arguments": arguments}
                    return result_text
                return str(content)

            return str(result)

        except Exception as e:
            return f"Error executing tool: {str(e)}"

    def assess_risk(self, tool_name: str, arguments: dict[str, object]) -> RiskAssessment:
        """Assess the risk level of a tool operation."""
        risk_level = self.risk_levels.get(tool_name, "medium")
        permission = self.permissions.get(tool_name, "ask")

        assessment: RiskAssessment = {
            "tool": tool_name,
            "risk_level": risk_level,
            "permission": permission,
            "requires_approval": permission in ["ask", "deny"],
            "description": "",
        }

        # Add risk-specific descriptions
        if risk_level == "low":
            assessment["description"] = "Safe operation with minimal impact"
        elif risk_level == "medium":
            assessment["description"] = "Moderate impact - modifies data"
        elif risk_level == "high":
            assessment["description"] = "High impact - destructive operation"
        elif risk_level == "critical":
            assessment["description"] = "Critical impact - system-level operation"

        return assessment

    async def chat(self, user_message: str, history: list[dict[str, str | None]]) -> str | None:
        """Chat with the LLM using permission-aware MCP tools."""
        await self.connect()

        # Check if this is an approval response for a pending operation
        if self.pending_approval and user_message.strip().lower() in [
            "yes",
            "approve",
            "ok",
            "confirm",
            "y",
        ]:
            # Execute the pending tool with approval
            tool_name = self.pending_approval["tool_name"]
            arguments = self.pending_approval["arguments"]
            self.pending_approval = None  # Clear pending approval

            try:
                result = await self.call_tool_with_permission(tool_name, arguments, approved=True)

                # Extract text content
                if len(result) > 0:
                    content = result[0]
                    if isinstance(content, TextContent):
                        response_text = f"Operation approved and executed.\n\n{content.text}"
                    else:
                        response_text = f"Operation approved and executed.\n\n{str(content)}"
                else:
                    response_text = f"Operation approved and executed.\n\n{str(result)}"

                # Add to conversation history
                self.conversation_history.append({"role": "user", "content": user_message})
                self.conversation_history.append({"role": "assistant", "content": response_text})

                return response_text

            except Exception as e:
                error_msg = f"Error executing approved operation: {str(e)}"
                self.conversation_history.append({"role": "user", "content": user_message})
                self.conversation_history.append({"role": "assistant", "content": error_msg})
                return error_msg

        # Check if this is a denial response for a pending operation
        if self.pending_approval and user_message.strip().lower() in ["no", "deny", "cancel", "n"]:
            self.pending_approval = None  # Clear pending approval
            response_text = "Operation cancelled by user."

            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": response_text})

            return response_text

        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": user_message})

        # Get available tools
        tools = await self.get_available_tools()

        # Call OpenAI with tools (only pass tools if they exist)
        if tools:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                tools=tools,
                tool_choice="auto",
            )
        else:
            response = self.llm_client.chat.completions.create(
                model=self.model, messages=self.conversation_history
            )

        if not response or not response.choices:
            return "Error: No response from LLM"

        assistant_message = response.choices[0].message

        # Handle tool calls
        if assistant_message.tool_calls:
            # Add assistant's message with tool calls to history
            self.conversation_history.append(
                {
                    "role": "assistant",
                    "content": assistant_message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in assistant_message.tool_calls
                        if isinstance(tc, ChatCompletionMessageFunctionToolCall)
                    ],
                }
            )

            # Execute each tool call with permission checking
            for tool_call in assistant_message.tool_calls:
                if not isinstance(tool_call, ChatCompletionMessageFunctionToolCall):
                    continue
                function_name = tool_call.function.name
                function_args: dict[str, object] = json.loads(tool_call.function.arguments)

                # Assess risk before execution
                self.assess_risk(function_name, function_args)

                # Execute the tool (permission checking happens inside)
                tool_result = await self.execute_tool(function_name, function_args)

                # Add tool result to history
                self.conversation_history.append(
                    {"role": "tool", "tool_call_id": tool_call.id, "content": str(tool_result)}
                )

            # Get final response after tool execution
            final_response = self.llm_client.chat.completions.create(
                model=self.model, messages=self.conversation_history
            )

            if not final_response or not final_response.choices:
                return "Error: No response from LLM after tool execution"

            final_message = final_response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": final_message})

            return final_message

        else:
            # No tool calls, just return the response
            self.conversation_history.append(
                {"role": "assistant", "content": assistant_message.content}
            )

            return assistant_message.content

    def create_interface(self) -> gr.Blocks:
        """Create the Gradio chat interface with permission awareness."""

        async def chat_wrapper(
            message: str, history: list[dict[str, str | None]]
        ) -> list[dict[str, str | None]]:
            """Wrapper for chat method compatible with Gradio."""
            if not message.strip():
                return history

            response = await self.chat(message, history)
            # Return updated history with new messages
            return history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response},
            ]

        async def reset_conversation() -> list[dict[str, str | None]]:
            """Reset the conversation history."""
            self.conversation_history = []
            return []

        with gr.Blocks(title="MCP Permission AI Host") as interface:
            gr.Markdown(f"""
            # MCP Permission AI Host
            Chat with GPT-4o-mini using permission-aware MCP tools.

            **Model:** {self.model}
            **Permissions:** Enforced with audit logging
            **Risk Assessment:** Automatic for all operations

            All tool executions are subject to permission policies and logged to the audit trail.
            """)

            chatbot = gr.Chatbot(label="Conversation", height=500, type="messages")

            with gr.Row():
                msg = gr.Textbox(
                    label="Your message", placeholder="Ask me to use MCP tools...", scale=4
                )
                clear = gr.Button("Clear", scale=1)

            with gr.Accordion("Permission Status", open=False):
                gr.Markdown(self._get_permission_summary())

            msg.submit(fn=chat_wrapper, inputs=[msg, chatbot], outputs=chatbot).then(
                lambda: "", outputs=msg
            )

            clear.click(fn=reset_conversation, outputs=chatbot)

        return interface

    def _get_permission_summary(self) -> str:
        """Generate a summary of current permissions."""
        summary = "### Current Permission Policies:\n\n"
        for tool, policy in self.permissions.items():
            risk = self.risk_levels.get(tool, "medium")
            summary += f"- **{tool}**: {policy.upper()} (Risk: {risk})\n"
        return summary


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python mcp_permission_host_app.py <server_script>")
        print("Example: python mcp_permission_host_app.py mcp_permission_server.py")
        sys.exit(1)

    load_dotenv()

    server_script = sys.argv[1]

    client = MCPPermissionHostApp(server_script)
    interface = client.create_interface()
    interface.queue().launch(server_name="127.0.0.1", server_port=7864)


if __name__ == "__main__":
    main()
