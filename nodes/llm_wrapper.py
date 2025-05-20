from typing import overload
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, AnyMessage
from langchain_core.runnables import Runnable
from langchain_litellm import ChatLiteLLM


class LLMWrapper:
    """LLM Node to reason about the next player character's action
    """
    def __init__(self, system_prompt: SystemMessage, max_tokens: int = 512, model: str = "openai/qwen3-30b-a3b",
                 temperature: float = 0.6, top_k: int = 20, top_p: float = 0.95, min_p: float = 0.01):
        self.llm = ChatLiteLLM(
            model=model,
            api_base="http://127.0.0.1:8080",
            api_key="SK-no-key",
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            model_kwargs={
                "min_p" : min_p,
                "repeat_penalty":1.17
            }
        )

        self.system_prompt = system_prompt

    def bind_tools(self, tools: list[Runnable] | None = None) -> None:
        """Bind tools. Wrapper for Runnable.bind_tools() method.

        Args:
            tools (list[Runnable] | None, optional): List of tools to bind.
        """
        if tools:
            self.llm = self.llm.bind_tools(tools=tools)

    @overload
    def __call__(self, prompt: HumanMessage) -> AIMessage: ...
    @overload
    def __call__(self, prompt: str) -> AIMessage: ...
    @overload
    def __call__(self, prompt: list[AnyMessage]) -> AIMessage: ...
    def __call__(self, prompt: str | HumanMessage | list[AnyMessage]) -> AIMessage:
        if isinstance(prompt, list):
            result = self.llm.invoke(
                prompt
            )
            return result

        if isinstance(prompt, str):
            prompt = HumanMessage(content=prompt)
        
        messages = [
            self.system_prompt,
            prompt
        ]

        return self.llm.invoke(messages)


if __name__ == "__main__":
    agent = LLMWrapper(system_prompt=SystemMessage(content="You are a KURO2_COMBAT_AI"))

    out = agent("What is your name?")
    print(out)
    # print(out.generations[0][0].text)
