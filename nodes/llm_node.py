from loguru import logger
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from litellm.exceptions import APIError

from tools.controller import Controller
from nodes.llm_wrapper import LLMWrapper
from nodes.graph_state import CombatState, AgentConfig, TurnState


class LLMNode:
    """LLM Node responsible for thinking about the next action to take.
    """
    system_prompt: SystemMessage
    max_tokens: int = 4096

    def __init__(self, controller: Controller):
        self.llm = None
        self.controller = controller
        self.disabled = True
        self.nothink_prompt = False

    def __call__(self, state: CombatState, config: AgentConfig) -> str:
        self.disabled = config["configurable"]["debug_reasoner_off"]
        if self.disabled:
            logger.warning("REASONER IS OFF")
            return {
            "messages": [
                SystemMessage("REASONER IS OFF"),
                HumanMessage("REASONER IS OFF"),
                AIMessage("REASONER IS OFF")
            ],
            "turn_state": state["turn_state"]
        }

        # Lazy instantiation
        if not self.llm:
            self.nothink_prompt = config["configurable"]["reasoner_nothink_prompt"]
            logger.warning("Starting LLM Node")
            self.system_prompt = SystemMessage(content=config["configurable"]["reasoner_system_prompt"])
            self.llm = LLMWrapper(system_prompt=self.system_prompt, max_tokens=self.max_tokens, model=config["configurable"]["reasoner_model_name"])
            try:
                self.llm(r"Hi/no_think")
                self.llm.bind_tools(self.controller.get_tools())
                # self.llm.bind_tools([self.controller.action_attack]) # TODO: Debug set of tools
                logger.warning("LLM Node is ready")
            except APIError as err:
                logger.critical(err)
                exit(-1)

        logger.debug("---REASONER IS PLANNING---")
        # 1. Convert context from the current turn's TurnState object
        turn_context = TurnState.to_prompt(state["turn_state"])
        logger.info(f"Turn context:\n{turn_context}")
        
        # 2. Conflate system message, context, and prompt
        reasoner_prompt = f"{turn_context}\n\nIt is your turn now. Give order to the active character."
        if self.nothink_prompt:
            reasoner_prompt += "/no_think"
        reasoner_prompt = HumanMessage(content=reasoner_prompt)
        result = self.llm(reasoner_prompt)
        logger.info(f"Reasoner decision:\n{result}")
        if len(result.tool_calls) == 0:
            logger.debug("LLM Node: No tool calling. Re-run with enabled reasoning")
            result = self.llm(reasoner_prompt.replace("/no_think", ""))

        return {
            "messages": [
                self.system_prompt,
                reasoner_prompt,
                result
            ],
            "turn_state": state["turn_state"]
        }

