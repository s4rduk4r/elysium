from time import sleep
from typing import Any
import torch
from loguru import logger
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import RemoveMessage
from langgraph.errors import GraphRecursionError
from pyautogui import screenshot, hotkey

from nodes.graph_state import CombatState, AgentConfig
from nodes.vlm_node import VLMNode
from nodes.llm_node import LLMNode
from nodes.get_player_strengths import GetPlayerStrengthsNode
from tools.controller import Controller


# AGENT
class Kuro2CombatAgent:
    """Kuro 2 turn-based combat agent."""
    reasoner_system_prompt: str = ""

    def __init__(self):
        self.controller = Controller()
        self.stat_updater = VLMNode(self.controller)
        self.reasoner = LLMNode(self.controller)
        self.get_player_strengths = GetPlayerStrengthsNode(self.stat_updater.vlm, self.controller)

        logger.warning("Building graph")
        self.graph = StateGraph(state_schema=CombatState, config_schema=AgentConfig)

        # Entry point
        # self.graph.add_edge(START, "wait_for_agent_turn")
        # TODO: START AGENT OUTSIDE OF COMBAT
        self.graph.add_edge(START, "get_player_strengths")
        self.graph.add_sequence(
            [
                ("get_player_strengths", self.get_player_strengths),
                ("wait_for_agent_turn", self._wait_for_agent_turn),
                ("stat_updater", self.stat_updater),
                ("reasoner", self.reasoner),
                ("reasoner_command", ToolNode(tools=self.controller.get_tools())), # TODO: Uncomment
                # ("reasoner_command", ToolNode(tools=[self.controller.action_attack])),
                ("forget_turn_reasoning", self._forget_turn_reasoning)
            ]
        )
        self.graph.add_conditional_edges("forget_turn_reasoning", self._is_combat_over, [END, "wait_for_agent_turn"])
        self.graph = self.graph.compile()
        self.graph.get_graph().draw_png("agent_graph.png")
        logger.warning("Graph has been built")

    def _is_combat_over(self, state: CombatState, config: Any) -> str:
        """Checks if combat is over. Does not modify agent's state.

        Returns:
            "wait_for_agent_turn" if combat is going on.
            END if combat is over.
        """
        logger.debug("---CHECK IF COMBAT IS OVER---")
        # Seed = 1643; temperature = 0.7; max_tokens = 512
        seed = config["configurable"]["seed_vlm"]
        torch.manual_seed(seed)
        path_to_screenshot = config["configurable"]["path_to_screenshot"]

        # 1. If word "RESULTS" on screen, then it's a win - combat over
        prompt_test_win = """Is there word "RESULTS" on screen? Give only binary answer - yes or no."""
        test_win = self.stat_updater.vlm(text=prompt_test_win, image=path_to_screenshot)[0].replace(".", "").lower()
        if test_win == "yes":
            return END
        
        # 2. If menu with word "Retry" is on screen, then it's a loss - combat over
        prompt_test_loss = """Is there menu with item "Retry" on screen? Give only binary answer - yes or no."""
        test_loss = self.stat_updater.vlm(text=prompt_test_loss, image=path_to_screenshot)[0].replace(".", "").lower()
        if test_loss == "yes":
            return END

        # 3. Check if ATB-scale is on top of the screen
        prompt_test_atb = """Is character's turn order at the top of the screen is present? Ignore character portraits on the left. Give only binary answer - yes or no."""
        test_atb_result = self.stat_updater.vlm(text=prompt_test_atb, image=path_to_screenshot)[0].replace(".", "").lower()
        test_atb_result = test_atb_result == "yes"
        # 4. Check if HP, EP, CP of characters are on screen
        # More reliable method
        prompt_test_hpepcp = """Are character's HP,EP,CP values on screen? Ignore character portraits on top. Give only binary answer - yes or no."""
        test_hpepcp = self.stat_updater.vlm(text=prompt_test_hpepcp, image=path_to_screenshot)[0].replace(".", "").lower()
        test_hpepcp = test_hpepcp == "yes"
        # 5. If at least one of these criteria isn't met, then it's out of combat situation
        if (test_atb_result or test_hpepcp):
            return "wait_for_agent_turn"

        # Default route
        return "get_player_strengths"
    
    def _wait_for_agent_turn(self, state: CombatState, config: AgentConfig) -> CombatState:
        """Wait until it is agent's (player) turn. Does not modify agent's state.

        Returns:
            "stat_updater"
        """
        logger.debug("---WAIT FOR AGENT TURN---")
        seed = config["configurable"]["seed_vlm"]
        path_to_screenshot = config["configurable"]["path_to_screenshot"]
        timeout_sec = config["configurable"]["timeout_screenshot_sec"]

        while True: # TODO: Uncomment when it's ready for game
        # for i in range(5): # TODO: DEBUG: Imitate actual work
            # 1. Check if UI controls are on screen
            screenshot(path_to_screenshot) # TODO: Uncomment when it's ready for game
            torch.manual_seed(seed)
            prompt_test_ui_enabled = """Are there menu items "Attack" and "Defend" on screen? Give only binary answer - yes or no."""
            test_ui_enabled = self.stat_updater.vlm(text=prompt_test_ui_enabled, image=path_to_screenshot)[0].replace(".", "").lower()
            # 2. If UI controls are present, then it's agent's turn
            if test_ui_enabled == "yes":
                return state
            
            # 3. Otherwise, wait
            sleep(timeout_sec)
        
        return state
    
    def _forget_turn_reasoning(self, state: CombatState) -> CombatState:
        logger.debug("---FORGET TURN REASONING---")
        return CombatState(
            messages = [RemoveMessage(x.id) for x in state["messages"]],
            turn_state=state["turn_state"]
        )


if __name__ == '__main__':
    import os
    if os.name == "nt":
        os.environ['PATH'] += ";C:\\Program Files\\Graphviz\\bin"
    
    logger.add("logs//ttd2_combat_agent_{time}.log")

    agent = Kuro2CombatAgent()
    
    try:
        # TODO: Timeout before game has started
        # How to load:
        # 1. Start agent
        # 2. Start game
        import subprocess

        logger.warning("---STARTING GAME---")
        KURO2_START_CMD = "cmd /c start steam://rungameid/2668430"
        TIMEOUT_BEFORE_START = 40
        subprocess.run(KURO2_START_CMD, check=False)
        logger.info(f"STARTING GAME: Wait {TIMEOUT_BEFORE_START} seconds")
        sleep(TIMEOUT_BEFORE_START)

        # 3. Start llama.cpp for reasoner
        logger.warning("---STARTING LLAMA.CPP---")
        LLAMA_CPP_SERVER_PATH = r"cmd /c start F:\llama.cpp\b5425-vk\llama-server.exe"
        # LLAMA_CPP_SERVER_ARGS = r'-m F:\models\Qwen3-30B-A3B-GGUF\Qwen3-30B-A3B-Q4_1.gguf -ot ".ffn_.*_exps.=CPU" -ngl 49 -t 16 --ctx-size 10574 --temp 0.6 --top-k 20 --top-p 0.95 --alias openai/qwen3-30b-a3b --host 127.0.0.1 --jinja --reasoning-format deepseek -sm row --no-context-shift'
        LLAMA_CPP_SERVER_ARGS = r'-m F:\models\Qwen3-30B-A3B-GGUF\Qwen3-30B-A3B-Q4_1.gguf -ngl 49 -t 16 --ctx-size 10574 --temp 0.6 --top-k 20 --top-p 0.95 --alias openai/qwen3-30b-a3b --host 127.0.0.1 --jinja --reasoning-format deepseek -sm row --no-context-shift'
        REASONER_START_CMD = f"{LLAMA_CPP_SERVER_PATH} {LLAMA_CPP_SERVER_ARGS}"
        # subprocess.run(REASONER_START_CMD, check=False)
        subprocess.Popen(REASONER_START_CMD)
        # Switch back to game screen
        hotkey(["alt", "tab"], interval=0.1)
        logger.info(f"STARTING LLAMA.CPP: Wait {TIMEOUT_BEFORE_START} seconds")
        sleep(TIMEOUT_BEFORE_START)

        # 3. Wait a bit for the agent to profile active first characters
        # 4. Engage some enemies
        # 5. Observe agent's behaviour
        agent.graph.invoke(CombatState(), 
                           {
                               "recursion_limit": 5 * 20, 
                               "configurable": {
                                    "seed_vlm": 1643,
                                    "seed_vlm_node": 1741,
                                    # "path_to_screenshot": "imgs\\combat-pc-turn1-3-enemy_stats.png",
                                    # "path_to_screenshot": "imgs\\combat-pc-turn1-2.png",
                                    "path_to_screenshot": r"imgs/screenshot.png",
                                    "timeout_screenshot_sec": 2.5,
                                    "game_turn_order_region_origin": (660, 45),
                                    "game_turn_order_region_crop_size": (900, 140),
                                    "game_turn_order_enemy_shade_rgb": (219, 0, 72),
                                    "game_turn_order_enemy_max_distance": 10,
                                    "enemy_stat_hp_origin": (1566,284),
                                    "enemy_stat_hp_size": (200,28),
                                    "enemy_stat_stun_origin": (1566, 320),
                                    "enemy_stat_stun_size": (60,30),
                                    "enemy_stat_atk_ats_speed_origin": (1566,382),
                                    "enemy_stat_atk_ats_speed_size": (50,102),
                                    "enemy_stat_def_adf_origin": (1704,382),
                                    "enemy_stat_def_adf_size": (50,66),
                                    "enemy_stat_weakness_basic_origin": (1530, 522),
                                    "enemy_stat_weakness_basic_size": (50, 140),
                                    "enemy_stat_weakness_higher_elements_origin": (1715, 522),
                                    "enemy_stat_weakness_higher_elements_size": (50, 104),
                                    "enemy_stat_ailments_left_origin": (1556, 700),
                                    "enemy_stat_ailments_left_size": (40, 174),
                                    "enemy_stat_ailments_right_origin": (1750, 700),
                                    "enemy_stat_ailments_right_size": (40, 174),
                                    "enemy_target_detect_origin_delta": (20, 0),
                                    "enemy_target_detect_size": 10,
                                    "character_strength_origin": [(90, 108), (90, 208), (90, 308), (90, 408)], # 1st, 2nd, 3rd, 4th
                                    "character_strength_size": (240, 50),
                                    "character_active_strength_origin": [(160, 108), (160, 208), (160, 308), (160, 408)], # 1st, 2nd, 3rd, 4th
                                    "character_active_strength_size":  (240, 50),
                                    "character_atk_def_origin": (1280, 310),
                                    "character_atk_def_size": (250, 64),
                                    "character_ats_adf_origin": (1284, 370),
                                    "character_ats_adf_size": (250, 64),
                                    "character_speed_origin": (1668,364),
                                    "character_speed_size": (38,30),
                                    "reasoner_model_name": "openai/qwen3-30b-a3b",
                                    "reasoner_system_prompt": """You are playing a turn-based tactics game, where you have to control separate units in order to gain the upper hand. Each unit can move and attack in the same turn. Use of item is considered as attack action.
Attacks can be basic, craft, art. Basic attack doesn't consume CP or EP and is executed in an instant. Crafts consume CP and are executed in an instant, yield high damage, buff allies, or debuff enemies. Most crafts allow to hit multiple enemies at once while act as a damage multiplier. Crafts can't be canceled by enemy attacks. Treat all crafts as physical damage only.
CP pool is restored by attacking enemies with basic attacks and arts, and receiving damage. EP pool can be restored only through the item use.
Arts consume EP but have delayed execution before yielding high damage, buff allies, or debuff enemies. This delay means that arts may be casted after other characters', and enemies' turn. Some enemies' attacks are able to cancel casting art.
Units with high physical strength usually excel at using crafts, while units with high maximum EP and arts damage are better suited for using arts. 
Basic attacks are available to everyone. You must give the order to active character based of the current turn context.""",
                                    "reasoner_nothink_prompt": True,#False, # TODO: Set to True to speed up the process of decision making. May end up in an endless loop
                                    "debug_reasoner_off" : False, # TODO: Set to False when ready to deploy
                                    "tool_controller" : agent.controller,
                                    "tool_vlm": agent.stat_updater
                               }
                            }
                            )
    except GraphRecursionError:
        logger.critical("20-turn limit has been reached. Agent terminated")
