from time import sleep
from typing import Annotated, cast
import inspect
import pyautogui as gui
import pydirectinput as gui2
from PIL import Image
from loguru import logger
from langchain_core.tools import tool, StructuredTool
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode, InjectedState
# from nodes.vlm_node import VLMNode
import nodes.vlm_node as vlm
from nodes.graph_state import CombatState


class Controller:
    """Controller
    """
    display_details: bool = False
    enemy_specifics: bool = False
    attack_option: bool = True
    timeout: float = 0.5
    timeout_menu: float = 1.0
    target_direction_f: bool = True # F - right, R - left

    def get_tools(self) -> list[StructuredTool]:
        """Get members marked as tools

        Returns:
            list[StructuredTool]: list of methods with @tool decorator
        """
        # TODO: Uncomment line below
        return [member[1] for member in inspect.getmembers_static(self) if isinstance(member[1], StructuredTool)]

    def screenshot(self, img_path: str) -> Image.Image:
        """Take screenshot"""
        return gui.screenshot(img_path)

    def reset_controller(self) -> None:
        """Reset controller after combat resolution."""
        self.display_details = False
        self.enemy_specifics = False
        self.attack_option = True
        self.target_direction_f = True

    def toggle_display_enemy_specifics(self) -> None:
        """Invoke 'View Specifics' command."""
        if not self.enemy_specifics:
            gui2.press(keys="tab")
            self.enemy_specifics = True
            sleep(self.timeout)

    def toggle_display_details(self) -> None:
        """Display details of attacks, arts, and crafts"""
        if not self.display_details:
            gui2.middleClick()
            self.display_details = True
            sleep(self.timeout)

    def change_target(self, target_direction_f: bool | None = None):
        """Change target.

        Args:
            right (bool, optional): Change to the target to the right. Default.
        """
        if not target_direction_f:
            target_direction_f = self.target_direction_f
        
        if target_direction_f:
            gui2.press(keys="f")
        else:
            gui2.press(keys="r")

    def select_target_enemy(self, enemy_id: int, enemy_coords: list[tuple[int, int]], target_direction: bool, path_to_screenshot: str, vlm_node: object) -> bool:
        """Target specific enemy.

        Args:
            enemy_id (int): enemy ID.
            enemy_coords (list[tuple[int, int]]): enemy portrait coordinates in turn-order region.
            target_direction (bool): True - F, False - R
            path_to_screenshot (str): path to screenshot
            vlm_node (vlm.VLMNode): VLMNode object

        Returns:
            bool: True if target has been found
        """
        vlm_node = cast(vlm.VLMNode, vlm_node)
        # TODO: Select target enemy
        logger.debug("TOOL: Attemp to select requested target")
        # while True:
        if len(enemy_coords) <= 1:
            return True

        for _ in range(2 * len(enemy_coords)):
            img_turn = gui.screenshot(path_to_screenshot)
            img_turn_bw = vlm_node.produce_bw_image(img_turn, vlm_node.origin, vlm_node.size)
            target_id, _ = vlm_node.find_selected_target(img_turn_bw, enemy_coords)
            logger.debug(f"TOOL: Current {target_id=}")
            
            # Attack?
            if target_id == enemy_id:
                logger.debug(f"TOOL: Found requested target! {target_id=} == {enemy_id=}")
                return True
            else:
                self.change_target(target_direction)
                logger.debug("TOOL: Next target")
        
        return False

    @tool
    @staticmethod
    def action_attack(enemy_id: int, state: Annotated[CombatState, InjectedState], config: RunnableConfig) -> None:
        """Basic attack.

        Args:
            enemy_id: Target enemy ID.
        """
        logger.debug(f"TOOL: Attempting basic attack on {enemy_id=}")
        controller = cast(Controller, config["configurable"]["tool_controller"])
        # Find expected target selection method
        for enemy in state["turn_state"]["enemies"]:
            if enemy.enemy_id == enemy_id:
                target_direction = enemy.target_method_f
       
        vlm_node = cast(vlm.VLMNode, config["configurable"]["tool_vlm"])
        enemy_coords = state["turn_state"]["enemy_coords"]

        if controller.select_target_enemy(enemy_id, enemy_coords, target_direction, config["configurable"]["path_to_screenshot"], vlm_node):
            # Assume 'Attack' option selected
            if not controller.attack_option:
                logger.debug("TOOL: Scroll up")
                gui2.scroll(1)
                sleep(controller.timeout)
            
            # Attack
            logger.debug("TOOL: Attacking target!")
            gui2.press(keys="enter")

    @tool
    @staticmethod
    def action_defend(config: RunnableConfig) -> None:
        """Decrease damage for this character until it's next turn instead of attacking."""
        logger.debug("TOOL: Defending.")
        controller = cast(Controller, config["configurable"]["tool_controller"])
        # Assume 'Attack' option selected
        if controller.attack_option:
            # Scroll down
            gui2.scroll(-1)
            sleep(controller.timeout)

        # Defend
        gui2.press(keys="enter")

    @tool
    @staticmethod
    def action_use_art(enemy_id: int, state: Annotated[CombatState, InjectedState], config: RunnableConfig) -> None:
        """Use art attack.

        Args:
            enemy_id: Target enemy ID.
        """
        controller = cast(Controller, config["configurable"]["tool_controller"])
        # Select target enemy
        # Find expected target selection method
        for enemy in state["turn_state"]["enemies"]:
            if enemy.enemy_id == enemy_id:
                target_direction = enemy.target_method_f
       
        vlm_node = cast(vlm.VLMNode, config["configurable"]["tool_vlm"])
        enemy_coords = state["turn_state"]["enemy_coords"]

        controller.select_target_enemy(enemy_id, enemy_coords, target_direction, config["configurable"]["path_to_screenshot"], vlm_node)
        # Find specific art
        # ! Use VLM to do so
        # TODO: Use specific art
        logger.debug("TOOL: Using art")
        gui2.press(keys="q")
        sleep(controller.timeout_menu)
        gui2.press(keys="enter")

    @tool
    @staticmethod
    def action_use_craft(enemy_id: int, state: Annotated[CombatState, InjectedState], config: RunnableConfig) -> None:
        """Use craft attack.

        Args:
            enemy_id: Target enemy ID.
        """
        controller = cast(Controller, config["configurable"]["tool_controller"])
        # Select target enemy
        # Find expected target selection method
        for enemy in state["turn_state"]["enemies"]:
            if enemy.enemy_id == enemy_id:
                target_direction = enemy.target_method_f
       
        vlm_node = cast(vlm.VLMNode, config["configurable"]["tool_vlm"])
        enemy_coords = state["turn_state"]["enemy_coords"]

        controller.select_target_enemy(enemy_id, enemy_coords, target_direction, config["configurable"]["path_to_screenshot"], vlm_node)
        # Find specific craft
        # ! Use VLM to do so
        # TODO: Use specific craft
        logger.debug("TOOL: Using craft")
        gui2.press(keys="e")
        sleep(controller.timeout_menu)
        gui2.press(keys="enter")
    
    @tool
    @staticmethod
    def action_use_item(config: RunnableConfig) -> None:
        """Use item from the inventory."""
        controller = cast(Controller, config["configurable"]["tool_controller"])
        # Open Items menu
        gui2.press(keys='x')
        sleep(controller.timeout)
        # Find specific item
        # ! Use VLM to do so
        # TODO: Use specific item
        gui2.press(keys="enter")

    def action_open_character_screen(self) -> None:
        """Opens character screen from out-of-combat situation."""
        gui2.press(keys="esc")
        sleep(self.timeout_menu)
        gui2.press(keys="z")
        sleep(self.timeout_menu)

    def action_close_character_screen(self) -> None:
        """Close character screen from out-ouf-combat situation."""
        gui2.press(keys="esc")
        sleep(self.timeout_menu)
        gui2.press(keys="esc")
        sleep(self.timeout_menu)

    def action_next_menu_item(self) -> None:
        """Selects next item on the menu list."""
        gui2.press(keys="down")
        sleep(self.timeout_menu)
