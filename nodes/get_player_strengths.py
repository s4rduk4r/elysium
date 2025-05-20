import torch
from loguru import logger
from tools.controller import Controller
from nodes.vlm_wrapper import VLMWrapper
from nodes.vlm_node import VLMNode
from nodes.graph_state import CombatState, AgentConfig, TurnState
from state.player_stat import PlayerCharacterStat


class GetPlayerStrengthsNode:
    """Get player strengths node. Query first four characters before the combat happens, then just quickly passes control to next node.
    
    Relies on VLM, and Controller.
    """
    def __init__(self, vlm: VLMWrapper, controller: Controller):
        self.vlm = vlm
        self.controller = controller
        # Initialize four default PlayerCharacterStat objects
        self.player_characters = [
            PlayerCharacterStat(character_id=0),
            PlayerCharacterStat(character_id=1),
            PlayerCharacterStat(character_id=2),
            PlayerCharacterStat(character_id=3)
        ]
        # If set, then this node tries to update character strengths
        self.need_to_update = True
    
    def __call__(self, state: CombatState, config: AgentConfig) -> CombatState:
        # Pass controls to other node
        if not self.need_to_update:
            logger.debug("CHARACTER STRENGTHS: No need to update")
            return state
        
        # Get config
        path_to_screenshot = config["configurable"]["path_to_screenshot"]
        seed = config["configurable"]["seed_vlm_node"]
        character_atk_def_origin = config["configurable"]["character_atk_def_origin"]
        character_atk_def_size = config["configurable"]["character_atk_def_size"]
        character_ats_adf_origin = config["configurable"]["character_ats_adf_origin"]
        character_ats_adf_size = config["configurable"]["character_ats_adf_size"]
        character_speed_origin = config["configurable"]["character_speed_origin"]
        character_speed_size = config["configurable"]["character_speed_size"]

        logger.debug("---CHARACTER STRENGTHS: GETTING PLAYER CHARACTERS STRENGTHS---")
        # 0. Assume agent is out of combat
        # 1. Go to system menu
        # 2. Go to character screen
        self.controller.action_open_character_screen()
        # 3. Extract basic parameters
        for idx, _ in enumerate(self.player_characters):
            img_character_screen = self.controller.screenshot(path_to_screenshot)
            # 3.1. Attack/Def - Origin: 1280, 310. Size (W,H): 250,64
            origin = character_atk_def_origin #(1280, 310)
            size = character_atk_def_size #(250, 64)
            img_atk_def = img_character_screen.crop(VLMNode.get_crop_box(origin, size))
            text_prompt_atk_def = """First value is Strength, and second value is Defense. What are these values? Be very concise.""" # ok
            # seed = 1741
            torch.manual_seed(seed)
            result = self.vlm(text=text_prompt_atk_def, image=img_atk_def)[0]
            # Extract values
            params = {}
            for line in result.lower().splitlines():
                if "str" in line:
                    params["str"] = int(line.split(":")[-1])
                if "def" in line:
                    params["def"] = int(line.split(":")[-1])

            # 3.2. Arts attack/Arts def - Origin: 1284, 370. Size (W,H): 250,64
            origin = character_ats_adf_origin #(1284, 370)
            size = character_ats_adf_size #(250, 64)
            img_ats_adf = img_character_screen.crop(VLMNode.get_crop_box(origin, size))
            text_prompt_ats_adf = """First value is Arts Strength, and second value is Arts Defense. What are these values? Be very concise.""" # ok
            # seed = 1741
            torch.manual_seed(seed)
            result = self.vlm(text=text_prompt_ats_adf, image=img_ats_adf)[0]
            for line in result.lower().splitlines():
                if "str" in line:
                    params["ats"] = int(line.split(":")[-1])
                if "def" in line:
                    params["adf"] = int(line.split(":")[-1])

            # 3.3. Speed - Origin: 1668,364. Size (W,H): 38,30
            origin = character_speed_origin #(1668,364)
            size = character_speed_size #(38,30)
            img_speed = img_character_screen.crop(VLMNode.get_crop_box(origin, size))
            text_prompt_speed = """What number is it? Just give the number.""" # ok
            # seed = 1741
            torch.manual_seed(seed)
            result = self.vlm(text=text_prompt_speed, image=img_speed)[0]
            params["speed"] = int(result)

            # Update stats
            self.player_characters[idx].attack = params["str"]
            self.player_characters[idx].defense = params["def"]
            self.player_characters[idx].arts_attack = params["ats"]
            self.player_characters[idx].arts_defense = params["adf"]
            self.player_characters[idx].speed = params["speed"]

            # 4. Select next character
            self.controller.action_next_menu_item()

        self.controller.action_close_character_screen()
        self.need_to_update = False

        # Update agent's state to reflect garnered info
        turn_state = TurnState(
            enemies=None,
            player_characters=self.player_characters
        )

        return CombatState(
            messages=state["messages"],
            turn_state=turn_state
        )
