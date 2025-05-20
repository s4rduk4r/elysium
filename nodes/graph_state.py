from typing_extensions import TypedDict
from langgraph.graph import MessagesState

from state.enemy_stat import EnemyStat
from state.player_stat import PlayerCharacterStat


class TurnState(TypedDict):
    """State of the current player turn for Reasoner."""
    enemies: list[EnemyStat]
    enemy_coords: list[tuple[int,int]] | None  # Coordinates of enemy portraits in turn-order region
    player_characters: list[PlayerCharacterStat]

    @staticmethod
    def to_prompt(turn_state: object) -> str:
        """Get string for LLM reasoner context."""
        # Enemies
        enemy_str_context = [enemy.to_prompt() for enemy in turn_state["enemies"]]
        enemy_str_context = "\n".join(enemy_str_context)

        # Characters
        character_str_context = [character.to_prompt() for character in turn_state["player_characters"]]
        character_str_context = "\n".join(character_str_context)

        # Active character
        for character in turn_state["player_characters"]:
            if character.is_active:
                active_character_str_context = f"You can give orders to Character {character.character_id}."

        return f"# CHARACTERS STATS\n{character_str_context}\n\n{active_character_str_context}\n\n# ENEMIES STATS\n{enemy_str_context}"



class CombatState(MessagesState):
    """Combat log."""
    # `messages` are growing by adding returned messages from agent
    turn_state: TurnState | None = None # must be specified explicitly, otherwise it won't be updated


class AgentConfig(TypedDict):
    """Agent config.
        seed_vlm - seed for agentic use of VLM outside of VLMNode
        seed_vlm_node - seed for VLM to use inside of VLMNode
        path_to_screenshot - path to where the screenshot is stored
        timeout_screenshot_sec - timeout between taking two screenshots
    """
    seed_vlm: int = 1643
    seed_vlm_node: int = 1741
    path_to_screenshot: str = "imgs\\screenshot.png"
    timeout_screenshot_sec: float = 2.5
    # Settings to process screenshots with VLM Node. All values are in pixels. Default resolution: 1920x1080
    game_turn_order_region_origin: tuple[int, int] = (660, 45)
    game_turn_order_region_crop_size: tuple[int, int] = (900, 140)
    game_turn_order_enemy_shade_rgb: tuple[int, int, int] = (219, 0, 72)
    game_turn_order_enemy_max_distance: int = 10
    enemy_stat_hp_origin: tuple[int,int] = (1566,284)
    enemy_stat_hp_size: tuple[int,int] = (200,28)
    enemy_stat_stun_origin: tuple[int,int] = (1566, 320)
    enemy_stat_stun_size: tuple[int,int] = (60,30)
    enemy_stat_atk_ats_speed_origin: tuple[int,int] = (1566,382)
    enemy_stat_atk_ats_speed_size: tuple[int,int] = (50,102)
    enemy_stat_def_adf_origin: tuple[int,int] = (1704,382)
    enemy_stat_def_adf_size: tuple[int,int] = (50,66)
    enemy_stat_weakness_basic_origin: tuple[int, int] = (1530, 522)
    enemy_stat_weakness_basic_size: tuple[int, int] = (50, 140)
    enemy_stat_weakness_higher_elements_origin: tuple[int, int] = (1715, 522)
    enemy_stat_weakness_higher_elements_size: tuple[int, int] = (50, 104)
    enemy_stat_ailments_left_origin: tuple[int, int] = (1556, 700)
    enemy_stat_ailments_left_size: tuple[int, int] = (40, 174)
    enemy_stat_ailments_right_origin: tuple[int, int] = (1750, 700)
    enemy_stat_ailments_right_size: tuple[int, int] = (40, 174)
    enemy_target_detect_origin_delta: tuple[int, int] = (20, 0)
    enemy_target_detect_size: int = 10
    character_strength_origin: list[tuple[int, int]] = [(90, 108), (90, 208), (90, 308), (90, 408)] # 1st, 2nd, 3rd, 4th
    character_strength_size: list[tuple[int, int]] = (240, 50)
    character_active_strength_origin: list[tuple[int, int]] = [(160, 108), (160, 208), (160, 308), (160, 408)] # 1st, 2nd, 3rd, 4th
    character_active_strength_size: tuple[int, int] = (240, 50)
    character_atk_def_origin: tuple[int,int] = (1280, 310)
    character_atk_def_size: tuple[int,int] = (250, 64)
    character_ats_adf_origin: tuple[int,int] = (1284, 370)
    character_ats_adf_size: tuple[int,int] = (250, 64)
    character_speed_origin: tuple[int,int] = (1668,364)
    character_speed_size: tuple[int,int] = (38,30)
    reasoner_model_name: str = "openai/qwen3-30b-a3b"
    reasoner_system_prompt: str = ""
    reasoner_nothink_prompt: bool = False # True - turn off reasoning
    debug_reasoner_off: bool = False
