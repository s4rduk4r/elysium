from json import loads
import torch
from loguru import logger
from PIL import Image
import numpy as np


from nodes.graph_state import CombatState, TurnState, AgentConfig
from nodes.vlm_wrapper import VLMWrapper
from state.enemy_stat import EnemyStat
from state.player_stat import PlayerCharacterStat
from tools.controller import Controller


class VLMNode:
    """VLM Node responsible for context extraction for LLM Node."""
    def __init__(self, controller: Controller):
        logger.warning("Starting VLM")
        self.vlm = VLMWrapper()
        self.controller = controller
        logger.warning("VLM Node is ready")

    def __call__(self, state: CombatState, config: AgentConfig) -> CombatState:
        logger.debug("---UPDATING COMBAT STATS---")
        # TODO: Implement VLM
        # Get config
        self.path_to_screenshot = config["configurable"]["path_to_screenshot"]
        self.seed = config["configurable"]["seed_vlm_node"]
        self.origin = config["configurable"]["game_turn_order_region_origin"]
        self.size = config["configurable"]["game_turn_order_region_crop_size"]
        self.enemy_color = config["configurable"]["game_turn_order_enemy_shade_rgb"]
        self.enemy_stat_hp_origin = config["configurable"]["enemy_stat_hp_origin"]
        self.enemy_stat_hp_size = config["configurable"]["enemy_stat_hp_size"]
        self.enemy_stat_stun_origin = config["configurable"]["enemy_stat_stun_origin"]
        self.enemy_stat_stun_size = config["configurable"]["enemy_stat_stun_size"]
        self.enemy_stat_atk_ats_speed_origin = config["configurable"]["enemy_stat_atk_ats_speed_origin"]
        self.enemy_stat_atk_ats_speed_size = config["configurable"]["enemy_stat_atk_ats_speed_size"]
        self.enemy_stat_def_adf_origin = config["configurable"]["enemy_stat_def_adf_origin"]
        self.enemy_stat_def_adf_size = config["configurable"]["enemy_stat_def_adf_size"]
        self.enemy_stat_weakness_basic_origin = config["configurable"]["enemy_stat_weakness_basic_origin"]
        self.enemy_stat_weakness_basic_size = config["configurable"]["enemy_stat_weakness_basic_size"]
        self.enemy_stat_weakness_higher_elements_origin = config["configurable"]["enemy_stat_weakness_higher_elements_origin"]
        self.enemy_stat_weakness_higher_elements_size = config["configurable"]["enemy_stat_weakness_higher_elements_size"]
        self.enemy_stat_ailments_left_origin = config["configurable"]["enemy_stat_ailments_left_origin"]
        self.enemy_stat_ailments_left_size = config["configurable"]["enemy_stat_ailments_left_size"]
        self.enemy_stat_ailments_right_origin = config["configurable"]["enemy_stat_ailments_right_origin"]
        self.enemy_stat_ailments_right_size = config["configurable"]["enemy_stat_ailments_right_size"]
        self.enemy_target_detect_origin_delta = config["configurable"]["enemy_target_detect_origin_delta"]
        self.enemy_target_detect_size = config["configurable"]["enemy_target_detect_size"]
        self.character_strength_origin = config["configurable"]["character_strength_origin"]
        self.character_strength_size = config["configurable"]["character_strength_size"]
        self.character_active_strength_origin = config["configurable"]["character_active_strength_origin"]
        self.character_active_strength_size = config["configurable"]["character_active_strength_size"]

        # Get screenshot of current player turn
        img_player_turn = Image.open(self.path_to_screenshot)
        # Get enemies' profiles
        n_enemies, enemy_coords = self._estimate_number_of_enemies(img_player_turn, self.origin, self.size, self.enemy_color)
        enemy_stat = self._get_enemy_strength(img_player_turn, n_enemies, enemy_coords, self.seed)
        # Update player characters' profiles
        pc_stat = self._update_active_characters_strengths(img_player_turn, state["turn_state"]["player_characters"], self.seed)
        
        new_state = CombatState(
            messages=state["messages"],
            turn_state=TurnState(
                enemies=enemy_stat,
                player_characters=pc_stat,
                enemy_coords=enemy_coords
            )
        )

        return new_state
    
    @staticmethod
    def get_crop_box(origin: tuple[int, int], size: tuple[int, int]) -> tuple[int, int, int, int]:
        """Return crop box suitable for use with PIL.Image.crop() method.

        Args:
            origin (tuple[int, int]): origin point in pixels (left, top)
            size (tuple[int, int]): size of cropping area in pixels (width, height)

        Returns:
            tuple[int, int, int, int]: left, top, right, bottom
        """
        return origin + (origin[0] + size[0], origin[1] + size[1])

    def _estimate_number_of_enemies(self, img_player_turn: Image.Image, origin: tuple[int, int], size: tuple[int, int], 
                                    enemy_color: tuple[int, int, int] = (219, 0, 72)) -> tuple[int, list[tuple[int, int]]]:
        """Estimate number of enemies engaged in combat.

        Args:
            origin (tuple[int, int]): Origin coordinates to crop turn-order area
            size (tuple[int, int]): Size of turn-order area

        Returns:
            int: number of enemies
            list[tuple[int, int]]: enemy portrait coordinates on turn order image region
        """
        logger.debug("STAT_UPDATER: Counting enemies")
        # 1. Get screenshot of current turn
        # img_player_turn = Image.open(self.path_to_screenshot)
        # 2. Crop turn order from it. Origin: 660, 45. Size (W, H): 900, 140
        crop_area = self.get_crop_box(origin, size)
        img_turn_order = img_player_turn.crop(crop_area)
        # 3. Look for all occurences of the specific pixel
        pixel_needle = enemy_color# (219, 0, 72) # Color of the tip of red-shaped left chevron on enemy portrait
        img_haystack = img_turn_order

        # 3.1. Get all unique `x` coordinates for specific color
        enemy_positions_x = []
        enemy_positions = []
        for x in range(img_haystack.size[0]):
            for y in range(img_haystack.size[1]):
                pixel = img_haystack.getpixel((x, y))
                if pixel == pixel_needle:
                    enemy_positions_x.append(x)
                    enemy_positions.append((x, y))

        enemy_positions_x = np.unique_values(enemy_positions_x).tolist()

        # 3.2. Filter every `x` coordinate that is too far from the origin point
        origins_x = [enemy_positions_x[0]]
        max_distance = 10
        for x in enemy_positions_x:
            if x - origins_x[-1] >= max_distance:
                origins_x.append(x)

        # Gather enemy coordinates
        enemy_coords = {}
        for x in enemy_positions:
            if x[0] in origins_x:
                if not enemy_coords.get(x[0]):
                    enemy_coords[x[0]] = x

        enemy_coords = list(enemy_coords.values())

        logger.info(f"STAT UPDATER: Enemies number: {len(origins_x)}")
        return (len(enemy_coords), enemy_coords)
        # return (len(origins_x), origins_x)
    
    def _get_enemy_strength(self, img_player_turn: Image.Image, n_enemies: int, enemy_coords: list[tuple[int,int]], seed: int = 1741) -> list[EnemyStat]:
        """Iterate through found enemies, and get their stats.
            Returns:
                list[EnemyStat]: list of enemy stats
        """
        logger.debug("STAT_UPDATER: Getting enemies' strengths and weaknesses")
        # 1. Activate "View Specifics"
        self.controller.toggle_display_enemy_specifics() # TODO: Uncomment when ready to use on actual gameplay
        self.controller.target_direction_f = True
        profiled_enemies = []
        profiled_enemies_indices = []

        for i in range(2 * n_enemies + 1): # Limit max target iterations
            # Update screenshot
            img_player_turn = self.controller.screenshot(self.path_to_screenshot)
            # Check if this enemy has already been profiled
            # 1 2 3 4
            # first_enemy = 3
            # F -> (1,3,4)
            # R -> (2, 1, 3)
            # profiled_enemies = [3,4,1] # F->profile->F->profile->F->already_profiled->R until it's 2 ->profile->END

            # 2. Find which enemy is selected - it's the 1st
            #   2.1. Produce black-and-white image out of red channel from HSV-representation of the image
            img_turn_order_bw = self.produce_bw_image(img_player_turn, self.origin, self.size)
            enemy_id, _ = self.find_selected_target(img_turn_order_bw, enemy_coords, self.enemy_target_detect_origin_delta, self.enemy_target_detect_size)

            # TODO: Fix algorithm to find all enemies - including ones not detected by specific pixel color

            # 2.3. If this enemy has already been profiled, then attempt to select next
            if enemy_id in profiled_enemies_indices:
                # 2.4. Test if all detected enemies has been profiled
                if np.isin(range(n_enemies), profiled_enemies_indices).all() and self.controller.target_direction_f:
                    self.controller.target_direction_f = False
                elif np.isin(range(n_enemies), profiled_enemies_indices).all() and not self.controller.target_direction_f:
                    break

                # 2.5. Iterate through targets to the right
                self.controller.change_target() # TODO: Uncomment when it's ready to deploy
                continue

            profiled_enemies_indices.append(enemy_id)

            # 2.6. Profile enemy
            profiled_enemies.append(self.__profile_selected_enemy(img_player_turn, seed, enemy_id, self.controller.target_direction_f))

            self.controller.change_target()

        logger.debug(f"Enemies profiled: {profiled_enemies_indices=}")
        return profiled_enemies

    def __profile_selected_enemy(self, img_player_turn: Image.Image, seed: int, enemy_id: int, target_direction_f: bool = True) -> EnemyStat:
        """Profile selected enemy

        Args:
            img_player_turn (Image.Image): screenshot of current player turn
            seed (int): generation seed

        Returns:
            EnemyStat: enemy stat object
        """
        # 1. Get enemy parameters
        enemy_params = self.__extract_enemy_parameters(img_player_turn, seed)
        # 2. Get enemy weaknesses
        enemy_weakness = self.__extract_enemy_weaknesses(img_player_turn, seed)
        # 3. Get enemy ailments
        enemy_ailments = self.__extract_enemy_ailments(img_player_turn, seed)
        # 4. Check if it can be attacked by basic attack
        enemy_can_be_attacked = self.__is_enemy_within_reach(img_player_turn, seed)
        return EnemyStat.from_dicts(enemy_id, target_direction_f, enemy_can_be_attacked, enemy_params, enemy_weakness, enemy_ailments)

    def __is_enemy_within_reach(self, img_player_turn: Image.Image, seed: int) -> bool:
        """Check if selected enemy can be attacked with basic attack.

        Args:
            img_player_turn (Image.Image): screenshot of current player turn
            seed (int): generation seed

        Returns:
            bool: True - can be attacked
        """
        text_prompt = """Is there a big red X on screen? Give only yes or no answer."""
        torch.manual_seed(seed)
        return False if "yes" in self.vlm(text=text_prompt, image=img_player_turn)[0].lower() else True

    @staticmethod
    def find_selected_target(img_turn_order_bw: Image.Image, enemy_coords: list[tuple[int,int]], origin_delta: tuple[int,int] = (20, 0), size:int = 10, significance: float = 0.5) -> tuple[int,float]:
        """Get index of enemy coordinates that correspond to currently selected enemy

        Args:
            img_turn_order_bw (Image.Image): image with turn order
            enemy_coords (list[tuple[int,int]]): list of enemy (x,y) coordinates

        Returns:
            tuple[int,float]: Index of enemy_coords, level of confidence
        """
        # TODO: Make it robust. It skips on profiles of some enemies
        def is_white(bw_color: int) -> bool:
            return bw_color == 255

        # Get 10x10 region with coordinates (20, 0) relative to the selected portrait
        proportions = []
        for idx, pos in enumerate(enemy_coords):
            pos = (pos[0] + origin_delta[0], pos[1] + origin_delta[1] - size)
            crop_area = VLMNode.get_crop_box(origin=pos, size=(size, size))
            pixels = list(img_turn_order_bw.crop(crop_area).getdata())
            proportions.append(np.count_nonzero(np.array(pixels)) / len(pixels))
            # If it's perfectly white - it is selected enemy
            if all(is_white(px) for px in pixels):
                return (idx, 1.0)
        
        # If no perfect white square has been found - return enemy index with maximum proportion of white
        # Threshold = 0.5
        enemy_id = np.argmax(proportions).item()
        if proportions[enemy_id] < significance:
            return len(enemy_coords) + 1, 1.0
        
        return enemy_id, proportions[enemy_id]

    @staticmethod
    def produce_bw_image(image: Image.Image, origin: tuple[int,int], size: tuple[int,int], name: str | None = None) -> Image.Image:
        """Produce black-and-white image out of red channel from HSV-representation of the image

        Args:
            image (Image.Image): screenshot of current player turn
            origin (tuple[int,int]): origin of turn order region
            size (tuple[int,int]): size of turn order region
            name (str | None, optional): filename to save image if needed. Defaults to None - don't save.

        Returns:
            Image.Image: black-and-white image in RGB
        """
        img_s = image.crop(VLMNode.get_crop_box(origin=origin, size=size)).convert("HSV").getchannel("S")
        img_v = image.crop(VLMNode.get_crop_box(origin=origin, size=size)).convert("HSV").getchannel("V")

        # Filter to increase contrast
        threshold = 127
        img_v.putdata([255 if x >= threshold else 0 for x in list(img_v.getdata())])
        img_s.putdata([255 if x >= threshold else 0 for x in list(img_s.getdata())])

        bw_image = Image.merge("HSV", [Image.new("L", size=img_v.size), img_s, img_v]).convert("RGB").getchannel("R")
        if name:
            bw_image.convert("RGB").save(name)
        
        return bw_image

    def __extract_enemy_parameters(self, img_player_turn: Image.Image, seed: int) -> dict:
        # TODO: Rework internals so the VLM is fed only with cropped images depicting actual numbers
        # torch.manual_seed(seed)
        # prompt_get_enemy_parameters = """Focus on "Enemy Data" information panel. Extract numbers of current HP, maximum HP, Stun, Attack, Defense, Arts strenght, Arts Defense, Speed."""
        # result = self.vlm(text=prompt_get_enemy_parameters, image=img_player_turn)[0]

        prompt_get_param = "Extract all numbers as JSON list."
        # Get parameters regions
        img_hp_origin = img_player_turn.crop(self.get_crop_box(self.enemy_stat_hp_origin, self.enemy_stat_hp_size))
        img_stun = img_player_turn.crop(self.get_crop_box(self.enemy_stat_stun_origin, self.enemy_stat_stun_size))
        img_atk_ats_speed = img_player_turn.crop(self.get_crop_box(self.enemy_stat_atk_ats_speed_origin, self.enemy_stat_atk_ats_speed_size))
        img_def_adf = img_player_turn.crop(self.get_crop_box(self.enemy_stat_def_adf_origin, self.enemy_stat_def_adf_size))

        values = []
        for img in [img_hp_origin, img_stun, img_atk_ats_speed, img_def_adf]:
            torch.manual_seed(seed)
            result = self.vlm(text=prompt_get_param, image=img)[0]
            start_pos = result.find("```json") + len("```json")
            result = result[start_pos:result.find("```", start_pos)]
            values += loads(result)
        
        enemy_params = {}
        enemy_params["hp"] = values[0]
        enemy_params["hp_max"] = values[1]
        enemy_params["stun"] = values[2]
        enemy_params["attack"] = values[3]
        enemy_params["arts_attack"] = values[4]
        enemy_params["speed"] = values[5]
        enemy_params["defense"] = values[6]
        enemy_params["arts_defense"] = values[7]

        return enemy_params

    def __extract_enemy_weaknesses(self, img_player_turn: Image.Image, seed: int) -> dict:
        #   3.1. Crop basic elements region. Origin: 1530, 522. Size (W, H): 50, 140
        # origin = (1530, 522)
        # size = (50, 140)
        origin = self.enemy_stat_weakness_basic_origin
        size = self.enemy_stat_weakness_basic_size
        crop_area = self.get_crop_box(origin, size)
        img_weaknesses = img_player_turn.crop(crop_area)
        #   3.2. Query VLM to get values for Ea, Wa, F, Wi
        prompt_get_weakness_numbers = """List numbers. Just give the numbers."""
        torch.manual_seed(seed)
        result = self.vlm(text=prompt_get_weakness_numbers, image=img_weaknesses)[0]
        result = result.replace("\n", ",").split(",")

        enemy_weakness = {
            "earth": int(result[0].strip()),
            "water": int(result[1].strip()),
            "fire": int(result[2].strip()),
            "wind": int(result[3].strip())
        }

        #   3.2. Crop higher elements region. Origin: 1715, 522. Size (W, H): 50, 104
        # origin = (1715, 522)
        # size = (50, 104)
        origin = self.enemy_stat_weakness_higher_elements_origin
        size = self.enemy_stat_weakness_higher_elements_size
        crop_area = self.get_crop_box(origin, size)
        img_weaknesses = img_player_turn.crop(crop_area)
        torch.manual_seed(seed)
        result = self.vlm(text=prompt_get_weakness_numbers, image=img_weaknesses)[0]
        result = result.replace("\n", ",").split(",")
        # Simple anti-hallucination sanity check
        if len(result) > 3:
            enemy_weakness = enemy_weakness | {
                "time": 100,
                "space": 100,
                "mirage": 100
            }
        else:
            enemy_weakness = enemy_weakness | {
                "time": int(result[0].strip()),
                "space": int(result[1].strip()),
                "mirage": int(result[2].strip())
            }
        
        return enemy_weakness

    def __extract_enemy_ailments(self, img_player_turn: Image.Image, seed: int) -> dict:
        #   4.1. Crop 1st column. Origin: 1556, 700. Size (W, H): 40, 174
        # origin = (1556, 700)
        # size = (40, 174)
        origin = self.enemy_stat_ailments_left_origin
        size = self.enemy_stat_ailments_left_size
        crop_area = self.get_crop_box(origin, size)
        img_ailments = img_player_turn.crop(crop_area)
        #   4.2. Query VLM to get ailments
        prompt_get_ailments = """List circles, and triangles in order of their occurence in the column. For each circle return 1, for each triangle - 0. Just give the list."""
        torch.manual_seed(seed)
        result = self.vlm(text=prompt_get_ailments, image=img_ailments)[0]
        result = result.split(",")
        #   4.3. Process returned string
        enemy_ailments = {
            "ailment_stat_down": bool(int(result[0])),
            "ailment_burn": bool(int(result[1])),
            "ailment_seal": bool(int(result[2])),
            "ailment_rot": bool(int(result[3])),
            "ailment_fear": bool(int(result[4]))
        }
        #   4.4. Crop 2nd column. Origin: 1750, 700. Size (W, H): 40, 174
        # origin = (1750, 700)
        # size = (40, 174)
        origin = self.enemy_stat_ailments_right_origin
        size = self.enemy_stat_ailments_right_size
        crop_area = self.get_crop_box(origin, size)
        img_ailments = img_player_turn.crop(crop_area)
        #   4.5. Query VLM to get ailments
        torch.manual_seed(seed)
        result = self.vlm(text=prompt_get_ailments, image=img_ailments)[0]
        result = result.split(",")
        #   4.6. Process returned string
        enemy_ailments = enemy_ailments | {
            "ailment_delay": bool(int(result[0])),
            "ailment_freeze": bool(int(result[1])),
            "ailment_mute": bool(int(result[2])),
            "ailment_blind": bool(int(result[3])),
            "ailment_deathblow": bool(int(result[4]))
        }

        return enemy_ailments   
    
    def _update_active_characters_strengths(self, img_player_turn: Image.Image, player_characters: list[PlayerCharacterStat], seed: int = 1741) -> list[PlayerCharacterStat]:
        logger.debug("STAT_UPDATER: Updating awareness on active characters")
        active_character_index = None

        for i, _ in enumerate(self.character_active_strength_origin):
            if not active_character_index:
                # Assume it's an active character
                origin = self.character_active_strength_origin[i] #(160, 108)
                size = self.character_active_strength_size #(240, 50)
            # else:
            #     origin = character_strength_origin[i] #(160, 108)
            #     size = character_strength_size[i] #(240, 50)
            
            # Prepare HP,EP,CP area
            crop_area = self.get_crop_box(origin, size)
            img_player_stats = img_player_turn.crop(crop_area)
            
            # Extract HP, EP, CP
            prompt_get_player_hpepcp = """Extract values of HP, EP, CP as CSV: name,value"""
            torch.manual_seed(seed)
            result = self.vlm(text=prompt_get_player_hpepcp, image=img_player_stats)

            character_params = {"character_id": i, "is_active": False}
            for line in result[0].splitlines():
                if len(line) <= 20 and "," in line.lower():
                    if "hp" in line.lower():
                        character_params["hp"] = int(line.split(",")[-1])
                    if "ep" in line.lower():
                        character_params["ep"] = int(line.split(",")[-1])
                    if "cp" in line.lower():
                        character_params["cp"] = int(line.split(",")[-1])

            # Find out if it's active character
            if not active_character_index:
                prompt_get_player_hpepcp = """Are words HP, EP, CP all present on the image? Give only answer."""
                torch.manual_seed(seed)
                result = self.vlm(text=prompt_get_player_hpepcp, image=img_player_stats)
                if result[0].lower() == "yes":
                    active_character_index = i
                    character_params["is_active"] = True
                    logger.info(f"STAT_UPDATER: Active character ID: {active_character_index}")
            
            # character_profiles.append(PlayerCharacterStat.from_dict(character_params, None))
            player_characters[i].hp = character_params["hp"]
            if player_characters[i].hp_max < player_characters[i].hp:
                player_characters[i].hp_max = player_characters[i].hp
            player_characters[i].ep = character_params["ep"]
            if player_characters[i].ep_max < player_characters[i].ep:
                player_characters[i].ep_max = player_characters[i].ep
            player_characters[i].cp = character_params["cp"]
            player_characters[i].is_active = character_params["is_active"]
        
        logger.debug("Character stats updated")
        return player_characters
