from typing import Annotated
from pydantic import BaseModel, Field

class EnemyStat(BaseModel):
    """Enemy strength, and weaknesses.
    """
    # Generic information
    enemy_id: Annotated[int, Field(ge=0)]
    target_method_f: Annotated[bool, Field(default=True)] # True - Use F to target this enemy; False - Use R
    basic_attack_enabled: Annotated[bool, Field(default=False)] # True - can be attacked with basic attack
    # Parameters
    hp: Annotated[int, Field(ge=0)]
    hp_max: Annotated[int, Field(ge=0)]
    stun: Annotated[int, Field(ge=0, default=0)]
    attack: Annotated[int, Field(ge=0)]
    defense: Annotated[int, Field(ge=0)]
    arts_attack: Annotated[int, Field(ge=0)]
    arts_defense: Annotated[int, Field(ge=0)]
    speed: Annotated[int, Field(ge=0)]
    # Weaknesses
    weakness_earth: Annotated[int, Field(ge=0, default=100)]
    weakness_water: Annotated[int, Field(ge=0, default=100)]
    weakness_fire: Annotated[int, Field(ge=0, default=100)]
    weakness_wind: Annotated[int, Field(ge=0, default=100)]
    weakness_time: Annotated[int, Field(ge=0, default=100)]
    weakness_space: Annotated[int, Field(ge=0, default=100)]
    weakness_mirage: Annotated[int, Field(ge=0, default=100)]
    # Ailments:
    # true - susceptible
    ailment_stat_down: Annotated[bool, Field(default=False)]
    ailment_burn: Annotated[bool, Field(default=False)]
    ailment_seal: Annotated[bool, Field(default=False)]
    ailment_rot: Annotated[bool, Field(default=False)]
    ailment_fear: Annotated[bool, Field(default=False)]
    ailment_delay: Annotated[bool, Field(default=False)]
    ailment_freeze: Annotated[bool, Field(default=False)]
    ailment_mute: Annotated[bool, Field(default=False)]
    ailment_blind: Annotated[bool, Field(default=False)]
    ailment_deathblow: Annotated[bool, Field(default=False)]

    def to_prompt(self) -> str:
        """Get string for LLM reasoner context."""
        # Get enemies
        enemy_str_template = "{parameters} {weaknesses} {resistances} {ailments}{relative_position}"
        
        # Parameters
        enemy_parameters = "Enemy {enemy_id} has HP {hp} out of {hp_max}, stun level {stun}%. Its attack {attack}, defense {defense}, arts strength {arts_attack}, arts defense {arts_defense}, initiative {speed}."
        enemy_parameters = enemy_parameters.format(
            enemy_id = self.enemy_id,
            hp = self.hp, hp_max = self.hp_max,
            stun = self.stun, attack = self.attack, defense = self.defense,
            arts_attack = self.arts_attack, arts_defense = self.arts_defense, speed = self.speed
        )

        # Weaknesses
        enemy_weaknesses = "It is weak to elements: {weakness_list}."

        weakness_earth = f"earth {self.weakness_earth}% of arts damage" if self.weakness_earth > 100 else None
        weakness_water = f"water {self.weakness_water}% of arts damage" if self.weakness_water > 100 else None
        weakness_fire = f"fire {self.weakness_fire}% of arts damage" if self.weakness_fire > 100 else None
        weakness_wind = f"wind {self.weakness_wind}% of arts damage" if self.weakness_wind > 100 else None
        weakness_time = f"time {self.weakness_time}% of arts damage" if self.weakness_time > 100 else None
        weakness_space = f"space {self.weakness_space}% of arts damage" if self.weakness_space > 100 else None
        weakness_mirage = f"mirage {self.weakness_mirage}% of arts damage" if self.weakness_mirage > 100 else None

        # Filter out empty strings
        weakness_list: list[str] = []
        for weakness in [weakness_earth, weakness_water, weakness_fire, weakness_wind, weakness_time, weakness_space, weakness_mirage]:
            if weakness:
                weakness_list.append(weakness)

        if "" == "".join(weakness_list):
            enemy_weaknesses = ""
        else:
            weakness_list = ", ".join(weakness_list)
            enemy_weaknesses = enemy_weaknesses.format(weakness_list=weakness_list)

        # Resistances
        enemy_resistances = "It is resistant to elements: {resistance_list}."

        resist_earth = f"earth {self.weakness_earth}% of arts damage" if self.weakness_earth < 100 else None
        resist_water = f"water {self.weakness_water}% of arts damage" if self.weakness_water < 100 else None
        resist_fire = f"fire {self.weakness_fire}% of arts damage" if self.weakness_fire < 100 else None
        resist_wind = f"wind {self.weakness_wind}% of arts damage" if self.weakness_wind < 100 else None
        resist_time = f"time {self.weakness_time}% of arts damage" if self.weakness_time < 100 else None
        resist_space = f"space {self.weakness_space}% of arts damage" if self.weakness_space < 100 else None
        resist_mirage = f"mirage {self.weakness_mirage}% of arts damage" if self.weakness_mirage < 100 else None

        # Filter out empty strings
        resistance_list: list[str] = []
        for resist in [resist_earth, resist_water, resist_fire, resist_wind, resist_time, resist_space, resist_mirage]:
            if resist:
                resistance_list.append(weakness)

        if "" == "".join(resistance_list):
            resistance_list = ""
        else:
            resistance_list = ", ".join(resistance_list)
            enemy_resistances = enemy_resistances.format(resistance_list=resistance_list)

        # Ailments
        enemy_ailments = "It is susceptible to get {ailments_list}."

        ailment_stat_down = "stat down" if self.ailment_stat_down else None
        ailment_burn = "burn" if self.ailment_burn else None
        ailment_seal = "seal" if self.ailment_seal else None
        ailment_rot = "rot" if self.ailment_rot else None
        ailment_fear = "fear" if self.ailment_fear else None
        ailment_delay = "delay" if self.ailment_delay else None
        ailment_freeze = "freeze" if self.ailment_freeze else None
        ailment_mute = "mute" if self.ailment_mute else None
        ailment_blind = "blind" if self.ailment_blind else None
        ailment_deathblow = "deathblow" if self.ailment_deathblow else None

        # Filter out empty strings
        ailments_list: list[str] = []
        for ailment in [ailment_stat_down, ailment_burn, ailment_seal, ailment_rot, ailment_fear, ailment_delay, ailment_freeze, ailment_mute, ailment_blind, ailment_deathblow]:
            if ailment:
                ailments_list.append(ailment)

        if "" == "".join(ailments_list):
            enemy_ailments = ""
        else:
            ailments_list = ", ".join(ailments_list)
        
            enemy_ailments = enemy_ailments.format(ailments_list=ailments_list)
        
        # Relative enemy position
        relative_position = " This enemy is within reach of basic attack, crafts, and attack arts." if self.basic_attack_enabled else " This enemy is out of reach for basic attack, but can be attacked by crafts, and attack arts."

        return enemy_str_template.format(parameters=enemy_parameters, weaknesses=enemy_weaknesses, resistances=enemy_resistances, ailments=enemy_ailments, relative_position=relative_position).strip()

    @staticmethod
    def from_dicts(enemy_id: int, target_method_f: bool, basic_attack_enabled: bool, parameters: dict, weaknesses: dict, ailments: dict) -> object:
        """Construct EnemyStat object from dictionaries."""
        return EnemyStat(
            enemy_id = enemy_id,
            target_method_f = target_method_f,
            basic_attack_enabled = basic_attack_enabled,
            # Parameters
            hp = parameters["hp"],
            hp_max = parameters["hp_max"],
            stun = parameters["stun"],
            attack = parameters["attack"],
            defense = parameters["defense"],
            arts_attack = parameters["arts_attack"],
            arts_defense = parameters["arts_defense"],
            speed = parameters["speed"],
            # Weaknesses
            weakness_earth = weaknesses.get("earth", 100),
            weakness_water = weaknesses.get("water", 100),
            weakness_fire = weaknesses.get("fire", 100),
            weakness_wind = weaknesses.get("wind", 100),
            weakness_time = weaknesses.get("time", 100),
            weakness_space = weaknesses.get("space", 100),
            weakness_mirage = weaknesses.get("mirage", 100),
            # Ailments
            ailment_stat_down = ailments["ailment_stat_down"],
            ailment_burn = ailments["ailment_burn"],
            ailment_seal = ailments["ailment_seal"],
            ailment_rot = ailments["ailment_rot"],
            ailment_fear = ailments["ailment_fear"],
            ailment_delay = ailments["ailment_delay"],
            ailment_freeze = ailments["ailment_freeze"],
            ailment_mute = ailments["ailment_mute"],
            ailment_blind = ailments["ailment_blind"],
            ailment_deathblow = ailments["ailment_deathblow"]
        )

    def is_alive(self) -> bool:
        """Check if this enemy is alive.
        """
        return self.hp > 0

