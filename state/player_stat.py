from typing import Annotated
from pydantic import BaseModel, Field


class PlayerCharacterStat(BaseModel):
    """Player character's strength, and status effects.
    """
    character_id: Annotated[int, Field(ge=0, default=0)]
    is_active: Annotated[bool, Field(default=False)]
    hp: Annotated[int, Field(ge=0, default=100)]
    hp_max: Annotated[int, Field(ge=0, default=100)]
    ep: Annotated[int, Field(ge=0, default=100)]
    ep_max: Annotated[int, Field(ge=0, default=100)]
    cp: Annotated[int, Field(ge=0, le=200, default=200)]
    attack: Annotated[int, Field(ge=0, default=100)]
    defense: Annotated[int, Field(ge=0, default=100)]
    arts_attack: Annotated[int, Field(ge=0, default=100)]
    arts_defense: Annotated[int, Field(ge=0, default=100)]
    speed: Annotated[int, Field(ge=0, default=100)]
    # Ailments:
    # true - in effect
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

    def is_alive(self) -> bool:
        """Check if this character is alive.
        """
        return self.hp > 0
    
    def to_prompt(self) -> str:
        """Get string for LLM reasoner context."""
        # Get characters
        character_str_template = "{parameters} {ailments}"

        # Parameters
        character_parameters = f"Character {self.character_id} has HP {self.hp} out of {self.hp_max}, EP {self.ep} out of {self.ep_max}, CP {self.cp} out of 200. " \
        f"Its attack {self.attack}, defense {self.defense}, arts strength {self.arts_attack}, arts defense {self.arts_defense}, initiative {self.speed}."

        # Ailments
        character_ailments = "It is affected by {ailments_list}"

        ailment_stat_down = "stats down" if self.ailment_stat_down else ""
        ailment_burn = "burn" if self.ailment_burn else ""
        ailment_seal = "seal" if self.ailment_seal else ""
        ailment_rot = "rot" if self.ailment_rot else ""
        ailment_fear = "fear" if self.ailment_fear else ""
        ailment_delay = "delay" if self.ailment_delay else ""
        ailment_freeze = "freeze" if self.ailment_freeze else ""
        ailment_mute = "mute" if self.ailment_mute else ""
        ailment_blind = "blind" if self.ailment_blind else ""
        ailment_deathblow = "deathblow" if self.ailment_deathblow else ""

        # Filter out empty strings
        ailments_list: list[str] = []
        for ailment in [ailment_stat_down, ailment_burn, ailment_seal, ailment_rot, ailment_fear, ailment_delay, ailment_freeze, ailment_mute, ailment_blind, ailment_deathblow]:
            if ailment:
                ailments_list.append(ailment)

        if "" == "".join(ailments_list):
            character_ailments = ""
        else:
            ailments_list = ", ".join(ailments_list)
        
            character_ailments = character_ailments.format(ailments_list=ailments_list)

        return character_str_template.format(parameters=character_parameters, ailments=character_ailments)

    @staticmethod
    def from_dict(parameters: dict, ailments: dict | None = None) -> object:
        """Construct PlayerCharacterStat from dictionaries

        Args:
            parameters (dict): character strength
            ailments (dict): current acting ailments. None - no ailments. TODO: IGNORED FOR NOW

        Raises:
            NotImplementedError: _description_

        Returns:
            object: PlayerCharacterStat object
        """
        if not ailments:
            ailments = {
                "ailment_stat_down": False,
                "ailment_burn": False,
                "ailment_seal": False,
                "ailment_rot": False,
                "ailment_fear": False,
                "ailment_delay": False,
                "ailment_freeze": False,
                "ailment_mute": False,
                "ailment_blind": False,
                "ailment_deathblow": False,
            }

        return PlayerCharacterStat(
            character_id = parameters["character_id"],
            is_active = parameters["is_active"],
            hp = parameters["hp"],
            hp_max = parameters.get("hp_max", parameters["hp"]),
            ep = parameters["ep"],
            ep_max = parameters.get("ep_max", parameters["ep"]),
            cp = parameters["cp"],
            # TODO: STUB. Extract strength
            attack = parameters.get("attack", 100),
            defense = parameters.get("defense", 100),
            arts_attack = parameters.get("arts_attack", 100),
            arts_defense = parameters.get("arts_defense", 100),
            speed = parameters.get("speed", 100),
            ailment_stat_down = ailments["ailment_stat_down"],
            ailment_burn = ailments["ailment_burn"],
            ailment_seal = ailments["ailment_seal"],
            ailment_rot = ailments["ailment_rot"],
            ailment_fear = ailments["ailment_fear"],
            ailment_delay = ailments["ailment_delay"],
            ailment_freeze = ailments["ailment_freeze"],
            ailment_mute = ailments["ailment_mute"],
            ailment_blind = ailments["ailment_blind"],
            ailment_deathblow = ailments["ailment_deathblow"],
        )
