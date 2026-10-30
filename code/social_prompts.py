"""Task text of the p1_social prompt (zero-shot --prompt_method p1_social, SFT / RL --variant p1_social), kept in one
place so the dspy signatures (prompts.py) and the training system prompts use exactly the same words. Same inputs as
p1_visual (annotated image + person_id, x, y, z; full mode adds movement_direction); only the reasoning it asks for
differs: proximity plus orientation, shared motion and interaction read from the image, and no chaining. Groups are
disjoint and cover every person in both datasets' GT, hence "exactly one group".
"""

_CUES_HEAD = (
    "Weigh these cues together:\n"
    "1. Proximity: members are usually within personal or social distance of each other; judge it from the "
    "distances in this scene, not a fixed threshold.\n"
    "2. Orientation, from the image: standing members usually face each other or a shared point (a circle, side by "
    "side, or an L around a shared space); close people facing away, or one behind another in a queue, are usually "
    "not a group.\n"
)
_CUES_TAIL = (
    "4. Interaction, from the image: talking, gestures, touching or holding hands, carrying something together, a "
    "child with an adult, sitting at the same table.\n"
    "5. No chaining: do not merge a crowd or a queue into one group just because each person is close to the next.\n"
    "Each person belongs to exactly one group; a person with no group is a group of one."
)
_INTRO = ("Given one video frame with every person's bounding box and id label drawn on it, decide which people form "
          "social groups: people who are together, interacting with each other or moving as one unit. ")

# single frame: keys person_id, x, y, z
SOCIAL_TASK = (
    _INTRO
    + "You are also given a JSON array of the people in it, each an object with keys 'person_id', 'x', 'y', 'z' "
      "(3D position).\n"
    + _CUES_HEAD
    + "3. Shared motion, from the image: people walking together keep pace side by side, heading the same way; "
      "someone close but walking elsewhere is usually not with them.\n"
    + _CUES_TAIL
)

# full mode: keys person_id, x, y, z, movement_direction
SOCIAL_FULL_TASK = (
    _INTRO
    + "You are also given a JSON array of the people in it, each an object with keys 'person_id', 'x', 'y', 'z' "
      "(3D position) and 'movement_direction' (their net movement direction across the frames leading up to this "
      "one, or 'stationary').\n"
    + _CUES_HEAD
    + "3. Shared motion: people walking together share the same movement_direction and stay side by side; someone "
      "close but with a different movement_direction is usually not with them. Two stationary people can still be "
      "a group.\n"
    + _CUES_TAIL
)

# training system prompts (sft/rl_training_vlm_nodspy_merged.py --variant p1_social): the task plus the output format
_JSON_ONLY = '\nRespond with ONLY a JSON object of the form {"groups": [[person_id, ...], ...]} and no other text.'
SYSTEM_PROMPT_SOCIAL = SOCIAL_TASK + _JSON_ONLY
SYSTEM_PROMPT_SOCIAL_FULL = SOCIAL_FULL_TASK + _JSON_ONLY
