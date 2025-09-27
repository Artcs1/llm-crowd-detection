import dspy


class IdentifyGroups(dspy.Signature):
    """"Identify sets of people that are very close to each other from the given 3D coordinates. Select an appropriate grouping threshold based on all pairwise distances. Do not hallucinate empty sets."""
    detections: list[dict] = dspy.InputField(desc="list of dictionary objects with the keys: person_id, x, y, z.")
    groups: list[list[int]] = dspy.OutputField(desc="A list of lists with person_ids of people grouped together.")


class IdentifyGroups_Zcoord(dspy.Signature):
    """"Identify sets of people that are very close to each other from the given 3D coordinates. Use all three coordinates in distance calculation. Select an appropriate grouping threshold based on all pairwise distances. Do not hallucinate empty sets."""
    detections: list[dict] = dspy.InputField(desc="list of dictionary objects with the keys: person_id, x, y, z.")
    groups: list[list[int]] = dspy.OutputField(desc="A list of lists with person_ids of people grouped together.")


class IdentifyGroups_Zcoord_Direction(dspy.Signature):
    """"Identify sets of people that are very close to each other from the given 3D coordinates. Use all three coordinates in distance calculation. Select an appropriate grouping threshold based on all pairwise distances. Groups of people should follow the same direction. Do not hallucinate empty sets."""
    detections: list[dict] = dspy.InputField(desc="list of dictionary objects with the keys: person_id, direction, x, y, z.")
    groups: list[list[int]] = dspy.OutputField(desc="A list of lists with person_ids of people grouped together.")
