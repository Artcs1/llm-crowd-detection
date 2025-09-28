# Inconsistent grouping. Often merges nearby people into same group even if there is space between some of them
# class IdentifyGroups_hint(dspy.Signature):
#     """Given a list of people with their 3D positions, group them into sets where each set contains people who are close to each other in space. Compute all pairwise distances between people. Choose a reasonable grouping threshold based on the distribution of these distances (for example, a small percentile that separates close points from distant ones). People belong to the same group if they are connected by pairwise distances below this threshold. Return only non-empty groups. Do not merge distant people into the same group."""
#     detections: list[dict] = dspy.InputField(desc="List of people, where each dictionary has keys: 'person_id', 'x', 'y', 'z'.")
#     groups: list[list[int]] = dspy.OutputField(desc="A list of groups, where each group is a list of person_ids that are close together.")


# After using a clearer prompt from ChatGPT, the exclusion of z coordinate in distance calculation was fixed.
# class IdentifyGroups_Zcoord(dspy.Signature):
#     """"Identify sets of people that are very close to each other from the given 3D coordinates. Use all three coordinates in distance calculation. Select an appropriate grouping threshold based on all pairwise distances. Do not hallucinate empty sets."""
#     detections: list[dict] = dspy.InputField(desc="list of dictionary objects with the keys: person_id, x, y, z.")
#     groups: list[list[int]] = dspy.OutputField(desc="A list of lists with person_ids of people grouped together.")


# class IdentifyGroups_Zcoord_Direction(dspy.Signature):
#     """"Identify sets of people that are very close to each other from the given 3D coordinates. Use all three coordinates in distance calculation. Select an appropriate grouping threshold based on all pairwise distances. Groups of people should follow the same direction. Do not hallucinate empty sets."""
#     detections: list[dict] = dspy.InputField(desc="list of dictionary objects with the keys: person_id, direction, x, y, z.")
#     groups: list[list[int]] = dspy.OutputField(desc="A list of lists with person_ids of people grouped together.")