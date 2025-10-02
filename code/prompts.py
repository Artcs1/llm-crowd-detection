import dspy

class IdentifyGroups2D(dspy.Signature):
    """Given a list of people with their 2D bounding-box center's positions, group them into sets where each set contains people who are close to each other in space. Compute all pairwise distances between people. Choose a reasonable grouping threshold based on the distribution of these distances (e.g., a natural gap or a small percentile that separates close points from distant ones). People belong to the same group if they are connected by pairwise distances below this threshold (transitively include people: if A is close to B and B is close to C, all three should be in the same group). Return only non-empty groups. Do not merge distant people into the same group."""
    detections: list[dict] = dspy.InputField(desc="list of dictionary objects with the keys: person_id, x, y.")
    groups: list[list[int]] = dspy.OutputField(desc="A list of lists with person_ids of people grouped together.")



class IdentifyGroups(dspy.Signature):
    """Given a list of people with their 3D positions, group them into sets where each set contains people who are close to each other in space. Compute all pairwise distances between people. Choose a reasonable grouping threshold based on the distribution of these distances. People belong to the same group if their pairwise distances are below this threshold. Return only non-empty groups. Do not merge distant people into the same group. Do not hallucinate non-existent person_id."""
    detections: list[dict] = dspy.InputField(desc="List of people, where each dictionary has keys: 'person_id', 'x', 'y', 'z'.")
    groups: list[list[int]] = dspy.OutputField(desc="A list of groups, where each group is a list of person_ids that are close together.")



class IdentifyGroups_Direction(dspy.Signature):
    """Given a list of people with their 3D positions, group them into sets where each set contains people who are close to each other in space. Compute all pairwise distances between people. Choose a reasonable grouping threshold based on the distribution of these distances. People belong to the same group if their pairwise distances are below this threshold. All people in the group must have a facing direction that is roughly aligned. Do not merge distant people into the same group. Do not hallucinate non-existent person_id."""
    detections: list[dict] = dspy.InputField(desc="List of people, where each dictionary has keys: 'person_id', 'x', 'y', 'z', 'direction'.")
    groups: list[list[int]] = dspy.OutputField(desc="A list of groups, where each group is a list of person_ids that are close together.")



class IdentifyGroups_Transitive(dspy.Signature):
    """Given a list of people with their 3D positions, group them into sets where each set contains people who are close to each other in space. Compute all pairwise distances between people. Choose a reasonable grouping threshold based on the distribution of these distances. People belong to the same group if their pairwise distances are below this threshold. Transitively include people: if A is close to B and B is close to C, all three should be in the same group. Return only non-empty groups. Do not merge distant people into the same group. Do not hallucinate non-existent person_id."""
    detections: list[dict] = dspy.InputField(desc="List of people, where each dictionary has keys: 'person_id', 'x', 'y', 'z'.")
    groups: list[list[int]] = dspy.OutputField(desc="A list of groups, where each group is a list of person_ids that are close together.")



class IdentifyGroups_DirectionTransitive(dspy.Signature):
    """Given a list of people with their 3D positions, group them into sets where each set contains people who are close to each other in space. Compute all pairwise distances between people. Choose a reasonable grouping threshold based on the distribution of these distances. People belong to the same group if their pairwise distances are below this threshold. People belong to the same group if their pairwise distances are below this threshold. Transitively include people: if A is close to B and B is close to C, all three should be in the same group. Return only non-empty groups. Do not merge distant people into the same group. Do not hallucinate non-existent person_id."""
    detections: list[dict] = dspy.InputField(desc="List of people, where each dictionary has keys: 'person_id', 'x', 'y', 'z', 'direction'.")
    groups: list[list[int]] = dspy.OutputField(desc="A list of groups, where each group is a list of person_ids that are close together.")
