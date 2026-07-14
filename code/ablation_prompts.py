import dspy



class IdentifyGroups2D(dspy.Signature):
    """Given a list of people with their 2D bounding-box center's positions, group them into sets where each set contains people who are close to each other in space. Compute all pairwise distances between people. Choose a reasonable grouping threshold based on the distribution of these distances (e.g., a natural gap or a small percentile that separates close points from distant ones). People belong to the same group if they are connected by pairwise distances below this threshold (transitively include people: if A is close to B and B is close to C, all three should be in the same group). Return only non-empty groups. Do not merge distant people into the same group."""

    detections: list[dict] = dspy.InputField(desc="list of dictionary objects with the keys: person_id, x, y.")
    groups: list[list[int]] = dspy.OutputField(desc="A list of lists with person_ids of people grouped together. All ids should appear at least once.")

class vlm_IdentifyGroups2DImage(dspy.Signature):
    """Given a list of people with their 2D bounding-box center's positions, group them into sets where each set contains people who are close to each other in space. Compute all pairwise distances between people. Choose a reasonable grouping threshold based on the distribution of these distances (e.g., a natural gap or a small percentile that separates close points from distant ones). People belong to the same group if they are connected by pairwise distances below this threshold (transitively include people: if A is close to B and B is close to C, all three should be in the same group). Return only non-empty groups. Do not merge distant people into the same group."""
    image: dspy.Image = dspy.InputField(desc="Image with people to group")
    detections: list[dict] = dspy.InputField(desc="list of dictionary objects with the keys: person_id, x, y.")
    groups: list[list[int]] = dspy.OutputField(desc="A list of lists with person_ids of people grouped together. All ids should appear at least once.")

class vlm_IdentifyGroups2DText(dspy.Signature):
    """Given a list of people with their 2D bounding-box center's positions, group them into sets where each set contains people who are close to each other in space. Compute all pairwise distances between people. Choose a reasonable grouping threshold based on the distribution of these distances (e.g., a natural gap or a small percentile that separates close points from distant ones). People belong to the same group if they are connected by pairwise distances below this threshold (transitively include people: if A is close to B and B is close to C, all three should be in the same group). Return only non-empty groups. Do not merge distant people into the same group."""
    detections: list[dict] = dspy.InputField(desc="list of dictionary objects with the keys: person_id, x, y.")
    groups: list[list[int]] = dspy.OutputField(desc="A list of lists with person_ids of people grouped together. All ids should appear at least once.")


class IdentifyGroups_withbboxes(dspy.Signature):
    """Given a list of people with their 3D positions, group them into sets where each set contains people who are close to each other in space. Compute all pairwise distances between people. Choose a reasonable grouping threshold based on the distribution of these distances. People belong to the same group if their pairwise distances are below this threshold. Return only non-empty groups. Do not merge distant people into the same group. Do not hallucinate non-existent person_id."""

    boundingboxes: list[dict] = dspy.InputField(desc="List of people, where each dictionary has keys in the standarf top-left bottom right notation [t, l, b, r]")
    detections: list[dict] = dspy.InputField(desc="List of people, where each dictionary has keys: 'person_id', 'x', 'y', 'z'.")
    groups: list[list[int]] = dspy.OutputField(desc="A list of groups, where each group is a list of person_ids that are close together. All ids should appear at least once.")

class vlm_IdentifyGroupsImage_withbboxes(dspy.Signature):
    """Given a list of people with their 3D positions, group them into sets where each set contains people who are close to each other in space. Compute all pairwise distances between people. Choose a reasonable grouping threshold based on the distribution of these distances. People belong to the same group if their pairwise distances are below this threshold. Return only non-empty groups. Do not merge distant people into the same group."""
    image: dspy.Image = dspy.InputField(desc="Image with people to group")
    boundingboxes: list[dict] = dspy.InputField(desc="List of people, where each dictionary has keys in the standarf top-left bottom right notation [t, l, b, r]")
    detections: list[dict] = dspy.InputField(desc="List of people, where each dictionary has keys: 'person_id', 'x', 'y', 'z'.")
    groups: list[list[int]] = dspy.OutputField(desc="A list of groups, where each group is a list of person_ids that are close together. All ids should appear at least once.")

    
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


class vlm_IdentifyGroups_DirectionText(dspy.Signature):
    """Given a list of people with their 3D positions and its direction, group them into sets where each set contains people who are close to each other in space. Compute all pairwise distances between people. Choose a reasonable grouping threshold based on the distribution of these distances. People belong to the same group if their pairwise distances are below this threshold. All people in the group might have a facing direction that is roughly aligned, You can infer this information from the image. Do not merge distant people into the same group."""
    detections: list[dict] = dspy.InputField(desc="List of people, where each dictionary has keys: 'person_id', 'x', 'y', 'z', 'direction'.")
    groups: list[list[int]] = dspy.OutputField(desc="A list of groups, where each group is a list of person_ids that are close together.")

class vlm_IdentifyGroups_TransitiveText(dspy.Signature):
    """Given a list of people with their 3D positions, group them into sets where each set contains people who are close to each other in space. Compute all pairwise distances between people. Choose a reasonable grouping threshold based on the distribution of these distances. People belong to the same group if their pairwise distances are below this threshold. Transitively include people: if A is close to B and B is close to C, all three should be in the same group. Extrapolate this reason to mroe groups. Return only non-empty groups. Do not merge distant people into the same group."""
    detections: list[dict] = dspy.InputField(desc="List of people, where each dictionary has keys: 'person_id', 'x', 'y', 'z'.")
    groups: list[list[int]] = dspy.OutputField(desc="A list of groups, where each group is a list of person_ids that are close together.")

class vlm_IdentifyGroups_DirectionTransitiveText(dspy.Signature):
    """Given a list of people with their 3D positions and its direction, group them into sets where each set contains people who are close to each other in space. Compute all pairwise distances between people. Choose a reasonable grouping threshold based on the distribution of these distances. People belong to the same group if their pairwise distances are below this threshold. Transitively include people: if A is close to B and B is close to C, all three should be in the same group. All people in the group might have a facing direction that is roughly aligned, You can infer this information from the image. Return only non-empty groups. Do not merge distant people into the same group."""
    detections: list[dict] = dspy.InputField(desc="List of people, where each dictionary has keys: 'person_id', 'x', 'y', 'z', 'direction'.")
    groups: list[list[int]] = dspy.OutputField(desc="A list of groups, where each group is a list of person_ids that are close together.")

class vlm_IdentifyGroups_DirectionImage(dspy.Signature):
    """Given a list of people with their 3D positions and its direction, group them into sets where each set contains people who are close to each other in space. Compute all pairwise distances between people. Choose a reasonable grouping threshold based on the distribution of these distances. People belong to the same group if their pairwise distances are below this threshold. All people in the group might have a facing direction that is roughly aligned, You can infer this information from the image. Do not merge distant people into the same group."""
    image: dspy.Image = dspy.InputField(desc="Image with people to group")
    detections: list[dict] = dspy.InputField(desc="List of people, where each dictionary has keys: 'person_id', 'x', 'y', 'z', 'direction'.")
    groups: list[list[int]] = dspy.OutputField(desc="A list of groups, where each group is a list of person_ids that are close together.")

class vlm_IdentifyGroups_TransitiveImage(dspy.Signature):
    """Given a list of people with their 3D positions, group them into sets where each set contains people who are close to each other in space. Compute all pairwise distances between people. Choose a reasonable grouping threshold based on the distribution of these distances. People belong to the same group if their pairwise distances are below this threshold. Transitively include people: if A is close to B and B is close to C, all three should be in the same group. Extrapolate this reason to mroe groups. Return only non-empty groups. Do not merge distant people into the same group."""
    image: dspy.Image = dspy.InputField(desc="Image with people to group")
    detections: list[dict] = dspy.InputField(desc="List of people, where each dictionary has keys: 'person_id', 'x', 'y', 'z'.")
    groups: list[list[int]] = dspy.OutputField(desc="A list of groups, where each group is a list of person_ids that are close together.")

class vlm_IdentifyGroups_DirectionTransitiveImage(dspy.Signature):
    """Given a list of people with their 3D positions and its direction, group them into sets where each set contains people who are close to each other in space. Compute all pairwise distances between people. Choose a reasonable grouping threshold based on the distribution of these distances. People belong to the same group if their pairwise distances are below this threshold. Transitively include people: if A is close to B and B is close to C, all three should be in the same group. All people in the group might have a facing direction that is roughly aligned, You can infer this information from the image. Return only non-empty groups. Do not merge distant people into the same group."""
    image: dspy.Image = dspy.InputField(desc="Image with people to group")
    detections: list[dict] = dspy.InputField(desc="List of people, where each dictionary has keys: 'person_id', 'x', 'y', 'z', 'direction'.")
    groups: list[list[int]] = dspy.OutputField(desc="A list of groups, where each group is a list of person_ids that are close together.")


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



# class RecognizeGroupActivity_MultiGroup(dspy.Signature):
#     """Given an image with multiple people and the 2D coordinates of bounding boxes enclosing different subsets of them, name the activity (or activities) that people inside each bounding box are engaged in. Consider their poses, interactions, and any objects they might be using."""
#     image: dspy.Image = dspy.InputField(desc="Image with people")
#     bbox: list[list[int]] = dspy.InputField(desc="List of bounding boxes around groups of people, each in top-left and bottom-right notation: [x1, y1, x2, y2]")
#     activity: list[list[str]] = dspy.OutputField(desc="Name of one or more activities that people inside each bounding box are engaged in.")
