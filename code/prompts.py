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



class IdentifyGroups_AllFrames(dspy.Signature):
    """Given detections of people with their 3D positions across 50 frames of a video, compute groups of people who are close together in the specified target frame. Use spatial information from all frames as context — for example, to infer stable group memberships even if people temporarily move apart or come closer. Compute pairwise distances between people in the target frame and choose a reasonable grouping threshold based on the distribution of these distances. People belong to the same group if they are spatially close and consistently remain close across frames. Return only non-empty groups. Do not merge distant people into the same group. Do not hallucinate non-existent person_id."""
    all_frames: list[list[dict]] = dspy.InputField(
        desc="List of frames, where each frame is a list of people detections. Each detection is a dict with keys: 'person_id', 'x', 'y', 'z'. Total ~50 frames."
    )
    target_frame: int = dspy.InputField(
        desc="The (1-based) index of the frame for which groups should be computed."
    )
    groups: list[list[int]] = dspy.OutputField(desc="Groups of person_ids who are close together in the target frame, inferred using spatial and temporal context from all frames.")



class IdentifyGroups2DImage(dspy.Signature):
    """Given a list of people with their 2D bounding-box center's positions, group them into sets where each set contains people who are close to each other in space. Compute all pairwise distances between people. Choose a reasonable grouping threshold based on the distribution of these distances (e.g., a natural gap or a small percentile that separates close points from distant ones). People belong to the same group if they are connected by pairwise distances below this threshold (transitively include people: if A is close to B and B is close to C, all three should be in the same group). Return only non-empty groups. Do not merge distant people into the same group."""
    image: dspy.Image = dspy.InputField(desc="Image with people to group")
    detections: list[dict] = dspy.InputField(desc="list of dictionary objects with the keys: person_id, x, y.")
    groups: list[list[int]] = dspy.OutputField(desc="A list of lists with person_ids of people grouped together.")

class IdentifyGroupsImage(dspy.Signature):
    """Given a list of people with their 3D positions, group them into sets where each set contains people who are close to each other in space. Compute all pairwise distances between people. Choose a reasonable grouping threshold based on the distribution of these distances. People belong to the same group if their pairwise distances are below this threshold. Return only non-empty groups. Do not merge distant people into the same group."""
    image: dspy.Image = dspy.InputField(desc="Image with people to group")
    detections: list[dict] = dspy.InputField(desc="List of people, where each dictionary has keys: 'person_id', 'x', 'y', 'z'.")
    groups: list[list[int]] = dspy.OutputField(desc="A list of groups, where each group is a list of person_ids that are close together.")

class IdentifyGroups_DirectionImage(dspy.Signature):
    """Given a list of people with their 3D positions, group them into sets where each set contains people who are close to each other in space. Compute all pairwise distances between people. Choose a reasonable grouping threshold based on the distribution of these distances. People belong to the same group if their pairwise distances are below this threshold. All people in the group must have a facing direction that is roughly aligned. Do not merge distant people into the same group."""
    image: dspy.Image = dspy.InputField(desc="Image with people to group")
    detections: list[dict] = dspy.InputField(desc="List of people, where each dictionary has keys: 'person_id', 'x', 'y', 'z', 'direction'.")
    groups: list[list[int]] = dspy.OutputField(desc="A list of groups, where each group is a list of person_ids that are close together.")

class IdentifyGroups_TransitiveImage(dspy.Signature):
    """Given a list of people with their 3D positions, group them into sets where each set contains people who are close to each other in space. Compute all pairwise distances between people. Choose a reasonable grouping threshold based on the distribution of these distances. People belong to the same group if their pairwise distances are below this threshold. Transitively include people: if A is close to B and B is close to C, all three should be in the same group. Return only non-empty groups. Do not merge distant people into the same group."""
    image: dspy.Image = dspy.InputField(desc="Image with people to group")
    detections: list[dict] = dspy.InputField(desc="List of people, where each dictionary has keys: 'person_id', 'x', 'y', 'z'.")
    groups: list[list[int]] = dspy.OutputField(desc="A list of groups, where each group is a list of person_ids that are close together.")

class IdentifyGroups_DirectionTransitiveImage(dspy.Signature):
    """Given a list of people with their 3D positions, group them into sets where each set contains people who are close to each other in space. Compute all pairwise distances between people. Choose a reasonable grouping threshold based on the distribution of these distances. People belong to the same group if their pairwise distances are below this threshold. People belong to the same group if their pairwise distances are below this threshold. Transitively include people: if A is close to B and B is close to C, all three should be in the same group. Return only non-empty groups. Do not merge distant people into the same group."""
    image: dspy.Image = dspy.InputField(desc="Image with people to group")
    detections: list[dict] = dspy.InputField(desc="List of people, where each dictionary has keys: 'person_id', 'x', 'y', 'z', 'direction'.")

