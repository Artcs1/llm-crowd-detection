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

class vlm_GroupsQAonlyImage(dspy.Signature):
    """
    From the image infer the list of people with their 3D positions, group them into sets where each set contains people who are close to each other in space. Mentally compute all pairwise distances between people. Choose a reasonable grouping threshold based on the distribution of these distances. People belong to the same group if their pairwise distances are below this threshold. Return only non-empty groups. Do not merge distant people into the same group.
    
    Grouping rules:
    - People must be close together.
    - Use intuition instead of fixed distance thresholds.

    Final answer format:
    - A list of groups.
    - Each group is a list of bounding boxes where each bbox contains only a person.
    - Output ONLY the list, no extra text.
    """

    image: dspy.Image = dspy.InputField(desc="Image with people to group")
    groups: list[list[list[int]]] = dspy.OutputField(
        desc="List of List of persons with bounding boxes, each in top left and bottom right notation: [x1, y1, x2, y2]"
    )

class GroupsQAImage_b2_p1(dspy.Signature):
    """
    Based on the image, how many groups of people that may know each other. Can you identify?
    """

    image: dspy.Image = dspy.InputField(desc="Image with people to group")
    answer: list[str] = dspy.OutputField(
        desc="List of string containing a sentence associated with the group"
    )

class fine_GroupsQAImage_b2_p2(dspy.Signature):
    """
    Based on the image, Locate the group mentioned in the question
    """

    question: str = dspy.InputField(desc="Reference to locate the group")
    image: dspy.Image = dspy.InputField(desc="Image with people to group")
    groups: list[list[int]] = dspy.OutputField(
        desc="List of List of persons with bounding boxes, each in top left and bottom right notation: [x1, y1, x2, y2]"
    )

class Predictions:
    def __init__(self, groups):
        self.groups = groups  # list of Group objects
    
class baseline2(dspy.Module):
    def __init__(self):
        super().__init__()
        self.p1 = dspy.ChainOfThought(GroupsQAImage_b2_p1)
        self.p2 = dspy.ChainOfThought(fine_GroupsQAImage_b2_p2)
        
    def forward(self, image=None, detections=None):
        # Analyze repository purpose and concepts

        pred = self.p1(image=dspy.Image.from_file(image))
        string_groups = pred.answer

        groups = []
        for group in string_groups:
            pred = self.p2(question="Locate the group which is a:"+group,image=dspy.Image.from_file(image))
            groups.append(pred.groups)

        return Predictions(groups)
    
class vlm_IdentifyGroupsImage(dspy.Signature):
    """Given a list of people with their 3D positions, group them into sets where each set contains people who are close to each other in space. Compute all pairwise distances between people. Choose a reasonable grouping threshold based on the distribution of these distances. People belong to the same group if their pairwise distances are below this threshold. Return only non-empty groups. Do not merge distant people into the same group."""
    image: dspy.Image = dspy.InputField(desc="Image with people to group")
    detections: list[dict] = dspy.InputField(desc="List of people, where each dictionary has keys: 'person_id', 'x', 'y', 'z'.")
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
    """Given a list of people with their 3D positions and its direction, group them into sets where each set contains people who are close to each other in space. Compute all pairwise distances between people. Choose a reasonable grouping threshold based on the distribution of these distances. People belong to the same group if their pairwise distances are below this threshold. People belong to the same group if their pairwise distances are below this threshold. Transitively include people: if A is close to B and B is close to C, all three should be in the same group. All people in the group might have a facing direction that is roughly aligned, You can infer this information from the image. Return only non-empty groups. Do not merge distant people into the same group."""
    image: dspy.Image = dspy.InputField(desc="Image with people to group")
    detections: list[dict] = dspy.InputField(desc="List of people, where each dictionary has keys: 'person_id', 'x', 'y', 'z', 'direction'.")
    groups: list[list[int]] = dspy.OutputField(desc="A list of groups, where each group is a list of person_ids that are close together.")

class vlm_IdentifyGroups2DText(dspy.Signature):
    """Given a list of people with their 2D bounding-box center's positions, group them into sets where each set contains people who are close to each other in space. Compute all pairwise distances between people. Choose a reasonable grouping threshold based on the distribution of these distances (e.g., a natural gap or a small percentile that separates close points from distant ones). People belong to the same group if they are connected by pairwise distances below this threshold (transitively include people: if A is close to B and B is close to C, all three should be in the same group). Return only non-empty groups. Do not merge distant people into the same group."""
    detections: list[dict] = dspy.InputField(desc="list of dictionary objects with the keys: person_id, x, y.")
    groups: list[list[int]] = dspy.OutputField(desc="A list of lists with person_ids of people grouped together.")

class vlm_IdentifyGroupsText(dspy.Signature):
    """Given a list of people with their 3D positions, group them into sets where each set contains people who are close to each other in space. Compute all pairwise distances between people. Choose a reasonable grouping threshold based on the distribution of these distances. People belong to the same group if their pairwise distances are below this threshold. Return only non-empty groups. Do not merge distant people into the same group."""
    detections: list[dict] = dspy.InputField(desc="List of people, where each dictionary has keys: 'person_id', 'x', 'y', 'z'.")
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
    """Given a list of people with their 3D positions and its direction, group them into sets where each set contains people who are close to each other in space. Compute all pairwise distances between people. Choose a reasonable grouping threshold based on the distribution of these distances. People belong to the same group if their pairwise distances are below this threshold. People belong to the same group if their pairwise distances are below this threshold. Transitively include people: if A is close to B and B is close to C, all three should be in the same group. All people in the group might have a facing direction that is roughly aligned, You can infer this information from the image. Return only non-empty groups. Do not merge distant people into the same group."""
    detections: list[dict] = dspy.InputField(desc="List of people, where each dictionary has keys: 'person_id', 'x', 'y', 'z', 'direction'.")

class vlm_GroupsQAonlyFullImage(dspy.Signature):
    """ 
    Compute groups of people who are close together in the reference image. From the image and video infer the list of people with their 3D positions, group them into sets where each set contains people who are close to each other in space. Mentally compute all pairwise distances between people. Choose a reasonable grouping threshold based on the distribution of these distances. People belong to the same group if their pairwise distances are below this threshold. Return only non-empty groups. Do not merge distant people into the same group.
    
    Grouping rules:
    - People must be close together.
    - Use intuition instead of fixed distance thresholds.

    Final answer format:
    - A list of groups.
    - Each group is a list of bounding boxes where each bbox contains only a person.
    - Output ONLY the list, no extra text.
    """

    image: dspy.Image = dspy.InputField(desc="Image with people to group")
    video: list[dspy.Image] = dspy.InputField(desc="Reference video to support the group decision")
    groups: list[list[list[int]]] = dspy.OutputField(
        desc="List of List of persons with bounding boxes, each in top left and bottom right notation: [x1, y1, x2, y2]"
    )   


class GroupsQAFullImage_b2_p1(dspy.Signature):
    """
    Based on the image and video, how many groups of people that may know each other in the reference image. Can you identify?
    """
    image: dspy.Image = dspy.InputField(desc="Image with people to group")
    video: list[dspy.Image] = dspy.InputField(desc="Reference video to support the group decision")
    answer: list[str] = dspy.OutputField(
        desc="List of string containing a sentence associated with the group"
    )

class fine_GroupsQAFullImage_b2_p2(dspy.Signature):
    """
    Based on the image, Locate the group mentioned in the question
    """

    question: str = dspy.InputField(desc="Reference to locate the group")
    image: dspy.Image = dspy.InputField(desc="Image with people to group")
    video: list[dspy.Image] = dspy.InputField(desc="Reference video to support the group decision")
    groups: list[list[int]] = dspy.OutputField(
        desc="List of List of persons with bounding boxes, each in top left and bottom right notation: [x1, y1, x2, y2]"
    )

class full_baseline2(dspy.Module):
    def __init__(self):
        super().__init__()
        self.p1 = dspy.ChainOfThought(GroupsQAFullImage_b2_p1)
        self.p2 = dspy.ChainOfThought(fine_GroupsQAFullImage_b2_p2)
        
    def forward(self, image=None, video=None, all_frames=None, target_frame=None):
        pred = self.p1(image=image, video = video)
        string_groups = pred.answer

        groups = []
        for group in string_groups:
            pred = self.p2(question="Locate the group which is a:"+group,image=image, video=video)
            groups.append(pred.groups)

        return Predictions(groups)
 

class vlm_IdentifyGroups_AllFramesText(dspy.Signature):
    """Given detections of people with their 3D positions across 50 frames of a video, compute groups of people who are close together in the specified target frame. Use spatial information from all frames as context — for example, to infer stable group memberships even if people temporarily move apart or come closer. Compute pairwise distances between people in the target frame and choose a reasonable grouping threshold based on the distribution of these distances. People belong to the same group if they are spatially close and consistently remain close across frames. Return only non-empty groups. Do not merge distant people into the same group. Do not hallucinate non-existent person_id. Reference Image and Video are given to support your decision."""
    all_frames: list[list[dict]] = dspy.InputField(
        desc="List of frames, where each frame is a list of people detections. Each detection is a dict with keys: 'person_id', 'x', 'y', 'z'. Total ~50 frames."
    )   
    target_frame: int = dspy.InputField(
        desc="The (1-based) index of the frame for which groups should be computed."
    )   
    groups: list[list[int]] = dspy.OutputField(desc="Groups of person_ids who are close together in the target frame, inferred using spatial and temporal context from all frames.")


class vlm_IdentifyGroups_AllFramesImage(dspy.Signature):
    """Given detections of people with their 3D positions across 50 frames of a video, the reference image and the video itself compute groups of people who are close together in the specified target frame. Use spatial information from all frames as context — for example, to infer stable group memberships even if people temporarily move apart or come closer. Compute pairwise distances between people in the target frame and choose a reasonable grouping threshold based on the distribution of these distances. People belong to the same group if they are spatially close and consistently remain close across frames. Return only non-empty groups. Do not merge distant people into the same group. Do not hallucinate non-existent person_id. Reference Image and Video are given to support your decision."""
    
    image: dspy.Image = dspy.InputField(desc="Image with people to group")
    video: list[dspy.Image] = dspy.InputField(desc="Reference video to support the group decision")
    all_frames: list[list[dict]] = dspy.InputField(
        desc="List of frames, where each frame is a list of people detections. Each detection is a dict with keys: 'person_id', 'x', 'y', 'z'. Total ~50 frames."
    )   
    target_frame: int = dspy.InputField(
        desc="The (1-based) index of the frame for which groups should be computed."
    )   
    groups: list[list[int]] = dspy.OutputField(desc="Groups of person_ids who are close together in the target frame, inferred using spatial and temporal context from all frames.")


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

