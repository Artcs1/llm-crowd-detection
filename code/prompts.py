import dspy

class IdentifyGroups(dspy.Signature):
    """Given a list of people with their 3D positions, group them into sets where each set contains people who are close to each other in space. Compute all pairwise distances between people. Choose a reasonable grouping threshold based on the distribution of these distances. People belong to the same group if their pairwise distances are below this threshold. Return only non-empty groups. Do not merge distant people into the same group. Do not hallucinate non-existent person_id."""

    detections: list[dict] = dspy.InputField(desc="List of people, where each dictionary has keys: 'person_id', 'x', 'y', 'z'.")
    groups: list[list[int]] = dspy.OutputField(desc="A list of groups, where each group is a list of person_ids that are close together. All ids should appear at least once.")


class IdentifyGroups_AllFrames(dspy.Signature):
    """Given detections of people with their 3D positions in a single video frame, compute groups of people who are close together. Compute pairwise distances between people and choose a reasonable grouping threshold based on the distribution of these distances. People belong to the same group if they are spatially close. Return only non-empty groups. Do not merge distant people into the same group. Do not hallucinate non-existent person_id."""
    all_frames: list[list[dict]] = dspy.InputField(
        desc="List containing a single list of people detections for the target frame. Each detection is a dict with keys: 'person_id', 'x', 'y', 'z', and 'movement_direction' (their net movement direction across the frames leading up to this one, or 'stationary')."
    )
    target_frame: int = dspy.InputField(
        desc="The 1-based index of the frame these detections belong to."
    )
    groups: list[list[int]] = dspy.OutputField(desc="Groups of person_ids who are close together in the target frame. All ids should appear at least once.")

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


class vlm_IdentifyGroupsImage_idsonly(dspy.Signature):
    """Given an image with people annotated by bounding boxes and id labels, group the given
    person_ids into sets where each set contains people who are close to each other in the image.
    Use only the visual positions of the labeled boxes to judge proximity — no numeric coordinates
    are provided. Return only non-empty groups. Do not merge distant people into the same group."""
    image: dspy.Image = dspy.InputField(desc="Image with bounding boxes and id labels drawn on each person")
    person_ids: list[int] = dspy.InputField(desc="List of person_ids visible in the image, to be grouped by their visual positions")
    groups: list[list[int]] = dspy.OutputField(desc="A list of groups, where each group is a list of person_ids that are close together in the image. All ids should appear at least once.")


class vlm_IdentifyGroupsImage(dspy.Signature):
    """Given a list of people with their 3D positions, group them into sets where each set contains people who are close to each other in space. Compute all pairwise distances between people. Choose a reasonable grouping threshold based on the distribution of these distances. People belong to the same group if their pairwise distances are below this threshold. Return only non-empty groups. Do not merge distant people into the same group."""
    image: dspy.Image = dspy.InputField(desc="Image with people to group")
    detections: list[dict] = dspy.InputField(desc="List of people, where each dictionary has keys: 'person_id', 'x', 'y', 'z'.")
    groups: list[list[int]] = dspy.OutputField(desc="A list of groups, where each group is a list of person_ids that are close together. All ids should appear at least once.")


class vlm_IdentifyGroupsText(dspy.Signature):
    """Given a list of people with their 3D positions, group them into sets where each set contains people who are close to each other in space. Compute all pairwise distances between people. Choose a reasonable grouping threshold based on the distribution of these distances. People belong to the same group if their pairwise distances are below this threshold. Return only non-empty groups. Do not merge distant people into the same group."""
    detections: list[dict] = dspy.InputField(desc="List of people, where each dictionary has keys: 'person_id', 'x', 'y', 'z'.")
    groups: list[list[int]] = dspy.OutputField(desc="A list of groups, where each group is a list of person_ids that are close together. All ids should appear at least once.")

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
        desc="List of string containing a sentence associated with the group."
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
    """Given detections of people with their 3D positions in a single video frame, compute groups of people who are close together. Compute pairwise distances between people and choose a reasonable grouping threshold based on the distribution of these distances. People belong to the same group if they are spatially close. Return only non-empty groups. Do not merge distant people into the same group. Do not hallucinate non-existent person_id."""
    all_frames: list[list[dict]] = dspy.InputField(
        desc="List containing a single list of people detections for the target frame. Each detection is a dict with keys: 'person_id', 'x', 'y', 'z', and 'movement_direction' (their net movement direction across the frames leading up to this one, or 'stationary')."
    )
    target_frame: int = dspy.InputField(
        desc="The (1-based) index of the frame these detections belong to."
    )
    groups: list[list[int]] = dspy.OutputField(desc="Groups of person_ids who are close together in the target frame. All ids should appear at least once.")


class vlm_IdentifyGroups_AllFramesImage(dspy.Signature):
    """Given the detections of people with their 3D positions in the target frame, a reference image of that frame, and a video of the frames leading up to it, compute groups of people who are close together in the target frame. Use the video to infer stable group memberships over time — people who temporarily move apart or come closer should still be grouped together if they consistently stay close across the video. Compute pairwise distances between people in the target frame and choose a reasonable grouping threshold based on the distribution of these distances. Return only non-empty groups. Do not merge distant people into the same group. Do not hallucinate non-existent person_id."""

    image: dspy.Image = dspy.InputField(desc="Image with people to group")
    video: list[dspy.Image] = dspy.InputField(desc="Frames 1 through the target frame (subsampled for long sequences), giving temporal context")
    all_frames: list[list[dict]] = dspy.InputField(
        desc="List containing a single list of people detections for the target frame only. Each detection is a dict with keys: 'person_id', 'x', 'y', 'z', and 'movement_direction' (their net movement direction across the frames leading up to this one, or 'stationary')."
    )
    target_frame: int = dspy.InputField(
        desc="The (1-based) index of the frame for which groups should be computed — matches the last frame of video and the frame shown in image."
    )
    groups: list[list[int]] = dspy.OutputField(desc="Groups of person_ids who are close together in the target frame, inferred using spatial context from the target frame and temporal context from the video. All ids should appear at least once.")


class vlm_IdentifyGroups_AllFramesImage_withbboxes(dspy.Signature):
    """Given the detections and pixel bounding boxes of people in the target frame, a reference
    image of that frame, and a video of the frames leading up to it, compute groups of people who
    are close together in the target frame. Use the video to infer stable group memberships over
    time — people who temporarily move apart or come closer should still be grouped together if
    they consistently stay close across the video. Compute pairwise distances between people in
    the target frame and choose a reasonable grouping threshold based on the distribution of
    these distances. Return only non-empty groups. Do not merge distant people into the same
    group. Do not hallucinate non-existent person_id."""

    image: dspy.Image = dspy.InputField(desc="Image with people to group")
    video: list[dspy.Image] = dspy.InputField(desc="Frames 1 through the target frame (subsampled for long sequences), giving temporal context")
    boundingboxes: list[dict] = dspy.InputField(desc="List of people in the target frame, where each dictionary has keys in the standard top-left bottom-right notation [t, l, b, r]")
    detections: list[dict] = dspy.InputField(desc="List of people in the target frame, where each dictionary has keys: 'person_id', 'x', 'y', 'z', and 'movement_direction' (their net movement direction across the frames leading up to this one, or 'stationary')")
    target_frame: int = dspy.InputField(desc="The (1-based) index of the frame for which groups should be computed — matches the last frame of video and the frame shown in image.")
    groups: list[list[int]] = dspy.OutputField(desc="Groups of person_ids who are close together in the target frame, inferred using spatial context from the target frame and temporal context from the video. All ids should appear at least once.")


class vlm_IdentifyGroups_AllFramesImage_idsonly(dspy.Signature):
    """Given an image with people in the target frame annotated by bounding boxes and id labels,
    a video of the frames leading up to it, and the list of person_ids to group, compute groups
    of people who are close together in the target frame. Use only the visual positions of the
    labeled boxes and the video for temporal context to judge proximity — no numeric coordinates
    are provided. Return only non-empty groups. Do not merge distant people into the same group."""

    image: dspy.Image = dspy.InputField(desc="Image with people in the target frame, bounding boxes and id labels drawn on each person")
    video: list[dspy.Image] = dspy.InputField(desc="Frames 1 through the target frame (subsampled for long sequences), giving temporal context")
    person_ids: list[int] = dspy.InputField(desc="List of person_ids visible in the target frame, to be grouped by their visual positions")
    target_frame: int = dspy.InputField(desc="The (1-based) index of the frame for which groups should be computed — matches the last frame of video and the frame shown in image.")
    groups: list[list[int]] = dspy.OutputField(desc="Groups of person_ids who are close together in the target frame, inferred using visual context from the image and temporal context from the video. All ids should appear at least once.")


class IdentifyGroups_Last5Frames(dspy.Signature):
    """Given detections of people with their 3D positions across past five frames of a video, compute groups of people who are close together in the specified target frame. Use spatial information from all frames as context — for example, to infer stable group memberships even if people temporarily move apart or come closer. Compute pairwise distances between people in the target frame and choose a reasonable grouping threshold based on the distribution of these distances. People belong to the same group if they are spatially close and consistently remain close across frames. Return only non-empty groups. Do not merge distant people into the same group. Do not hallucinate non-existent person_id."""
    past_frames: list[list[dict]] = dspy.InputField(
        desc="List of five frames, where each frame is a list of people detections. Each detection is a dict with keys: 'person_id', 'x', 'y', 'z'."
    )
    target_frame: dict = dspy.InputField(
        desc="Target frame with a list of people detections. Each detection is a dict with keys: 'person_id', 'x', 'y', 'z'."
    )
    groups: list[list[int]] = dspy.OutputField(desc="Groups of person_ids who are close together in the target frame, inferred using spatial and temporal context from all frames. All ids should appear at least once.")


class RecognizeGroupActivity(dspy.Signature):
    """Given an image with multiple people and the 2D coordinates of a bounding box enclosing a subset of them, name the activity (or activities) that people inside the bounding box are engaged in. Consider their poses, interactions, and any objects they might be using."""
    image: dspy.Image = dspy.InputField(desc="Image with people")
    bbox: list[int] = dspy.InputField(desc="Bounding box around a group of people, in top-left and bottom-right notation: [x1, y1, x2, y2]")
    output: list[str] = dspy.OutputField(desc="Name of one or more activities that people inside the bounding box are engaged in.")


class RecognizeGroupClothing(dspy.Signature):
    """Given an image with multiple people and the 2D coordinates of a bounding box enclosing a subset of them, name the clothing and accessories that people inside the bounding box are wearing."""
    image: dspy.Image = dspy.InputField(desc="Image with people")
    bbox: list[int] = dspy.InputField(desc="Bounding box around a group of people, in top-left and bottom-right notation: [x1, y1, x2, y2]")
    output: list[str] = dspy.OutputField(desc="Name of one or more clothing and accessories that people inside the bounding box are wearing.")


class RecognizeGroupHandholding(dspy.Signature):
    """Given an image with multiple people and the 2D coordinates of a bounding box enclosing a subset of them, answer if they are holding hands."""
    image: dspy.Image = dspy.InputField(desc="Image with people")
    bbox: list[int] = dspy.InputField(desc="Bounding box around a group of people, in top-left and bottom-right notation: [x1, y1, x2, y2]")
    output: bool = dspy.OutputField(desc="True or False, answer if they are holding hands.")


class RecognizeGroupHugging(dspy.Signature):
    """Given an image with multiple people and the 2D coordinates of a bounding box enclosing a subset of them, answer if they are hugging or holding each other."""
    image: dspy.Image = dspy.InputField(desc="Image with people")
    bbox: list[int] = dspy.InputField(desc="Bounding box around a group of people, in top-left and bottom-right notation: [x1, y1, x2, y2]")
    output: bool = dspy.OutputField(desc="True or False, answer if they are hugging or holding each other.")



