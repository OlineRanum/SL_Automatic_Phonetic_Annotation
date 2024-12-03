from collections import Counter

def detect_transitions(handshapes, transition_window=25, change_threshold=3):
    """
    Detect rapid transitions in handshape labels and mark them as 'Transition'.

    Parameters:
        handshapes (list of str): The sequence of handshapes.
        transition_window (int): The size of the window to check for rapid changes.
        change_threshold (int): The number of changes within the window that indicates a transition.

    Returns:
        list of str: The sequence with transitions marked as 'Transition'.
    """
    length = len(handshapes)
    labeled_handshapes = handshapes.copy()

    for i in range(length):
        # Define the window range
        start = max(0, i - transition_window // 2)
        end = min(length, i + transition_window // 2 + 1)

        # Get the labels in the current window
        window_labels = handshapes[start:end]
        
        # Count the occurrences of each label in the window
        label_counts = Counter(window_labels)

        # Find the main cluster label (the most common label in the window)
        main_cluster_label, _ = label_counts.most_common(1)[0]

        # If the current label is not part of the main cluster and there are rapid changes, mark as 'Transition'
        unique_labels = len(set(window_labels))
        if unique_labels > change_threshold and handshapes[i] != main_cluster_label:
            labeled_handshapes[i] = 'Transition'

    return labeled_handshapes

def label_smoothing_and_transitions(handshapes, window_size=15, recurrence_threshold=0.5):
    """
    Smooth handshape labels using a sliding window approach to identify recurring patterns,
    while also marking rapid transitions as 'Transition'.

    Parameters:
        handshapes (list of str): The sequence of handshapes.
        window_size (int): The size of the sliding window for smoothing.
        recurrence_threshold (float): The proportion of times a label must appear in the window to be considered stable.

    Returns:
        list of str: The smoothed sequence of handshapes.
    """
    # Step 1: Detect transitions and mark them as 'Transition'
    handshapes_with_transitions = handshapes #detect_transitions(handshapes)

    # Step 2: Smooth the handshapes while preserving 'Transition' labels
    half_window = window_size // 2
    smoothed_handshapes = handshapes_with_transitions.copy()
    length = len(handshapes_with_transitions)

    for i in range(length):
        if handshapes_with_transitions[i] == 'Transition':
            # Skip smoothing for transition labels
            continue

        # Define the window range
        start = max(0, i - half_window)
        end = min(length, i + half_window + 1)

        # Get the labels in the current window
        window_labels = handshapes_with_transitions[start:end]

        # Count the occurrences of each label
        label_counts = Counter(window_labels)

        # Remove 'Transition' from the counts, as we do not want it to influence smoothing
        if 'Transition' in label_counts:
            del label_counts['Transition']

        # Determine if there is a recurring pattern in the window
        if label_counts:
            most_common_label, most_common_count = label_counts.most_common(1)[0]
            recurrence_ratio = most_common_count / len(window_labels)

            # Replace the label if the most common label appears frequently enough
            if recurrence_ratio >= recurrence_threshold:
                smoothed_handshapes[i] = most_common_label

    return smoothed_handshapes