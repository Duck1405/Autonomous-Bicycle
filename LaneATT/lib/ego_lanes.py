import numpy as np

def get_ego_lanes2(img_w, predictions):
    if predictions is None or len(predictions) < 2:
        return None, None, None, None


    mid_point = img_w / 2
    # print(f"predictions: {predictions}")
    length = len(predictions)
    # print(f"length: {length}")
    y_min = np.zeros(length)
    y_max = np.zeros(length)

    for i, lane in enumerate(predictions):

        if lane is None or len(lane) < 3:
            return None, None, None, None

        y = lane[:, 1]

        y_min[i] = np.min(y)
        y_max[i] = np.max(y)

    highest_min = np.max(y_min)
    lowest_max = np.min(y_max)


    if highest_min >= lowest_max:
        return None, None, None, None


    y_values = np.linspace(
        highest_min,
        lowest_max,
        100
    )


    left_candidates = []
    right_candidates = []

    for lane_index, lane in enumerate(predictions):

        x = lane[:, 0]
        y = lane[:, 1]
        mask = (
            (y >= highest_min) &
            (y <= lowest_max)
        )

        filtered_lane = lane[mask]


        if len(filtered_lane) < 3:
            continue


        filtered_x = filtered_lane[:, 0]
        filtered_y = filtered_lane[:, 1]


        # x = f(y)
        coefficients = np.polyfit(
            filtered_y,
            filtered_x,
            2
        )
        x_values = np.polyval(
            coefficients,
            y_values
        )


        resampled_lane = np.column_stack((
            x_values,
            y_values
        ))

        x_difference = (
            x_values - mid_point
        )


        signed_average = np.mean(
            x_difference
        )
        average_distance = np.mean(
            np.abs(x_difference)
        )


        candidate = {
            "index": lane_index,
            "distance": average_distance,
            "signed_average": signed_average,
            "points": resampled_lane,
            "coefficients": coefficients
        }


        if signed_average < 0:
            left_candidates.append(candidate)

        else:
            right_candidates.append(candidate)


    if not left_candidates or not right_candidates:
        return None, None, None, None



    closest_left = min(
        left_candidates,
        key=lambda lane: lane["distance"]
    )

    closest_right = min(
        right_candidates,
        key=lambda lane: lane["distance"]
    )


    left_points = closest_left["points"]
    right_points = closest_right["points"]

    middle_x = (
        left_points[:, 0]
        + right_points[:, 0]
    ) / 2


    mid_points = np.column_stack((
        middle_x,
        y_values
    ))


    synthesized = None

    left_points = np.round(
        left_points
    ).astype(int)

    right_points = np.round(
        right_points
    ).astype(int)

    mid_points = np.round(
        mid_points
    ).astype(int)


    return (
        left_points,
        right_points,
        mid_points,
        synthesized
    )

import numpy as np

def get_ego_lanes3(img_w, predictions):
    if predictions is None or len(predictions) < 2:
        return None, None, None, None


    mid_point = img_w / 2
    # print(f"predictions: {predictions}")
    length = len(predictions)
    # print(f"length: {length}")
    y_min = np.zeros(length)
    y_max = np.zeros(length)

    for i, lane in enumerate(predictions):

        if lane is None or len(lane) < 3:
            return None, None, None, None

        y = lane[:, 1]

        y_min[i] = np.min(y)
        y_max[i] = np.max(y)

    highest_min = np.max(y_min)
    lowest_max = np.min(y_max)


    if highest_min >= lowest_max:
        return None, None, None, None


    y_values = np.linspace(
        highest_min,
        lowest_max,
        100
    )


    left_candidates = []
    right_candidates = []

    for lane_index, lane in enumerate(predictions):

        x = lane[:, 0]
        y = lane[:, 1]
        mask = (
            (y >= highest_min) &
            (y <= lowest_max)
        )

        filtered_lane = lane[mask]


        if len(filtered_lane) < 3:
            continue


        filtered_x = filtered_lane[:, 0]
        filtered_y = filtered_lane[:, 1]


        # x = f(y)
        coefficients = np.polyfit(
            filtered_y,
            filtered_x,
            2
        )
        x_values = np.polyval(
            coefficients,
            y_values
        )


        resampled_lane = np.column_stack((
            x_values,
            y_values
        ))

        x_difference = (
            x_values - mid_point
        )


        signed_average = np.mean(
            x_difference
        )
        average_distance = np.mean(
            np.abs(x_difference)
        )


        candidate = {
            "index": lane_index,
            "distance": average_distance,
            "signed_average": signed_average,
            "points": resampled_lane,
            "coefficients": coefficients
        }


        if signed_average < 0:
            left_candidates.append(candidate)

        else:
            right_candidates.append(candidate)


    if not left_candidates or not right_candidates:
        return None, None, None, None



    closest_left = min(
        left_candidates,
        key=lambda lane: lane["distance"]
    )

    closest_right = min(
        right_candidates,
        key=lambda lane: lane["distance"]
    )


    left_points = closest_left["points"]
    right_points = closest_right["points"]

    middle_x = (
        left_points[:, 0]
        + right_points[:, 0]
    ) / 2


    mid_points = np.column_stack((
        middle_x,
        y_values
    ))


    synthesized = None

    left_points = np.round(
        left_points
    ).astype(int)

    right_points = np.round(
        right_points
    ).astype(int)

    mid_points = np.round(
        mid_points
    ).astype(int)


    return (
        left_points,
        right_points,
        mid_points,
        synthesized
    )
