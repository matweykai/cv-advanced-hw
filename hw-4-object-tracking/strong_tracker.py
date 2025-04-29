from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from kalman_filter import KalmanFilter


@dataclass
class KalmanTrackObject:
    track_id: int
    last_seen: int
    kalman_filter: KalmanFilter


def calc_giou_distance(left_bboxes: np.ndarray, right_bboxes: np.ndarray) -> np.ndarray:
    eps = 1e-6

    # Calculate IOU
    x_left_min = left_bboxes[:, 0]  # Mx1
    y_left_min = left_bboxes[:, 1]  # Mx1
    x_left_max = left_bboxes[:, 2]  # Mx1
    y_left_max = left_bboxes[:, 3]  # Mx1

    x_right_min = right_bboxes[:, 0]  # Nx1
    y_right_min = right_bboxes[:, 1]  # Nx1
    x_right_max = right_bboxes[:, 2]  # Nx1
    y_right_max = right_bboxes[:, 3]  # Nx1

    x_min = np.maximum(x_left_min[:, None], x_right_min[None, :])   # MxN
    y_min = np.maximum(y_left_min[:, None], y_right_min[None, :])   # MxN
    x_max = np.minimum(x_left_max[:, None], x_right_max[None, :])   # MxN
    y_max = np.minimum(y_left_max[:, None], y_right_max[None, :])   # MxN

    intersection = np.maximum(x_max - x_min, np.zeros_like(x_max)) * np.maximum(y_max - y_min, np.zeros_like(x_max))
    left_bboxes_area = (x_left_max - x_left_min) * (y_left_max - y_left_min)        # Mx1
    right_bboxes_area = (x_right_max - x_right_min) * (y_right_max - y_right_min)   # Nx1
    union = left_bboxes_area[:, None] + right_bboxes_area[None, :] - intersection     # MxN

    iou = intersection / (union + eps) # MxN
    iou[union == 0] = 0

    # Calculate contour value
    x_min = np.minimum(x_left_min[:, None], x_right_min[None, :])   # MxN
    y_min = np.minimum(y_left_min[:, None], y_right_min[None, :])   # MxN
    x_max = np.maximum(x_left_max[:, None], x_right_max[None, :])   # MxN
    y_max = np.maximum(y_left_max[:, None], y_right_max[None, :])   # MxN

    contour_area = (x_max - x_min) * (y_max - y_min)

    giou = iou - (contour_area - union) / (contour_area + eps)
    giou = np.clip(giou, -1.0, 1.0)

    return 1 - giou


class StrongTracker:
    def __init__(self, forget_time: int, max_match_dist: float):
        # SORT + gIOU
        self._tracks_list: list[KalmanTrackObject] = []
        self._lost_tracks_list: list[KalmanTrackObject] = []
        self._frame_ind = 0
        self._forget_time = forget_time
        self._last_track_id = 0
        self._max_match_dist = max_match_dist

    def track(self, new_dets: list[list[int]]) -> list[int | None]:
        all_tracks_list = self._tracks_list + self._lost_tracks_list

        tracked_objects_bboxes = np.array([track.kalman_filter.predict() for track in all_tracks_list])
        new_bboxes = np.array(new_dets)
        result_ids = [None for _ in range(len(new_dets))]

        if len(tracked_objects_bboxes) != 0 and len(new_bboxes) != 0:
            # tracked_size x new_size
            tracked_new_dists = calc_giou_distance(tracked_objects_bboxes, new_bboxes)
            print("DISTANCE MATRIX", tracked_new_dists)
            rows_ind_list, cols_ind_list = linear_sum_assignment(tracked_new_dists, maximize=False)
        else:
            rows_ind_list = []
            cols_ind_list = []

        new_tracks_list = []
        new_lost_tracks_list = []

        taken_rows = []
        taken_cols = []

        for row_ind, col_ind in zip(rows_ind_list, cols_ind_list):
            if tracked_new_dists[row_ind, col_ind] < self._max_match_dist:
                track_object: KalmanTrackObject = all_tracks_list[row_ind]
                track_object.kalman_filter.update(new_dets[col_ind])
                track_object.last_seen = self._frame_ind
                result_ids[col_ind] = track_object.track_id

                new_tracks_list.append(track_object)
                taken_rows.append(row_ind)
                taken_cols.append(col_ind)

        rows_ind_list = taken_rows
        cols_ind_list = taken_cols

        # Add lost tracks
        for row_ind in set(range(len(all_tracks_list))) - set(rows_ind_list):
            new_lost_tracks_list.append(all_tracks_list[row_ind])

        # Add new tracks
        for col_ind in set(range(len(new_dets))) - set(cols_ind_list):
            track_object = KalmanTrackObject(
                track_id=self._last_track_id + 1,
                last_seen=self._frame_ind,
                kalman_filter=KalmanFilter(),
            )

            track_object.kalman_filter.initiate(new_dets[col_ind])
            self._last_track_id += 1
            
            new_tracks_list.append(track_object)
            result_ids[col_ind] = track_object.track_id

        self._tracks_list = new_tracks_list
        # Clean old tracks
        self._lost_tracks = [lost_track for lost_track in new_lost_tracks_list if lost_track.last_seen >= self._frame_ind - self._forget_time]

        return result_ids
