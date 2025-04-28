from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class TrackObject:
    track_id: int
    last_seen: int
    last_bbox: list[int]


def calc_distance(left_bboxes: np.ndarray, right_bboxes: np.ndarray) -> np.ndarray:
    # Boxes in xyxy format

    # left_bboxes Nx4
    # right_bboxes Mx4
    left_cent = np.zeros((left_bboxes.shape[0], 2))     # Nx2

    left_cent[:, 0] = (left_bboxes[:, 2] + left_bboxes[:, 0]) / 2
    left_cent[:, 1] = (left_bboxes[:, 3] + left_bboxes[:, 1]) / 2

    right_cent = np.zeros((right_bboxes.shape[0], 2))   # Mx2

    right_cent[:, 0] = (right_bboxes[:, 2] + right_bboxes[:, 0]) / 2
    right_cent[:, 1] = (right_bboxes[:, 3] + right_bboxes[:, 1]) / 2

    dist = (right_cent[None, :, :] - left_cent[:, None, :]) ** 2    # NxMx2
    dist = dist.sum(-1)
    
    return dist ** 0.5


class SoftTracker:
    def __init__(self, forget_time: int, max_match_dist: float):
        self._tracks_list: list[TrackObject] = []
        self._lost_tracks: list[TrackObject] = []
        self._frame_ind = 0
        self._forget_time = forget_time
        self._last_track_id = 0
        self._max_match_dist = max_match_dist

    def track(self, new_dets: list[list[int]]):
        self._frame_ind += 1

        print('NEW DETS VALUES', new_dets)

        all_tracks = self._tracks_list + self._lost_tracks
        tracked_objects_bboxes = np.array([track.last_bbox for track in all_tracks])
        new_bboxes = np.array(new_dets)
        result_ids = [None for _ in range(len(new_dets))]
        
        if len(tracked_objects_bboxes) != 0 and len(new_bboxes) != 0:
            # tracked_size x new_size
            tracked_new_dists = calc_distance(tracked_objects_bboxes, new_bboxes)
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
                track_object = all_tracks[row_ind]
                track_object.last_bbox = new_dets[col_ind]
                track_object.last_seen = self._frame_ind
                result_ids[col_ind] = track_object.track_id

                new_tracks_list.append(track_object)
                taken_rows.append(row_ind)
                taken_cols.append(col_ind)

        rows_ind_list = taken_rows
        cols_ind_list = taken_cols

        # Add lost tracks
        for row_ind in set(range(len(all_tracks))) - set(rows_ind_list):
            new_lost_tracks_list.append(all_tracks[row_ind])

        # Add new tracks
        for col_ind in set(range(len(new_dets))) - set(cols_ind_list):
            track_object = TrackObject(
                track_id=self._last_track_id + 1,
                last_seen=self._frame_ind,
                last_bbox=new_dets[col_ind],
            )

            self._last_track_id += 1
            
            new_tracks_list.append(track_object)
            result_ids[col_ind] = track_object.track_id

        self._tracks_list = new_tracks_list
        # Clean old tracks
        self._lost_tracks = [lost_track for lost_track in new_lost_tracks_list if lost_track.last_seen >= self._frame_ind - self._forget_time]

        return result_ids
