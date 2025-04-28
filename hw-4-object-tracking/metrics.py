from models import FramePrediction


class TrackerMetricCollector:
    def __init__(self) -> None:
        self._tracked_objects_list: list[FramePrediction] = []

    def update(self, pred_obj: FramePrediction):
        self._tracked_objects_list.append(pred_obj)

    def calculate(self):
        # Calculate switches of ids
        gt_to_pred_id_map = dict()
        switches_count = 0
        gt_count = 0

        for pred_obj in self._tracked_objects_list:
            for det in pred_obj.data:
                if len(det.bounding_box) != 0:
                    gt_track = det.cb_id
                    pred_track = det.track_id

                    if gt_track not in gt_to_pred_id_map:
                        gt_to_pred_id_map[gt_track] = pred_track

                    if gt_to_pred_id_map[gt_track] != pred_track:
                        switches_count += 1
                        gt_to_pred_id_map[gt_track] = pred_track
                    
                    gt_count += 1

        return 1 - switches_count / gt_count
