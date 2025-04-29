import numpy as np


def convert_xyxy_to_cxcyah(bbox: list[int]) -> list[float]:
    # xyxy
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    aspect_ratio = width / (height + 1e-6)

    return [cx, cy, aspect_ratio, height]


def convert_cxcyah_to_xyxy(input_data: list[float]) -> list[int]:
    cx, cy, aspect_ratio, height = input_data

    width = aspect_ratio * height

    x_min = int(cx - width / 2)
    x_max = int(cx + width / 2)
    y_min = int(cy - height / 2)
    y_max = int(cy + height / 2)

    return [x_min, y_min, x_max, y_max]


class KalmanFilter(object):    
    def __init__(self):
        self.ndim = 8
        self.mdim = 4

        self.F = np.eye(self.ndim)
        for i in range(self.mdim):
            self.F[i, self.mdim + i] = 1.0

        self.H = np.eye(self.mdim, self.ndim)
        self.Q = np.eye(self.ndim) * 0.01
        self.R = np.eye(self.mdim) * 1.0

        self.P = np.eye(self.ndim) * 10.0
        
    def initiate(self, bbox: list[int]) -> None:
        """
        Initialize the Kalman filter with the first measurement.
        
        Args:
            measurement: numpy array of shape (4,) - [x, y, a, h]
        """
        measurement = convert_xyxy_to_cxcyah(bbox)

        # Initial state
        state = np.zeros(self.ndim)
        state[:self.mdim] = measurement
        
        # Initial covariance
        covariance = np.eye(self.ndim)
        covariance[self.mdim:, self.mdim:] *= 1000.0  # High uncertainty for velocities
        covariance *= 10.0

        self._saved_track_mean = state
        self._saved_track_cov = covariance
    
    def predict(self):
        """
        Run the Kalman filter prediction step.
        
        Args:
            mean: numpy array of shape (8,) - the previous state mean
            covariance: numpy array of shape (8, 8) - the previous state covariance
            
        Returns:
            The predicted state mean (8,) and covariance (8, 8)
        """
        mean = np.dot(self.F, self._saved_track_mean)
        covariance = np.dot(self.F, np.dot(self._saved_track_cov, self.F.T)) + self.Q

        self._saved_pred_mean = mean
        self._saved_pred_cov = covariance
        
        return convert_cxcyah_to_xyxy(mean[:4].tolist())
    
    def _project(self, mean, covariance):
        """
        Project state distribution to measurement space.
        
        Args:
            mean: numpy array of shape (8,) - the state mean
            covariance: numpy array of shape (8, 8) - the state covariance
            
        Returns:
            The projected measurement mean (4,) and covariance (4, 4)
        """
        measurement_mean = np.dot(self.H, mean)
        measurement_covariance = np.dot(self.H, np.dot(covariance, self.H.T)) + self.R
        
        return measurement_mean, measurement_covariance
    
    def update(self, bbox: list[int]):
        """
        Run the Kalman filter update step.
        
        Args:
            mean: numpy array of shape (8,) - the predicted state mean
            covariance: numpy array of shape (8, 8) - the predicted state covariance
            measurement: numpy array of shape (4,) - the measurement [x, y, a, h]
            
        Returns:
            The updated state mean (8,) and covariance (8, 8)
        """
        mean = self._saved_pred_mean
        covariance = self._saved_pred_cov
        measurement = convert_xyxy_to_cxcyah(bbox)

        measurement_mean, measurement_covariance = self._project(mean, covariance)
        
        kalman_gain = np.dot(covariance, np.dot(self.H.T, np.linalg.inv(measurement_covariance)))
        mean = mean + np.dot(kalman_gain, (measurement - measurement_mean))
    
        covariance = np.dot((np.eye(self.ndim) - np.dot(kalman_gain, self.H)), covariance)

        self._saved_track_mean = mean
        self._saved_track_cov = covariance


if __name__ == '__main__':
    kalman_filter = KalmanFilter()

    kalman_filter.initiate([0, 0, 100, 100])
    new_bbox = kalman_filter.predict()
    print(new_bbox)
    
    new_bbox = [10, 10, 110, 110]
    kalman_filter.update(new_bbox)
    new_bbox = kalman_filter.predict()
    print(new_bbox)

    new_bbox = [20, 20, 120, 120]
    kalman_filter.update(new_bbox)
    new_bbox = kalman_filter.predict()
    print(new_bbox)
