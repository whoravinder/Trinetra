import numpy as np


class PolarKalmanTracker:
    """Constant-rate Kalman filter for range, radial velocity and bearing.

    State: [range_m, radial_velocity_mps, bearing_deg, bearing_rate_deg_s].
    Measurements: [range_m, radial_velocity_mps, bearing_deg].
    """

    def __init__(self, dt=0.1, range_std=8.0, velocity_std=2.5, bearing_std=3.0):
        self.dt = float(dt)
        self.x = None
        self.P = None

        dt = self.dt
        self.F = np.array([
            [1.0, dt, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, dt],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=float)

        # Small process noise allows the track to follow maneuvering targets.
        q_range = 2.0
        q_bearing = 0.5
        self.Q = np.diag([
            q_range ** 2,
            1.0 ** 2,
            q_bearing ** 2,
            0.25 ** 2,
        ])
        self.R = np.diag([
            range_std ** 2,
            velocity_std ** 2,
            bearing_std ** 2,
        ])

    @staticmethod
    def _wrap_angle(angle):
        return (angle + 180.0) % 360.0 - 180.0

    def initialize(self, measurement):
        r, vr, bearing = map(float, measurement)
        self.x = np.array([r, vr, bearing, 0.0], dtype=float)
        self.P = np.diag([25.0, 9.0, 16.0, 4.0])
        return self.x.copy()

    def update(self, measurement):
        z = np.asarray(measurement, dtype=float)
        if z.shape != (3,):
            raise ValueError("measurement must be [range_m, radial_velocity_mps, bearing_deg]")

        if self.x is None:
            return self.initialize(z)

        # Predict.
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Innovation with circular bearing residual.
        innovation = z - self.H @ self.x
        innovation[2] = self._wrap_angle(innovation[2])
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.x[2] = self._wrap_angle(self.x[2])
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.P = (self.P + self.P.T) / 2.0
        return self.x.copy()

    @property
    def uncertainty(self):
        if self.P is None:
            return None
        return np.sqrt(np.maximum(np.diag(self.P), 0.0))
