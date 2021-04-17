# vim: expandtab:ts=4:sw=4

import numpy as np
from deep_sort_pytorch.utils.tools  import objid2name
import collections


class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted   = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    """
    計画：
    ・トラックに向きをもたせる
    ・トラックに座標をもたせる
    ・ageが一定になったらカウントを上げる（ageは１つづつ大きくなるか確認）
    ・カウントクラスを作成する。（向きｘ車種）
    ・トラッカーとトラックのどちらでいろいろ比較して更新するか決める=> update in update

    """

    def __init__(self, mean, covariance, track_id, n_init, max_age, n_count, road_direction, obj_cls, corrdinate, feature=None):
        self.road_direction = road_direction

        self.mean = mean
        self.covariance = covariance
        self.track_id   = track_id
        self.hits = 1
        self.age  = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init  = n_init
        self._max_age = max_age

        # here
        self.n_count      = n_count
        self.obj_cls      = obj_cls
        self.obj_cls_list = [obj_cls]
        self.direction    = 0
        self.corrdinate   = corrdinate
        self.is_detected  = False

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2]  *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def increment_age(self):
        self.age += 1
        self.time_since_update += 1

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.increment_age()

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """

        self.obj_cls_list.append(detection.obj_cls)
        c = collections.Counter(self.obj_cls_list)

        most_voted = c.most_common()[0]
        try:
            next_voted = c.most_common()[1]
        except:
            pass
        else:
            # 車をトラックと誤認識された場合に修正するため
            if most_voted[0] == 7 and next_voted[0] == 2:
                if next_voted[1] * 1.2 > most_voted[1]:
                    most_voted, next_voted = next_voted, most_voted



        # 毎フレーム多数決してカウントを修正するため
        old_cls = self.obj_cls
        self.obj_cls = most_voted[0]

        return_list = list()
        return_list.extend(_move(self, detection))

        if self.is_detected == True:
            if self.obj_cls != old_cls:
                d_tmp = "right" if self.direction > 0 else "left"
                return_list.extend([objid2name(old_cls), d_tmp])



        self.mean, self.covariance = kf.update(self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

        return return_list

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if  self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

# here~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def _move(self, detection):
    move_right = (detection.tlwh[0] - self.corrdinate[0])
    move_down  = (detection.tlwh[1] - self.corrdinate[1])

    if self.road_direction == "left_is_up":
        right_count_condition = (5 < move_right) or (move_down < -5)
        left_count_condition  = (move_right < -5) or (5 < move_down)
    else:
        right_count_condition = (5 < move_right)  or (5 < move_down)
        left_count_condition  = (move_right < -5) or (move_down < -5)
    
    if right_count_condition:
        self.direction = min(self.direction + 1, self.n_count) 

    elif left_count_condition:
        self.direction = max(self.direction - 1, -self.n_count) 
    else:
        return objid2name(self.obj_cls), None

    self.corrdinate = detection.tlwh[:2]
    d_tmp = None
    if not self.is_detected: 
        if self.direction == self.n_count:
            d_tmp = "right"
            self.is_detected = True
        elif self.direction == -self.n_count:
            d_tmp = "left"
            self.is_detected = True

    if self.direction == 0:
        self.is_detected = False

    if abs(move_right) > 500 or abs(move_down) > 500:
        self.direction = 0
        self.is_detected = False
    return objid2name(self.obj_cls), d_tmp
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~