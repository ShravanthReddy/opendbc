from dataclasses import dataclass

import numpy as np


NUM_INDICES = 10
OFFSETS_PER_INDEX = 4
NUM_POINTS = NUM_INDICES * OFFSETS_PER_INDEX

# Four redundant banks of ten slots, observed on the stock Accord 11G radar.
MUX_CYCLE = tuple(index + bank * 16 for bank in range(4) for index in range(1, NUM_INDICES + 1))

OFFSET_UNAVAILABLE = 2047
OFFSET_VALID_MAX = 2046
CANFD_MIN_VALID_POINTS = 6
CANFD_MAX_VALID_POINTS = 30
CANFD_IDLE_OFFSETS = [0] * CANFD_MIN_VALID_POINTS + [OFFSET_UNAVAILABLE] * (NUM_POINTS - CANFD_MIN_VALID_POINTS)

LOOKAHEAD = np.linspace(2.0, 100.0, NUM_POINTS)
GAIN = 6.27 + 0.0106 * LOOKAHEAD + 0.000354 * LOOKAHEAD ** 2

PATH_PROB_ON = 0.25
PATH_PROB_OFF = 0.10
HALF_LANE_WIDTH = 1.65
SPEED_POINT_OFFSET = 6.75
POINTS_PER_MS = 0.875
FULL_LENGTH_LEAD_DISTANCE = 70.0
LKAS_STATE_CHANGE_FRAMES = 30


@dataclass
class DashLane:
  offsets: list[int]
  reach: float
  left_line: bool
  right_line: bool


def blank_dash_lane() -> DashLane:
  return DashLane([OFFSET_UNAVAILABLE] * NUM_POINTS, 0.0, False, False)


def _line_trusted(probability: float, was_on: bool) -> bool:
  return probability >= (PATH_PROB_OFF if was_on else PATH_PROB_ON)


def _select_lane_center(model, previous_left: bool, previous_right: bool):
  lane_lines = model.laneLines
  probabilities = model.laneLineProbs
  if len(lane_lines) < 3 or len(probabilities) < 3:
    return None

  left_x = np.asarray(lane_lines[1].x, dtype=float)
  left_y = np.asarray(lane_lines[1].y, dtype=float)
  right_x = np.asarray(lane_lines[2].x, dtype=float)
  right_y = np.asarray(lane_lines[2].y, dtype=float)
  if min(left_x.size, left_y.size, right_x.size, right_y.size) < 2:
    return None
  if not (left_x.size == left_y.size == right_x.size == right_y.size):
    return None
  if not all(np.all(np.isfinite(values)) for values in (left_x, left_y, right_x, right_y)):
    return None

  left_on = _line_trusted(float(probabilities[1]), previous_left)
  right_on = _line_trusted(float(probabilities[2]), previous_right)
  if left_on and right_on:
    center_y = (left_y + right_y) / 2.0
  elif left_on:
    center_y = left_y + HALF_LANE_WIDTH
  elif right_on:
    center_y = right_y - HALF_LANE_WIDTH
  else:
    return None

  return left_x, center_y, left_on, right_on


def encode_lane_path(x, y) -> list[int]:
  x = np.asarray(x, dtype=float)
  y = np.asarray(y, dtype=float)
  if x.size < 2 or y.size != x.size or not np.all(np.diff(x) >= 0.0) or x[-1] < LOOKAHEAD[-1]:
    return [OFFSET_UNAVAILABLE] * NUM_POINTS
  raw = np.clip(np.round(-GAIN * np.interp(LOOKAHEAD, x, y)), -OFFSET_VALID_MAX, OFFSET_VALID_MAX)
  return [int(value) for value in raw]


def canfd_lane_length(dash_lane: DashLane) -> int:
  if dash_lane.reach <= 0.0 or dash_lane.offsets[0] == OFFSET_UNAVAILABLE:
    return CANFD_MIN_VALID_POINTS
  return max(CANFD_MIN_VALID_POINTS, min(CANFD_MAX_VALID_POINTS, round(dash_lane.reach * CANFD_MAX_VALID_POINTS)))


def canfd_lane_offsets(dash_lane: DashLane) -> list[int]:
  if dash_lane.reach <= 0.0 or dash_lane.offsets[0] == OFFSET_UNAVAILABLE:
    return list(CANFD_IDLE_OFFSETS)
  valid_points = canfd_lane_length(dash_lane)
  return list(dash_lane.offsets[:valid_points]) + [OFFSET_UNAVAILABLE] * (NUM_POINTS - valid_points)


def create_lane_path(packer, bus: int, offsets: list[int], mux: int):
  base = ((mux - 1) % 16) * OFFSETS_PER_INDEX
  values = {
    "MUX": mux,
    "PATH_OFFSET_1": offsets[base],
    "PATH_OFFSET_2": offsets[base + 1],
    "PATH_OFFSET_3": offsets[base + 2],
    "PATH_OFFSET_4": offsets[base + 3],
  }
  return packer.make_can_msg("LANE_PATH", bus, values)


class LanePathFitter:
  def __init__(self):
    self._left_on = False
    self._right_on = False

  def update(self, model, v_ego: float, lead_distance: float) -> DashLane:
    if model is None or not np.isfinite(v_ego) or not np.isfinite(lead_distance):
      self._left_on = False
      self._right_on = False
      return blank_dash_lane()

    selected = _select_lane_center(model, self._left_on, self._right_on)
    if selected is None:
      self._left_on = False
      self._right_on = False
      return blank_dash_lane()

    x, y, self._left_on, self._right_on = selected
    offsets = encode_lane_path(x, y)
    if offsets[0] == OFFSET_UNAVAILABLE:
      return blank_dash_lane()

    # Fitted from 290 matched, active, no-lead Accord sweeps:
    # valid_points ~= 6.75 + 0.875 * v_ego. A distant lead can extend the
    # path to the full 30-point stock prefix.
    speed_points = SPEED_POINT_OFFSET + POINTS_PER_MS * max(v_ego, 0.0)
    lead_points = lead_distance / FULL_LENGTH_LEAD_DISTANCE * CANFD_MAX_VALID_POINTS
    valid_points = float(np.clip(max(speed_points, lead_points), CANFD_MIN_VALID_POINTS, CANFD_MAX_VALID_POINTS))
    reach = valid_points / CANFD_MAX_VALID_POINTS
    return DashLane(offsets, reach, self._left_on, self._right_on)


class LkasStateChangePulse:
  def __init__(self):
    self._key = None
    self._remaining = 0

  def update(self, key) -> bool:
    if self._key is None:
      self._key = key
      self._remaining = LKAS_STATE_CHANGE_FRAMES
    elif key != self._key:
      self._key = key
      self._remaining = LKAS_STATE_CHANGE_FRAMES

    active = self._remaining > 0
    self._remaining = max(0, self._remaining - 1)
    return active
