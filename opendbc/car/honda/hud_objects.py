import math
from dataclasses import dataclass

import numpy as np


INACTIVE_OBJECT = {
  "OBJECT_ID": 0,
  "IS_LEAD_CAR": 0,
  "CAR_TYPE": -1,
  "ROTATION": -128,
  "LONG_DIST": 196.9,
  "LAT_DIST": 204.7,
}

LEAD_PROBABILITY_MIN = 0.5
LEAD_PROBABILITY_OFF = 0.35
# Exact-Accord captures only prove the primary object slot. Keep the model's
# alternate lead hypothesis out of the cluster until a stock multi-object
# capture establishes the second slot's semantics.
MAX_MODEL_LEADS = 1
MAX_OBJECT_ID = 31
MAX_RETAINED_TRACKS = 8
LEAD_DEDUP_LONGITUDINAL_M = 3.0
LEAD_DEDUP_LATERAL_M = 1.0
TRACK_MAX_AGE_S = 1.0
TRACK_LONGITUDINAL_GATE_MIN_M = 5.0
TRACK_LONGITUDINAL_GATE_MAX_M = 15.0
TRACK_LONGITUDINAL_GATE_FACTOR = 0.15
TRACK_LATERAL_GATE_M = 1.5
DISPLAY_DREL_TAU_S = 0.6
DISPLAY_YREL_TAU_S = 0.5
DISPLAY_VREL_FEEDFORWARD_MIN = 0.5
DISPLAY_DREL_RESIDUAL_MAX_M = 1.5
# modelV2 lead distance is camera-referenced. Honda's radar-originated HUD
# objects use the radar frame, matching radard's exact conversion.
MODEL_CAMERA_TO_RADAR_M = 1.52
# Honda HUD object classification: 7=car, 6=motorcycle, -7=truck.
CAR_TYPE_CAR = 7
LONG_DISTANCE_MAX = 194.0
LATERAL_DISTANCE_MAX = 204.7
# Honda's HUD_OBJECTS LAT_DIST is in physical meters. Exact-Accord stock
# object/model pairs do not support the 0.35 downscale inherited from the
# earlier generic implementation.
LATERAL_SCALE = 1.0


@dataclass
class ModelLead:
  status: bool
  d_rel: float
  y_rel: float
  v_rel: float = 0.0
  object_id: int = 0


def no_lead() -> ModelLead:
  return ModelLead(False, 0.0, 0.0, 0.0, 0)


def leads_from_model(model, v_ego: float, probability_active: list[bool] | None = None) -> list[ModelLead]:
  if model is None:
    return []

  model_velocity = getattr(getattr(model, "velocity", None), "x", [])
  model_v_ego = float(model_velocity[0]) if len(model_velocity) else v_ego
  leads = []
  for index in range(min(len(model.leadsV3), MAX_MODEL_LEADS)):
    lead = model.leadsV3[index]
    probability = float(lead.prob)
    active = probability_active[index] if probability_active is not None else False
    probability_threshold = LEAD_PROBABILITY_OFF if active else LEAD_PROBABILITY_MIN
    if not np.isfinite(probability) or probability < probability_threshold or min(len(lead.x), len(lead.y), len(lead.v)) == 0:
      continue

    d_rel = float(lead.x[0]) - MODEL_CAMERA_TO_RADAR_M
    # model leads use +right; Honda HUD objects and RadarPoint.yRel use +left.
    # radard applies the same sign conversion when matching model leads.
    y_rel = -float(lead.y[0])
    v_rel = float(lead.v[0]) - model_v_ego
    if not all(np.isfinite(value) for value in (d_rel, y_rel, v_rel)) or d_rel < 0.0:
      continue

    candidate = ModelLead(True, d_rel, y_rel, v_rel)
    duplicate = any(
      abs(candidate.d_rel - existing.d_rel) < LEAD_DEDUP_LONGITUDINAL_M
      and abs(candidate.y_rel - existing.y_rel) < LEAD_DEDUP_LATERAL_M
      for existing in leads
    )
    if not duplicate:
      leads.append(candidate)
  return leads


def lead_from_model(model) -> ModelLead:
  leads = leads_from_model(model, 0.0)
  if not leads:
    return no_lead()
  lead = leads[0]
  return ModelLead(lead.status, lead.d_rel, lead.y_rel, lead.v_rel, 1)


@dataclass
class _TrackedLead:
  object_id: int
  d_rel: float
  y_rel: float
  v_rel: float
  timestamp: float
  display_d_rel: float
  display_y_rel: float


class LeadObjectTracker:
  """Assigns persistent Honda object IDs to model lead hypotheses."""

  def __init__(self):
    self._tracks: dict[int, _TrackedLead] = {}
    self._next_object_id = 1
    self._hypothesis_active = [False] * MAX_MODEL_LEADS

  def _mint_id(self, reserved_ids: set[int]) -> int:
    for _ in range(MAX_OBJECT_ID):
      object_id = self._next_object_id
      self._next_object_id = self._next_object_id % MAX_OBJECT_ID + 1
      if object_id not in self._tracks and object_id not in reserved_ids:
        return object_id

    reclaimable = [
      track for object_id, track in self._tracks.items()
      if object_id not in reserved_ids
    ]
    if not reclaimable:
      raise RuntimeError("no unique Honda HUD object ID available")
    oldest = min(reclaimable, key=lambda track: track.timestamp)
    self._tracks.pop(oldest.object_id)
    return oldest.object_id

  def update_from_model(self, model, v_ego: float, now: float) -> list[ModelLead]:
    if model is None:
      self._hypothesis_active = [False] * MAX_MODEL_LEADS
      return self.update([], now)

    for index in range(MAX_MODEL_LEADS):
      probability = float(model.leadsV3[index].prob) if index < len(model.leadsV3) else 0.0
      if not np.isfinite(probability):
        self._hypothesis_active[index] = False
        continue
      threshold = LEAD_PROBABILITY_OFF if self._hypothesis_active[index] else LEAD_PROBABILITY_MIN
      self._hypothesis_active[index] = probability >= threshold
    return self.update(leads_from_model(model, v_ego, self._hypothesis_active), now)

  def update(self, leads: list[ModelLead], now: float) -> list[ModelLead]:
    self._tracks = {
      object_id: track for object_id, track in self._tracks.items()
      if 0.0 <= now - track.timestamp <= TRACK_MAX_AGE_S
    }

    matches = []
    for lead_index, lead in enumerate(leads):
      for object_id, track in self._tracks.items():
        dt = max(0.0, now - track.timestamp)
        predicted_distance = track.d_rel + track.v_rel * dt
        longitudinal_error = abs(lead.d_rel - predicted_distance)
        lateral_error = abs(lead.y_rel - track.y_rel)
        longitudinal_gate = float(np.clip(
          TRACK_LONGITUDINAL_GATE_FACTOR * max(lead.d_rel, track.d_rel),
          TRACK_LONGITUDINAL_GATE_MIN_M,
          TRACK_LONGITUDINAL_GATE_MAX_M,
        ))
        if longitudinal_error <= longitudinal_gate and lateral_error <= TRACK_LATERAL_GATE_M:
          cost = longitudinal_error / longitudinal_gate + lateral_error / TRACK_LATERAL_GATE_M
          matches.append((cost, lead_index, object_id))

    assigned_leads = set()
    assigned_tracks = set()
    object_ids = [0] * len(leads)
    for _, lead_index, object_id in sorted(matches):
      if lead_index in assigned_leads or object_id in assigned_tracks:
        continue
      assigned_leads.add(lead_index)
      assigned_tracks.add(object_id)
      object_ids[lead_index] = object_id

    tracked = []
    used_ids = set(assigned_tracks)
    for index, lead in enumerate(leads):
      object_id = object_ids[index] or self._mint_id(used_ids)
      used_ids.add(object_id)
      previous = self._tracks.get(object_id)
      if previous is None:
        display_d_rel = lead.d_rel
        display_y_rel = lead.y_rel
      else:
        dt = max(0.0, now - previous.timestamp)
        display_d_rel = previous.display_d_rel
        if abs(lead.v_rel) >= DISPLAY_VREL_FEEDFORWARD_MIN:
          display_d_rel += lead.v_rel * dt
        residual = float(np.clip(
          lead.d_rel - display_d_rel,
          -DISPLAY_DREL_RESIDUAL_MAX_M,
          DISPLAY_DREL_RESIDUAL_MAX_M,
        ))
        display_d_rel += (1.0 - math.exp(-dt / DISPLAY_DREL_TAU_S)) * residual
        display_y_rel = previous.display_y_rel + (
          1.0 - math.exp(-dt / DISPLAY_YREL_TAU_S)
        ) * (lead.y_rel - previous.display_y_rel)
      self._tracks[object_id] = _TrackedLead(
        object_id, lead.d_rel, lead.y_rel, lead.v_rel, now, display_d_rel, display_y_rel,
      )
      tracked.append(ModelLead(
        lead.status, display_d_rel, display_y_rel, lead.v_rel, object_id,
      ))

    if len(self._tracks) > MAX_RETAINED_TRACKS:
      retained_ids = set(used_ids)
      newest_unmatched = sorted(
        (track for object_id, track in self._tracks.items() if object_id not in retained_ids),
        key=lambda track: track.timestamp,
        reverse=True,
      )
      retained_ids.update(
        track.object_id for track in newest_unmatched[:max(0, MAX_RETAINED_TRACKS - len(retained_ids))]
      )
      self._tracks = {
        object_id: track for object_id, track in self._tracks.items()
        if object_id in retained_ids
      }
    return tracked


def _lead_rotation(y_rel: float) -> int:
  magnitude = min(round(abs(y_rel) / 1.5), 6)
  return -magnitude if y_rel > 0.0 else magnitude


def create_hud_object(packer, bus: int, mux: int, leads: list[ModelLead] | ModelLead):
  values = {"MUX": mux}
  slot = (mux - 1) % 16
  if isinstance(leads, ModelLead):
    leads = [leads] if leads.status else []
  lead = leads[slot] if slot < len(leads) else no_lead()
  if lead.status:
    lateral = float(np.clip(lead.y_rel * LATERAL_SCALE, -LATERAL_DISTANCE_MAX, LATERAL_DISTANCE_MAX))
    values.update({
      "OBJECT_ID": lead.object_id,
      "IS_LEAD_CAR": int(slot == 0),
      "CAR_TYPE": CAR_TYPE_CAR,
      "ROTATION": _lead_rotation(lateral / LATERAL_SCALE),
      "LONG_DIST": float(np.clip(lead.d_rel, 0.0, LONG_DISTANCE_MAX)),
      "LAT_DIST": lateral,
    })
  else:
    values.update(INACTIVE_OBJECT)
  return packer.make_can_msg("HUD_OBJECTS", bus, values)
