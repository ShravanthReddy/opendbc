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
LEAD_OBJECT_ID = 1
# Honda HUD object classification: 7=car, 6=motorcycle, -7=truck.
CAR_TYPE_CAR = 7
LONG_DISTANCE_MAX = 194.0
LATERAL_DISTANCE_MAX = 204.7
LATERAL_SCALE = 0.35


@dataclass
class ModelLead:
  status: bool
  d_rel: float
  y_rel: float


def no_lead() -> ModelLead:
  return ModelLead(False, 0.0, 0.0)


def lead_from_model(model) -> ModelLead:
  if model is None or len(model.leadsV3) == 0:
    return no_lead()

  lead = model.leadsV3[0]
  if lead.prob < LEAD_PROBABILITY_MIN or len(lead.x) == 0 or len(lead.y) == 0:
    return no_lead()

  d_rel = float(lead.x[0])
  # model leads use +right; Honda HUD objects and RadarPoint.yRel use +left.
  # radard applies the same sign conversion when matching model leads.
  y_rel = -float(lead.y[0])
  if not np.isfinite(d_rel) or not np.isfinite(y_rel) or d_rel < 0.0:
    return no_lead()
  return ModelLead(True, d_rel, y_rel)


def _lead_rotation(y_rel: float) -> int:
  magnitude = min(round(abs(y_rel) / 1.5), 6)
  return -magnitude if y_rel > 0.0 else magnitude


def create_hud_object(packer, bus: int, mux: int, lead: ModelLead):
  values = {"MUX": mux}
  slot = (mux - 1) % 16
  if slot == 0 and lead.status:
    lateral = float(np.clip(lead.y_rel * LATERAL_SCALE, -LATERAL_DISTANCE_MAX, LATERAL_DISTANCE_MAX))
    values.update({
      "OBJECT_ID": LEAD_OBJECT_ID,
      "IS_LEAD_CAR": 1,
      "CAR_TYPE": CAR_TYPE_CAR,
      "ROTATION": _lead_rotation(lateral / LATERAL_SCALE),
      "LONG_DIST": float(np.clip(lead.d_rel, 0.0, LONG_DISTANCE_MAX)),
      "LAT_DIST": lateral,
    })
  else:
    values.update(INACTIVE_OBJECT)
  return packer.make_can_msg("HUD_OBJECTS", bus, values)
