import unittest
from types import SimpleNamespace

import numpy as np

from opendbc.can import CANPacker, CANParser
from opendbc.car.honda import hondacan, hud_objects, lane_path
from opendbc.car.honda.values import CAR, CruiseButtons, CruiseSettings


DBC = "honda_common_canfd_generated"


def model_with_lanes_and_lead(lead_probability=0.9, second_lead_probability=0.0):
  x = np.linspace(0.0, 120.0, 33).tolist()
  left = SimpleNamespace(x=x, y=[1.8] * len(x))
  right = SimpleNamespace(x=x, y=[-1.8] * len(x))
  unused = SimpleNamespace(x=x, y=[0.0] * len(x))
  lead = SimpleNamespace(prob=lead_probability, x=[35.0], y=[0.5], v=[18.0])
  second_lead = SimpleNamespace(prob=second_lead_probability, x=[55.0], y=[-0.2], v=[20.0])
  return SimpleNamespace(
    laneLines=[unused, left, right],
    laneLineProbs=[0.0, 0.9, 0.9],
    leadsV3=[lead, second_lead],
    velocity=SimpleNamespace(x=[20.0]),
  )


class TestAccordClusterVisualization(unittest.TestCase):
  def setUp(self):
    self.packer = CANPacker(DBC)
    self.parser = CANParser(DBC, [("LANE_PATH", 0), ("HUD_OBJECTS", 0), ("RADAR_LEAD", 0)], 0)

  def test_mux_cycle(self):
    expected = tuple(range(1, 11)) + tuple(range(17, 27)) + tuple(range(33, 43)) + tuple(range(49, 59))
    self.assertEqual(lane_path.MUX_CYCLE, expected)

  def test_idle_path_is_stock_terminated_prefix(self):
    lane = lane_path.blank_dash_lane()
    self.assertEqual(lane_path.canfd_lane_length(lane), 6)
    self.assertEqual(lane_path.canfd_lane_offsets(lane), [0] * 6 + [2047] * 34)

  def test_active_path_length_matches_terminator(self):
    lane = lane_path.LanePathFitter().update(model_with_lanes_and_lead(), 20.0, 35.0)
    offsets = lane_path.canfd_lane_offsets(lane)
    valid_points = lane_path.canfd_lane_length(lane)
    self.assertEqual(valid_points, 24)
    self.assertNotEqual(offsets[valid_points - 1], lane_path.OFFSET_UNAVAILABLE)
    self.assertTrue(all(value == lane_path.OFFSET_UNAVAILABLE for value in offsets[valid_points:]))

  def test_lane_curve_uses_stock_accord_offset_scale(self):
    x = np.linspace(0.0, 100.0, 41)
    offsets = lane_path.encode_lane_path(x, x / 100.0)
    self.assertEqual(offsets[-1], -109)

  def test_lane_curve_matches_captured_stock_shape(self):
    # Exact Accord route 00000034--08849facfe--0 at t=52.530:
    # compare the curve shape after removing the instantaneous lateral origin.
    stock = np.array([
      -78, -76, -76, -78, -80, -84, -87, -92, -98, -106, -114,
      -123, -134, -145, -158, -171, -186, -201, -218, -235, -254,
    ])
    model_center = np.array([
      0.321, 0.307, 0.304, 0.315, 0.333, 0.365, 0.414,
      0.473, 0.543, 0.635, 0.735, 0.858, 0.987, 1.136,
      1.288, 1.457, 1.627, 1.813, 1.999, 2.185, 2.372,
    ])
    y = np.interp(lane_path.LOOKAHEAD, lane_path.LOOKAHEAD[:len(model_center)], model_center)
    encoded = np.array(lane_path.encode_lane_path(lane_path.LOOKAHEAD, y)[:len(stock)])
    stock_shape = stock - stock[0]
    encoded_shape = encoded - encoded[0]
    error = stock_shape - encoded_shape
    self.assertLess(np.sqrt(np.sum(error ** 2) / len(error)), 20.0)
    self.assertGreater(abs(encoded_shape[-1]), 150)

  def test_fitter_removes_absolute_lane_origin(self):
    model = model_with_lanes_and_lead()
    for line in model.laneLines[1:3]:
      line.y = (np.asarray(line.y) + 0.65).tolist()
    lane = lane_path.LanePathFitter().update(model, 20.0, 35.0)
    self.assertEqual(lane.offsets[0], 0)
    self.assertTrue(all(offset == 0 for offset in lane.offsets))

  def test_ego_relative_translation_preserves_curve_shape(self):
    x = np.linspace(0.0, 120.0, 49)
    curve = 0.4 + 0.0008 * x ** 2
    translated = lane_path.ego_relative_lane_path(x, curve)
    np.testing.assert_allclose(np.diff(translated), np.diff(curve), atol=1e-12)
    self.assertAlmostEqual(float(np.interp(lane_path.LOOKAHEAD[0], x, translated)), 0.0)

  def test_lane_offsets_clip_below_unavailable_sentinel(self):
    offsets = lane_path.encode_lane_path(lane_path.LOOKAHEAD, np.full(len(lane_path.LOOKAHEAD), 100.0))
    self.assertTrue(all(value == -lane_path.OFFSET_VALID_MAX for value in offsets))
    self.assertNotIn(lane_path.OFFSET_UNAVAILABLE, offsets)

  def test_missing_model_fails_blank(self):
    lane = lane_path.LanePathFitter().update(None, 20.0, 0.0)
    self.assertEqual(lane_path.canfd_lane_offsets(lane), lane_path.CANFD_IDLE_OFFSETS)
    self.assertFalse(hud_objects.lead_from_model(None).status)

  def test_model_lead_converts_positive_right_to_honda_positive_left(self):
    lead = hud_objects.lead_from_model(model_with_lanes_and_lead())
    self.assertEqual(lead.y_rel, -0.5)
    self.assertAlmostEqual(lead.d_rel, 35.0 - hud_objects.MODEL_CAMERA_TO_RADAR_M)
    self.assertEqual(lead.v_rel, -2.0)

  def test_lane_and_object_share_mux(self):
    lane = lane_path.LanePathFitter().update(model_with_lanes_and_lead(), 20.0, 35.0)
    lead = hud_objects.ModelLead(True, 35.0, -0.5, -2.0, 1)
    mux = 17
    messages = [
      lane_path.create_lane_path(self.packer, 0, lane_path.canfd_lane_offsets(lane), mux),
      hud_objects.create_hud_object(self.packer, 0, mux, lead),
    ]
    self.parser.update([0, messages])
    self.assertEqual(self.parser.vl["LANE_PATH"]["MUX"], mux)
    self.assertEqual(self.parser.vl["HUD_OBJECTS"]["MUX"], mux)

  def test_only_slot_zero_contains_lead(self):
    lead = hud_objects.ModelLead(True, 35.0, -0.5, -2.0, 1)
    slot_zero = hud_objects.create_hud_object(self.packer, 0, 1, lead)
    slot_one = hud_objects.create_hud_object(self.packer, 0, 2, lead)
    self.parser.update([0, [slot_zero]])
    self.assertEqual(self.parser.vl["HUD_OBJECTS"]["OBJECT_ID"], 1)
    self.assertEqual(self.parser.vl["HUD_OBJECTS"]["IS_LEAD_CAR"], 1)
    self.assertEqual(self.parser.vl["HUD_OBJECTS"]["CAR_TYPE"], 7)
    self.assertAlmostEqual(self.parser.vl["HUD_OBJECTS"]["LAT_DIST"], -0.5)
    self.parser.update([0, [slot_one]])
    self.assertEqual(self.parser.vl["HUD_OBJECTS"]["OBJECT_ID"], 0)
    self.assertEqual(self.parser.vl["HUD_OBJECTS"]["IS_LEAD_CAR"], 0)
    self.assertEqual(self.parser.vl["HUD_OBJECTS"]["CAR_TYPE"], -1)
    self.assertEqual(self.parser.vl["HUD_OBJECTS"]["ROTATION"], -128)
    self.assertAlmostEqual(self.parser.vl["HUD_OBJECTS"]["LONG_DIST"], 196.9, delta=0.01)
    self.assertAlmostEqual(self.parser.vl["HUD_OBJECTS"]["LAT_DIST"], 204.7)

  def test_unproven_second_model_lead_is_not_emitted(self):
    model = model_with_lanes_and_lead(second_lead_probability=0.8)
    tracker = hud_objects.LeadObjectTracker()
    leads = tracker.update(hud_objects.leads_from_model(model, 20.0), 1.0)
    self.assertEqual(len(leads), 1)

    self.parser.update([0, [hud_objects.create_hud_object(self.packer, 0, 1, leads)]])
    self.assertNotEqual(self.parser.vl["HUD_OBJECTS"]["OBJECT_ID"], 0)
    self.assertEqual(self.parser.vl["HUD_OBJECTS"]["IS_LEAD_CAR"], 1)
    self.parser.update([0, [hud_objects.create_hud_object(self.packer, 0, 2, leads)]])
    self.assertEqual(self.parser.vl["HUD_OBJECTS"]["OBJECT_ID"], 0)
    self.assertEqual(self.parser.vl["HUD_OBJECTS"]["IS_LEAD_CAR"], 0)

  def test_object_ids_follow_cars_when_model_order_swaps(self):
    tracker = hud_objects.LeadObjectTracker()
    first = [
      hud_objects.ModelLead(True, 30.0, 0.0, -1.0),
      hud_objects.ModelLead(True, 45.0, 2.0, 0.0),
    ]
    tracked_first = tracker.update(first, 1.0)
    swapped = [
      hud_objects.ModelLead(True, 44.8, 2.1, 0.0),
      hud_objects.ModelLead(True, 29.8, 0.1, -1.0),
    ]
    tracked_swapped = tracker.update(swapped, 1.1)
    self.assertEqual(tracked_swapped[0].object_id, tracked_first[1].object_id)
    self.assertEqual(tracked_swapped[1].object_id, tracked_first[0].object_id)

  def test_abrupt_handoff_gets_new_object_id(self):
    tracker = hud_objects.LeadObjectTracker()
    first = tracker.update([hud_objects.ModelLead(True, 20.0, 0.0, 0.0)], 1.0)
    handoff = tracker.update([hud_objects.ModelLead(True, 55.0, 0.0, 0.0)], 1.1)
    self.assertNotEqual(first[0].object_id, handoff[0].object_id)

  def test_far_distance_handoff_gets_new_object_id(self):
    tracker = hud_objects.LeadObjectTracker()
    first = tracker.update([hud_objects.ModelLead(True, 120.0, 0.0, 0.0)], 1.0)
    handoff = tracker.update([hud_objects.ModelLead(True, 160.0, 0.0, 0.0)], 1.05)
    self.assertNotEqual(first[0].object_id, handoff[0].object_id)

  def test_lateral_cut_in_gets_new_object_id(self):
    tracker = hud_objects.LeadObjectTracker()
    first = tracker.update([hud_objects.ModelLead(True, 50.0, 0.0, 0.0)], 1.0)
    handoff = tracker.update([hud_objects.ModelLead(True, 50.0, 3.0, 0.0)], 1.05)
    self.assertNotEqual(first[0].object_id, handoff[0].object_id)

  def test_display_smoothing_rejects_jitter_and_snaps_on_handoff(self):
    tracker = hud_objects.LeadObjectTracker()
    first = tracker.update([hud_objects.ModelLead(True, 30.0, 0.0, 0.0)], 1.0)
    jitter = tracker.update([hud_objects.ModelLead(True, 34.5, 1.0, 0.0)], 1.05)
    handoff = tracker.update([hud_objects.ModelLead(True, 60.0, -3.0, 0.0)], 1.1)
    self.assertEqual(jitter[0].object_id, first[0].object_id)
    self.assertLess(jitter[0].d_rel, 31.0)
    self.assertLess(jitter[0].y_rel, 0.2)
    self.assertNotEqual(handoff[0].object_id, jitter[0].object_id)
    self.assertEqual(handoff[0].d_rel, 60.0)
    self.assertEqual(handoff[0].y_rel, -3.0)

  def test_probability_hysteresis_avoids_icon_flicker(self):
    tracker = hud_objects.LeadObjectTracker()
    model = model_with_lanes_and_lead(lead_probability=0.9)
    first = tracker.update_from_model(model, 20.0, 1.0)
    model.leadsV3[0].prob = 0.4
    uncertain = tracker.update_from_model(model, 20.0, 1.05)
    model.leadsV3[0].prob = 0.2
    gone = tracker.update_from_model(model, 20.0, 1.1)
    self.assertEqual(uncertain[0].object_id, first[0].object_id)
    self.assertEqual(gone, [])

  def test_nonfinite_lead_probability_fails_blank(self):
    tracker = hud_objects.LeadObjectTracker()
    for probability in (np.nan, np.inf, -np.inf):
      model = model_with_lanes_and_lead(lead_probability=probability)
      self.assertEqual(hud_objects.leads_from_model(model, 20.0), [])
      self.assertEqual(tracker.update_from_model(model, 20.0, 1.0), [])

  def test_object_ids_remain_unique_under_exhaustion(self):
    tracker = hud_objects.LeadObjectTracker()
    for index in range(80):
      leads = tracker.update([
        hud_objects.ModelLead(True, 10.0 if index % 2 else 100.0, -3.0, 0.0),
        hud_objects.ModelLead(True, 20.0 if index % 2 else 150.0, 3.0, 0.0),
      ], 1.0 + index * 0.01)
      ids = [lead.object_id for lead in leads]
      self.assertEqual(len(ids), len(set(ids)))
      self.assertTrue(all(1 <= object_id <= hud_objects.MAX_OBJECT_ID for object_id in ids))

  def test_lkas_state_change_pulse(self):
    pulse = lane_path.LkasStateChangePulse()
    initial_results = [pulse.update((False, False)) for _ in range(lane_path.LKAS_STATE_CHANGE_FRAMES + 1)]
    self.assertTrue(all(initial_results[:lane_path.LKAS_STATE_CHANGE_FRAMES]))
    self.assertFalse(initial_results[-1])
    changed_results = [pulse.update((True, True)) for _ in range(lane_path.LKAS_STATE_CHANGE_FRAMES + 1)]
    self.assertTrue(all(changed_results[:lane_path.LKAS_STATE_CHANGE_FRAMES]))
    self.assertFalse(changed_results[-1])

  def test_stock_active_capture_decodes_expected_protocol(self):
    messages = [
      (0x6CD5558, bytes.fromhex("200220237ff7ff25"), 0),
      (0x6CD5559, bytes.fromhex("0420920070800434"), 0),
      (0xF31AA5C, bytes.fromhex("2093f0780000002c"), 0),
    ]
    self.parser.update([0, messages])
    self.assertEqual(self.parser.vl["LANE_PATH"]["MUX"], 8)
    self.assertEqual(self.parser.vl["LANE_PATH"]["PATH_OFFSET_1"], 34)
    self.assertEqual(self.parser.vl["LANE_PATH"]["PATH_OFFSET_2"], 35)
    self.assertEqual(self.parser.vl["LANE_PATH"]["PATH_OFFSET_3"], 2047)
    self.assertEqual(self.parser.vl["HUD_OBJECTS"]["OBJECT_ID"], 4)
    self.assertEqual(self.parser.vl["HUD_OBJECTS"]["IS_LEAD_CAR"], 1)
    self.assertEqual(self.parser.vl["HUD_OBJECTS"]["CAR_TYPE"], -7)
    self.assertEqual(self.parser.vl["HUD_OBJECTS"]["ROTATION"], 0)
    self.assertAlmostEqual(self.parser.vl["HUD_OBJECTS"]["LONG_DIST"], 77.15)
    self.assertAlmostEqual(self.parser.vl["HUD_OBJECTS"]["LAT_DIST"], 0.4)
    self.assertEqual(self.parser.vl["RADAR_LEAD"]["LANE_PATH_LENGTH"], 30)
    self.assertEqual(self.parser.vl["RADAR_LEAD"]["LEFT_LANE"], 3)
    self.assertEqual(self.parser.vl["RADAR_LEAD"]["RIGHT_LANE"], 3)

  def test_startup_only_radar_lead_does_not_become_alive_requirement(self):
    parser = CANParser(DBC, [("RADAR_LEAD", float('nan'))], 0)
    parser.update([20_000_000_000, []])
    self.assertTrue(parser.can_valid)

  def test_replacement_buttons_preserve_ambient_light(self):
    parser = CANParser(DBC, [("SCM_BUTTONS", 0)], 2)
    can = SimpleNamespace(pt=0, camera=2)
    msg = hondacan.spam_buttons_command(
      self.packer, can, CruiseButtons.RES_ACCEL, CAR.HONDA_ACCORD_11G,
      cruise_setting=CruiseSettings.LKAS, ambient_light=0x72, bus=can.camera,
    )
    parser.update([0, [msg]])
    self.assertEqual(parser.vl["SCM_BUTTONS"]["CRUISE_BUTTONS"], CruiseButtons.RES_ACCEL)
    self.assertEqual(parser.vl["SCM_BUTTONS"]["CRUISE_SETTING"], CruiseSettings.LKAS)
    self.assertEqual(parser.vl["SCM_BUTTONS"]["AMBIENT_LIGHT_MAYBE"], 0x72)


if __name__ == "__main__":
  unittest.main()
