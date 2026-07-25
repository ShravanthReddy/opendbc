import unittest
from types import SimpleNamespace

import numpy as np

from opendbc.can import CANPacker, CANParser
from opendbc.car.honda import hud_objects, lane_path


DBC = "honda_common_canfd_generated"


def model_with_lanes_and_lead(lead_probability=0.9):
  x = np.linspace(0.0, 120.0, 33).tolist()
  left = SimpleNamespace(x=x, y=[1.8] * len(x))
  right = SimpleNamespace(x=x, y=[-1.8] * len(x))
  unused = SimpleNamespace(x=x, y=[0.0] * len(x))
  lead = SimpleNamespace(prob=lead_probability, x=[35.0], y=[0.5])
  return SimpleNamespace(
    laneLines=[unused, left, right],
    laneLineProbs=[0.0, 0.9, 0.9],
    leadsV3=[lead],
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

  def test_missing_model_fails_blank(self):
    lane = lane_path.LanePathFitter().update(None, 20.0, 0.0)
    self.assertEqual(lane_path.canfd_lane_offsets(lane), lane_path.CANFD_IDLE_OFFSETS)
    self.assertFalse(hud_objects.lead_from_model(None).status)

  def test_model_lead_converts_positive_right_to_honda_positive_left(self):
    lead = hud_objects.lead_from_model(model_with_lanes_and_lead())
    self.assertEqual(lead.y_rel, -0.5)

  def test_lane_and_object_share_mux(self):
    lane = lane_path.LanePathFitter().update(model_with_lanes_and_lead(), 20.0, 35.0)
    lead = hud_objects.lead_from_model(model_with_lanes_and_lead())
    mux = 17
    messages = [
      lane_path.create_lane_path(self.packer, 0, lane_path.canfd_lane_offsets(lane), mux),
      hud_objects.create_hud_object(self.packer, 0, mux, lead),
    ]
    self.parser.update([0, messages])
    self.assertEqual(self.parser.vl["LANE_PATH"]["MUX"], mux)
    self.assertEqual(self.parser.vl["HUD_OBJECTS"]["MUX"], mux)

  def test_only_slot_zero_contains_lead(self):
    lead = hud_objects.lead_from_model(model_with_lanes_and_lead())
    slot_zero = hud_objects.create_hud_object(self.packer, 0, 1, lead)
    slot_one = hud_objects.create_hud_object(self.packer, 0, 2, lead)
    self.parser.update([0, [slot_zero]])
    self.assertEqual(self.parser.vl["HUD_OBJECTS"]["OBJECT_ID"], 1)
    self.assertEqual(self.parser.vl["HUD_OBJECTS"]["IS_LEAD_CAR"], 1)
    self.assertEqual(self.parser.vl["HUD_OBJECTS"]["CAR_TYPE"], 7)
    self.parser.update([0, [slot_one]])
    self.assertEqual(self.parser.vl["HUD_OBJECTS"]["OBJECT_ID"], 0)
    self.assertEqual(self.parser.vl["HUD_OBJECTS"]["IS_LEAD_CAR"], 0)
    self.assertEqual(self.parser.vl["HUD_OBJECTS"]["CAR_TYPE"], -1)
    self.assertEqual(self.parser.vl["HUD_OBJECTS"]["ROTATION"], -128)
    self.assertAlmostEqual(self.parser.vl["HUD_OBJECTS"]["LONG_DIST"], 196.9, delta=0.01)
    self.assertAlmostEqual(self.parser.vl["HUD_OBJECTS"]["LAT_DIST"], 204.7)

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

  def test_startup_only_radar_lead_does_not_become_alive_requirement(self):
    parser = CANParser(DBC, [("RADAR_LEAD", float('nan'))], 0)
    parser.update([20_000_000_000, []])
    self.assertTrue(parser.can_valid)


if __name__ == "__main__":
  unittest.main()
