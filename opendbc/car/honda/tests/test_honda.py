import re
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from opendbc.car.honda.carstate import CarState
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.fingerprints import FW_VERSIONS
from opendbc.car.honda.values import CAR, HONDA_BOSCH, HONDA_BOSCH_TJA_CONTROL

HONDA_FW_VERSION_RE = br"[A-Z0-9]{5}(-|,)[A-Z0-9]{3}(-|,)[A-Z0-9]{4}(\x00){2}$"


class TestHondaFingerprint(unittest.TestCase):
  def test_fw_version_format(self):
    # Asserts all FW versions follow an expected format
    for fw_by_ecu in FW_VERSIONS.values():
      for fws in fw_by_ecu.values():
        for fw in fws:
          assert re.match(HONDA_FW_VERSION_RE, fw) is not None, fw

  def test_tja_bosch_only(self):
    assert set(HONDA_BOSCH_TJA_CONTROL).issubset(set(HONDA_BOSCH)), "Nidec car found in TJA control list"


class TestAccordRadarHandoff(unittest.TestCase):
  def setUp(self):
    self.state = CarState.__new__(CarState)
    self.state.stock_acc_seen = False
    self.state.stock_acc_silent_frames = 0
    self.state.stock_acc_alive = True
    self.state.camera_steer_seen = False
    self.state.camera_steer_silent_frames = 0
    self.state.canfd_relay_open = False

  @staticmethod
  def cp(acc=False, steer=False):
    return SimpleNamespace(vl_all={
      "ACC_CONTROL": {"COUNTER": [1] if acc else []},
      "STEERING_CONTROL": {"COUNTER": [1] if steer else []},
    })

  def test_missing_startup_evidence_fails_closed(self):
    for _ in range(1000):
      self.state._update_accord_radar_handoff(self.cp())
    self.assertTrue(self.state.stock_acc_alive)
    self.assertFalse(self.state.canfd_relay_open)

  def test_relay_requires_seen_then_silent(self):
    self.state._update_accord_radar_handoff(self.cp(acc=True, steer=True))
    for _ in range(4):
      self.state._update_accord_radar_handoff(self.cp(acc=True))
      self.assertFalse(self.state.canfd_relay_open)
    self.state._update_accord_radar_handoff(self.cp(acc=True))
    self.assertTrue(self.state.canfd_relay_open)
    self.assertTrue(self.state.stock_acc_alive)

  def test_replacement_waits_for_four_missed_stock_frames(self):
    self.state._update_accord_radar_handoff(self.cp(acc=True, steer=True))
    for _ in range(3):
      self.state._update_accord_radar_handoff(self.cp())
      self.assertTrue(self.state.stock_acc_alive)
    self.state._update_accord_radar_handoff(self.cp())
    self.assertFalse(self.state.stock_acc_alive)

  def test_interface_defers_disable_but_deinit_reenables(self):
    cp = SimpleNamespace(
      carFingerprint=CAR.HONDA_ACCORD_11G,
      openpilotLongitudinalControl=True,
      safetyConfigs=[object()],
    )
    can_recv = object()
    can_send = object()
    with patch("opendbc.car.honda.interface.disable_ecu") as disable:
      CarInterface.init(cp, None, can_recv, can_send)
      disable.assert_not_called()

      CarInterface.deinit(cp, can_recv, can_send)
      disable.assert_called_once()
      self.assertEqual(disable.call_args.kwargs["com_cont_req"], b"\x28\x80\x03")
