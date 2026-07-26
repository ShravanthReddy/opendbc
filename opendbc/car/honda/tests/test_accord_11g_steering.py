import unittest

import numpy as np

from opendbc.can import CANPacker, CANParser
from opendbc.car import DT_CTRL, gen_empty_fingerprint, rate_limit
from opendbc.car.car_helpers import interfaces
from opendbc.car.honda import hondacan
from opendbc.car.honda.values import CAR, CarControllerParams


class TestAccord11GSteering(unittest.TestCase):
  def setUp(self):
    platform = CAR.HONDA_ACCORD_11G
    CarInterface = interfaces[platform]
    self.CP = CarInterface.get_params(
      platform, gen_empty_fingerprint(), [], alpha_long=True, is_release=False, docs=False,
    )
    self.params = CarControllerParams(self.CP)
    self.can_bus = hondacan.CanBus(self.CP)
    self.packer = CANPacker("honda_common_canfd_generated")
    self.parser = CANParser("honda_common_canfd_generated", [("STEERING_CONTROL", 0)], self.can_bus.lkas)

  def test_mvl_pid_tune(self):
    self.assertEqual(self.CP.lateralTuning.which(), "pid")
    self.assertAlmostEqual(self.CP.steerActuatorDelay, 0.3)
    self.assertEqual(list(self.CP.lateralParams.torqueBP), [0, 12789])
    self.assertEqual(list(self.CP.lateralParams.torqueV), [0, 12789])
    self.assertEqual(len(self.CP.lateralTuning.pid.kpV), 1)
    self.assertEqual(len(self.CP.lateralTuning.pid.kiV), 1)
    self.assertAlmostEqual(self.CP.lateralTuning.pid.kpV[0], 0.115)
    self.assertAlmostEqual(self.CP.lateralTuning.pid.kiV[0], 0.052)
    self.assertAlmostEqual(self.CP.lateralTuning.pid.kf, 0.000035)

  def test_linear_midrange_and_full_scale(self):
    def command(normalized_torque):
      return int(np.interp(
        -normalized_torque * self.params.STEER_MAX,
        self.params.STEER_LOOKUP_BP,
        self.params.STEER_LOOKUP_V,
      ))

    self.assertEqual(self.params.STEER_MAX, 12789)
    self.assertEqual(command(0.0), 0)
    self.assertEqual(command(0.5), -6394)
    self.assertEqual(command(-0.5), 6394)
    self.assertEqual(command(1.0), -12789)
    self.assertEqual(command(-1.0), 12789)

  def test_packed_full_scale_and_inactive_zero(self):
    for torque in (-12789, 12789):
      message = hondacan.create_steering_control(self.packer, self.can_bus, torque, True, False)
      self.parser.update([0, [message]])
      self.assertEqual(self.parser.vl["STEERING_CONTROL"]["STEER_TORQUE"], torque)
      self.assertEqual(self.parser.vl["STEERING_CONTROL"]["STEER_TORQUE_REQUEST"], 1)

    inactive = hondacan.create_steering_control(self.packer, self.can_bus, 12789, False, False)
    self.parser.update([0, [inactive]])
    self.assertEqual(self.parser.vl["STEERING_CONTROL"]["STEER_TORQUE"], 0)
    self.assertEqual(self.parser.vl["STEERING_CONTROL"]["STEER_TORQUE_REQUEST"], 0)

  def test_first_step_is_rate_limited(self):
    limited = rate_limit(1.0, 0.0, -self.params.STEER_DELTA_DOWN * DT_CTRL,
                         self.params.STEER_DELTA_UP * DT_CTRL)
    raw_command = int(np.interp(
      -limited * self.params.STEER_MAX,
      self.params.STEER_LOOKUP_BP,
      self.params.STEER_LOOKUP_V,
    ))
    self.assertAlmostEqual(limited, 0.03)
    self.assertEqual(raw_command, -383)


if __name__ == "__main__":
  unittest.main()
