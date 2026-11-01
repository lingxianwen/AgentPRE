"""Demonstrate training-only calibration and frozen raw-byte checking."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agents.frozen_length_profile import calibrate, validate

field = {'off': 4, 'len': 2, 'type': 'uint16_be', 'role': 'length',
         'name': 'length', 'value': 999}
training = [bytes.fromhex(x) for x in (
    '000100000003019b01', '000200000004019b0100', '000300000005019b010000')]
profile = calibrate([('modbus', raw, [field]) for raw in training])
good = bytes.fromhex('000400000003019b01')
collision = bytes.fromhex('000400000009019b01')
correct = validate(profile, 'modbus', field, [field], good)
incorrect = validate(profile, 'modbus', field, [field], collision)
assert correct == (True, 'bytes_after:big', 'accepted')
assert incorrect == (False, 'bytes_after:big', 'rejected')
print('Frozen bytes_after:big: valid frame accepted; altered frame rejected.')
