import unittest

import sys
sys.path.append('./')
from app import detectVehicleNCount

# ---- detectVehicleNCount dependencies ------
import torch
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import cv2
import warnings
warnings.filterwarnings('ignore')
# ------------------------------------------

class TestDetection(unittest.TestCase):
    def assertVehicleCountInRange(self, vehicle_counts, valid_vc, tc):
        print("---------------")
        print("\033[93mTEST CASE {} \033[00m" .format(tc))
        print("\033[94mPredictions: {}\033[00m " .format(vehicle_counts))
        print("\033[92m{}\033[00m " .format(valid_vc))
        # Check if each vehicle count is within the specified range
        for vehicle_type, count in vehicle_counts.items():
            lower_bound = valid_vc[vehicle_type][0]
            upper_bound = valid_vc[vehicle_type][1]
            assert((count>=lower_bound), (count<=upper_bound))

    def test_detectVehicleNCount(self):
        with self.subTest():
            # Testcase 1: no vehicles on road
            valid_vc = {
                'bicycle': (0, 0),
                'car': (0, 0),
                'motorcycle': (0, 0),
                'bus': (0, 0),
                'truck': (0, 0),
            }
            self.assertVehicleCountInRange(detectVehicleNCount(Image.open('./tests/images/1.png'))['data'], valid_vc, 1)

        with self.subTest():
            # Testcase 2
            valid_vc = {
                'bicycle': (0, 0),
                'car': (4, 4),
                'motorcycle': (0, 0),
                'bus': (0, 0),
                'truck': (0, 0),
            }
            self.assertVehicleCountInRange(detectVehicleNCount(Image.open('./tests/images/2.png'))['data'], valid_vc, 2)

        with self.subTest():
            # Testcase 3
            valid_vc = {
                'bicycle': (0, 0),
                'car': (3, 4),          # 4th isn't fully visible
                'motorcycle': (0, 2),   # are hard to detect
                'bus': (0, 0),
                'truck': (0, 0),
            }
            self.assertVehicleCountInRange(detectVehicleNCount(Image.open('./tests/images/3.png'))['data'], valid_vc, 3)

        with self.subTest():
            # Testcase 4
            valid_vc = {
                'bicycle': (0, 0),
                'car': (0, 0),
                'motorcycle': (0, 0),
                'bus': (0, 1),          # looks similar
                'truck': (0, 1),        # looks similar
            }
            self.assertVehicleCountInRange(detectVehicleNCount(Image.open('./tests/images/4.png'))['data'], valid_vc, 4)
        
        with self.subTest():
            # Testcase 5
            valid_vc = {
                'bicycle': (0, 0),
                'car': (4, 4),
                'motorcycle': (0, 1),
                'bus': (0, 0),          
                'truck': (0, 1),       
            }
            self.assertVehicleCountInRange(detectVehicleNCount(Image.open('./tests/images/5.png'))['data'], valid_vc, 5)

        with self.subTest():
            # Testcase 6
            valid_vc = {
                'bicycle': (0, 0),
                'car': (4, 4),
                'motorcycle': (1, 1),
                'bus': (0, 1),          
                'truck': (0, 1),       
            }
            self.assertVehicleCountInRange(detectVehicleNCount(Image.open('./tests/images/6.png'))['data'], valid_vc, 6)


if __name__ == '__main__':
    unittest.main()