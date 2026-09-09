import math
import unittest
import numpy as np
from lab.se3 import abs_pose_error_6d

class TestLabSE3(unittest.TestCase):
    def test_identity(self):
        e=abs_pose_error_6d(np.eye(4),np.eye(4))
        self.assertTrue(np.allclose(e,0.0,atol=1e-12))

    def test_known_translation_and_yaw(self):
        gt=np.eye(4); a=math.radians(10.0)
        gt[:3,:3]=[[math.cos(a),-math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1]]
        gt[:3,3]=[0.01,-0.02,0.03]
        e=abs_pose_error_6d(np.eye(4),gt)
        self.assertTrue(np.allclose(e[:3],[0.01,0.02,0.03],atol=1e-12))
        self.assertTrue(np.allclose(e[3:],[0.0,0.0,a],atol=1e-10))

if __name__=='__main__': unittest.main()
