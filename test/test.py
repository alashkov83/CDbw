import unittest

import numpy as np

from cdbw import CDbw

epsilon = 1e-12

# 1 - 3D DATA TEST 1
data_3d_1 = np.load("xyz.npy")
labels_3d_1 = np.load("labels.npy")
# 2 - 3D DATA TEST 2
data_3d_2 = np.load("xyz1.npy")
labels_3d_2 = np.load("labels1.npy")
# 3 - 2D BLOBS DATA TEST
data_2d_bl = np.load("xyzbl.npy")
labels_2d_bl = np.load("labelsbl.npy")
# 4 - 2D NOISY MOONS DATA TEST
data_2d_m = np.load("xyzm.npy")
labels_2d_m = np.load("labelsm.npy")
# 5 - 2D ANISO BLOBS DATA TEST
data_2d_ab = np.load("xyzsos.npy")
labels_2d_ab = np.load("labelssos.npy")


class Data3D1(unittest.TestCase):

    def test_compact(self):
        value = 0.0005166421642678491
        compact = CDbw(data_3d_1, labels_3d_1, multipliers=True)[0]
        self.assertTrue(abs(value - compact) < epsilon,
                        msg='test_Data_3D_1 compact = {:e}, must be {:e}'.format(compact, value))

    def test_cohesion(self):
        value = 0.0005163348165265217
        cohesion = CDbw(data_3d_1, labels_3d_1, multipliers=True)[1]
        self.assertTrue(abs(value - cohesion) < epsilon,
                        msg='test_Data_3D_1 cohesion = {:e}, must be {:e}'.format(cohesion, value))

    def test_separation(self):
        value = 8.425009181172841
        separation = CDbw(data_3d_1, labels_3d_1, multipliers=True)[2]
        self.assertTrue((abs(value - separation) < epsilon),
                        msg='test_Data_3D_1 separation = {:e}, must be {:e}'.format(separation, value))

    def test_cdbw(self):
        value = 2.247458289215871e-06
        cdbw = CDbw(data_3d_1, labels_3d_1)
        self.assertTrue(abs(value - cdbw) < epsilon,
                        msg='test_Data_3D_1 cdbw = {:e}, must be {:e}'.format(cdbw, value))


class Data3D2(unittest.TestCase):

    def test_compact(self):
        value = 0.0005908076852240068
        compact = CDbw(data_3d_2, labels_3d_2, multipliers=True)[0]
        self.assertTrue(abs(value - compact) < epsilon,
                        msg='test_Data_3D_2 compact = {:e}, must be {:e}'.format(compact, value))

    def test_cohesion(self):
        value = 0.0005904808602705167
        cohesion = CDbw(data_3d_2, labels_3d_2, multipliers=True)[1]
        self.assertTrue(abs(value - cohesion) < epsilon,
                        msg='test_Data_3D_2 cohesion = {:e}, must be {:e}'.format(cohesion, value))

    def test_separation(self):
        value = 5.3479142860422
        separation = CDbw(data_3d_2, labels_3d_2, multipliers=True)[2]
        self.assertTrue(abs(value - separation) < epsilon,
                        msg='test_Data_3D_2 separation = {:e}, must be {:e}'.format(separation, value))

    def test_cdbw(self):
        value = 1.8656767482206592e-06
        cdbw = CDbw(data_3d_2, labels_3d_2)
        self.assertTrue(abs(value - cdbw) < epsilon,
                        msg='test_Data_3D_2 cdbw = {:e}, must be {:e}'.format(cdbw, value))


class Data2DBlobs(unittest.TestCase):

    def test_compact(self):
        value = 1.0296372965635063
        compact = CDbw(data_2d_bl, labels_2d_bl, multipliers=True)[0]
        self.assertTrue(abs(value - compact) < epsilon,
                        msg='test_Data_2D_Blobs compact = {:e}, must be {:e}'.format(compact, value))

    def test_cohesion(self):
        value = 0.6825376143504982
        cohesion = CDbw(data_2d_bl, labels_2d_bl, multipliers=True)[1]
        self.assertTrue(abs(value - cohesion) < epsilon,
                        msg='test_Data_2D_Blobs cohesion = {:e}, must be {:e}'.format(cohesion, value))

    def test_separation(self):
        value = 0.4065978613279437
        separation = CDbw(data_2d_bl, labels_2d_bl, multipliers=True)[2]
        self.assertTrue(abs(value - separation) < epsilon,
                        msg='test_Data_2D_Blobs separation = {:e}, must be {:e}'.format(separation, value))

    def test_cdbw(self):
        value = 0.28574322744538305
        cdbw = CDbw(data_2d_bl, labels_2d_bl)
        self.assertTrue(abs(value - cdbw) < epsilon,
                        msg='test_Data_2D_Blobs cdbw = {:e}, must be {:e}'.format(cdbw, value))


class Data2DMoons(unittest.TestCase):

    def test_compact(self):
        value = 0.6056587949929342
        compact = CDbw(data_2d_m, labels_2d_m, multipliers=True)[0]
        self.assertTrue(abs(value - compact) < epsilon,
                        msg='test_Data_2D_Moons compact = {:e}, must be {:e}'.format(compact, value))

    def test_cohesion(self):
        value = 0.5799329699706257
        cohesion = CDbw(data_2d_m, labels_2d_m, multipliers=True)[1]
        self.assertTrue(abs(value - cohesion) < epsilon,
                        msg='test_Data_2D_Moons cohesion = {:e}, must be {:e}'.format(cohesion, value))

    def test_separation(self):
        value = 0.6473693757630843
        separation = CDbw(data_2d_m, labels_2d_m, multipliers=True)[2]
        self.assertTrue(abs(value - separation) < epsilon,
                        msg='test_Data_2D_Moons separation = {:e}, must be {:e}'.format(separation, value))

    def test_cdbw(self):
        value = 0.2273829930370780
        cdbw = CDbw(data_2d_m, labels_2d_m)
        self.assertTrue(abs(value - cdbw) < epsilon,
                        msg='test_Data_2D_Moons cdbw = {:e}, must be {:e}'.format(cdbw, value))


class Data2DAnisoBlobs(unittest.TestCase):

    def test_compact(self):
        value = 1.542671293857125
        compact = CDbw(data_2d_ab, labels_2d_ab, multipliers=True)[0]
        self.assertTrue(abs(value - compact) < epsilon,
                        msg='test_Data_2D_AnisoBlobs compact = {:e}, must be {:e}'.format(compact, value))

    def test_cohesion(self):
        value = 1.046377171756573
        cohesion = CDbw(data_2d_ab, labels_2d_ab, multipliers=True)[1]
        self.assertTrue(abs(value - cohesion) < epsilon,
                        msg='test_Data_2D_AnisoBlobs cohesion = {:e}, must be {:e}'.format(cohesion, value))

    def test_separation(self):
        value = 0.25110619723766703
        separation = CDbw(data_2d_ab, labels_2d_ab, multipliers=True)[2]
        self.assertTrue(abs(value - separation) < epsilon,
                        msg='test_Data_2D_AnisoBlobs separation = {:e}, must be {:e}'.format(separation, value))

    def test_cdbw(self):
        value = 0.40533964766238123
        cdbw = CDbw(data_2d_ab, labels_2d_ab)
        self.assertTrue(abs(value - cdbw) < epsilon,
                        msg='test_Data_2D_AnisoBlobs cdbw = {:e}, must be {:e}'.format(cdbw, value))


if __name__ == '__main__':
    unittest.main()
