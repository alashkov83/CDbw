import unittest

import numpy as np

from cdbw import CDbw

epsilon = 1e-16

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
        value = 0.020238323923050704
        compact = CDbw(data_3d_1, labels_3d_1, multipliers=True)[0]
        self.assertTrue(abs(value - compact) < epsilon,
                        msg='test_Data_3D_1 compact = {:e}, must be {:e}'.format(compact, value))

    def test_cohesion(self):
        value = 0.020167225676202064
        cohesion = CDbw(data_3d_1, labels_3d_1, multipliers=True)[1]
        self.assertTrue(abs(value - cohesion) < epsilon,
                        msg='test_Data_3D_1 cohesion = {:e}, must be {:e}'.format(cohesion, value))

    def test_separation(self):
        value = 5.117729198421542
        separation = CDbw(data_3d_1, labels_3d_1, multipliers=True)[2]
        self.assertTrue((abs(value - separation) < epsilon),
                        msg='test_Data_3D_1 separation = {:e}, must be {:e}'.format(separation, value))

    def test_cdbw(self):
        value = 0.002088805501239885
        cdbw = CDbw(data_3d_1, labels_3d_1)
        self.assertTrue(abs(value - cdbw) < epsilon,
                        msg='test_Data_3D_1 cdbw = {:e}, must be {:e}'.format(cdbw, value))


class Data3D2(unittest.TestCase):

    def test_compact(self):
        value = 0.030106505876089613
        compact = CDbw(data_3d_2, labels_3d_2, multipliers=True)[0]
        self.assertTrue(abs(value - compact) < epsilon,
                        msg='test_Data_3D_2 compact = {:e}, must be {:e}'.format(compact, value))

    def test_cohesion(self):
        value = 0.029839934787904636
        cohesion = CDbw(data_3d_2, labels_3d_2, multipliers=True)[1]
        self.assertTrue(abs(value - cohesion) < epsilon,
                        msg='test_Data_3D_2 cohesion = {:e}, must be {:e}'.format(cohesion, value))

    def test_separation(self):
        value = 4.337501690166292
        separation = CDbw(data_3d_2, labels_3d_2, multipliers=True)[2]
        self.assertTrue(abs(value - separation) < epsilon,
                        msg='test_Data_3D_2 separation = {:e}, must be {:e}'.format(separation, value))

    def test_cdbw(self):
        value = 0.003896708164603387
        cdbw = CDbw(data_3d_2, labels_3d_2)
        self.assertTrue(abs(value - cdbw) < epsilon,
                        msg='test_Data_3D_2 cdbw = {:e}, must be {:e}'.format(cdbw, value))


class Data2DBlobs(unittest.TestCase):

    def test_compact(self):
        value = 0.45727766830363764
        compact = CDbw(data_2d_bl, labels_2d_bl, multipliers=True)[0]
        self.assertTrue(abs(value - compact) < epsilon,
                        msg='test_Data_2D_Blobs compact = {:e}, must be {:e}'.format(compact, value))

    def test_cohesion(self):
        value = 0.32071421306947445
        cohesion = CDbw(data_2d_bl, labels_2d_bl, multipliers=True)[1]
        self.assertTrue(abs(value - cohesion) < epsilon,
                        msg='test_Data_2D_Blobs cohesion = {:e}, must be {:e}'.format(cohesion, value))

    def test_separation(self):
        value = 0.6346662881857889
        separation = CDbw(data_2d_bl, labels_2d_bl, multipliers=True)[2]
        self.assertTrue(abs(value - separation) < epsilon,
                        msg='test_Data_2D_Blobs separation = {:e}, must be {:e}'.format(separation, value))

    def test_cdbw(self):
        value = 0.09307726853513183
        cdbw = CDbw(data_2d_bl, labels_2d_bl)
        self.assertTrue(abs(value - cdbw) < epsilon,
                        msg='test_Data_2D_Blobs cdbw = {:e}, must be {:e}'.format(cdbw, value))


class Data2DMoons(unittest.TestCase):

    def test_compact(self):
        value = 0.5001327835563026
        compact = CDbw(data_2d_m, labels_2d_m, multipliers=True)[0]
        self.assertTrue(abs(value - compact) < epsilon,
                        msg='test_Data_2D_Moons compact = {:e}, must be {:e}'.format(compact, value))

    def test_cohesion(self):
        value = 0.4583940701869082
        cohesion = CDbw(data_2d_m, labels_2d_m, multipliers=True)[1]
        self.assertTrue(abs(value - cohesion) < epsilon,
                        msg='test_Data_2D_Moons cohesion = {:e}, must be {:e}'.format(cohesion, value))

    def test_separation(self):
        value = 0.27079184296654096
        separation = CDbw(data_2d_m, labels_2d_m, multipliers=True)[2]
        self.assertTrue(abs(value - separation) < epsilon,
                        msg='test_Data_2D_Moons separation = {:e}, must be {:e}'.format(separation, value))

    def test_cdbw(self):
        value = 0.06208116987528693
        cdbw = CDbw(data_2d_m, labels_2d_m)
        self.assertTrue(abs(value - cdbw) < epsilon,
                        msg='test_Data_2D_Moons cdbw = {:e}, must be {:e}'.format(cdbw, value))


class Data2DAnisoBlobs(unittest.TestCase):

    def test_compact(self):
        value = 1.0455304698351948
        compact = CDbw(data_2d_ab, labels_2d_ab, multipliers=True)[0]
        self.assertTrue(abs(value - compact) < epsilon,
                        msg='test_Data_2D_AnisoBlobs compact = {:e}, must be {:e}'.format(compact, value))

    def test_cohesion(self):
        value = 0.6647251466801855
        cohesion = CDbw(data_2d_ab, labels_2d_ab, multipliers=True)[1]
        self.assertTrue(abs(value - cohesion) < epsilon,
                        msg='test_Data_2D_AnisoBlobs cohesion = {:e}, must be {:e}'.format(cohesion, value))

    def test_separation(self):
        value = 0.3085609726166483
        separation = CDbw(data_2d_ab, labels_2d_ab, multipliers=True)[2]
        self.assertTrue(abs(value - separation) < epsilon,
                        msg='test_Data_2D_AnisoBlobs separation = {:e}, must be {:e}'.format(separation, value))

    def test_cdbw(self):
        value = 0.21444691221568293
        cdbw = CDbw(data_2d_ab, labels_2d_ab)
        self.assertTrue(abs(value - cdbw) < epsilon,
                        msg='test_Data_2D_AnisoBlobs cdbw = {:e}, must be {:e}'.format(cdbw, value))


if __name__ == '__main__':
    unittest.main()
