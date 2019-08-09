import numpy as np
import cv2
import glob

def print_params(blob_params):
    for p in dir(blob_params):
        if not '__' in p:
            print('{:>20}:\t{}'.format(p, str(getattr(blob_params, p))))
    return

class CameraParameters:
    def __init__(self, **kwargs):
        Kdefault = np.eye(3)
        Kdefault[0, 2] = 1
        Kdefault[1, 2] = 1
        self.K = kwargs.get('K', Kdefault)
        self.newK = kwargs.get('newK', Kdefault)
        self.R = kwargs.get('R', np.zeros(3))
        self.D = kwargs.get('D', np.zeros(2))
        self.resolution = kwargs.get('resolution', np.array((2*self.K[0,2], 2*self.K[1,2])))
        self.roi = kwargs.get('roi', [0,0,-1,-1])
        self.mapX = []
        self.mapY = []
        self.invMapX = []
        self.invMapY = []
    def __str__(self):
        out = 'R: '+str(np.round(self.R,decimals=3))+'\n'+\
              'D: '+str(np.round(self.D,decimals=6))+'\n'+\
              'K:\n'+str(np.round(self.K,decimals=1))+'\n'#+\
              #'newK:\n'+str(np.round(self.newK,decimals=1))+'\n'+\
              #'mapX:\n'+str(np.round(self.mapX[0:6,0:6],decimals=4))+'\n'+\
              #'mapY:\n'+str(np.round(self.mapY[0:6,0:6],decimals=4))+'\n'+\
              #'invMapX:\n'+str(np.round(self.invMapX[0:6,0:4],decimals=4))+'\n'+\
              #'invMapY:\n'+str(np.round(self.invMapY[0:6,0:4],decimals=4))
        out += '\nResolution: '+str(np.round(self.resolution,decimals=0))
        return out

class Calibration:
    class IntrinsicsOut:
        def __init__(self, super, prefix):
            self.CP = CameraParameters()
            self.imgPaths = np.array(glob.glob(prefix + '*'))
            self.imgPoints = np.zeros((self.imgPaths.__len__(),super.nrows * super.ncols, 1, 2), np.float32)
            self.centers = np.zeros((super.nrows * super.ncols, 2,self.imgPaths.__len__()), np.float32)
            self.imgValid = np.zeros(self.imgPaths.__len__(), np.bool_)
            self.validInd = np.zeros(self.imgPaths.__len__(), np.uint8)
            self.rms = 0
            self.imgPointsUnd = np.zeros((self.imgPaths.__len__(),super.nrows * super.ncols, 1, 2), np.float32)
            self.imgUnd = []
            self.objPoints = []
            for i in range(self.imgPaths.__len__()):
                self.imgUnd.append([])
        def __str__(self):
            out = 'Camera Parameters:\n'+str(self.CP)+'\n'
            out += 'RMS error: ' + str(np.round(self.rms,decimals=3))
            indexes = np.where(self.imgValid)[0]
            out += '\nValid images (' + str(len(indexes)) + '/' + str(len(self.imgPaths)) + '):\n' + str(indexes)
            # for i in range(len(indexes)):
            #     out += '\t' + self.imgPaths[indexes[i]] + '\n'
            out += '\nSelected images (' + str(len(self.validInd)) + '/' + str(len(indexes)) + '):\n' + str(self.validInd)
            return out

    def set_blob_params(self,camera):
        tmp = cv2.SimpleBlobDetector_Params()
        if 'ToF' in camera or camera == 'tof':
            minDist = 5
            minA = 10
            maxA = 1000
        elif 'Pol' in camera:
            minDist = 10
            minA = 100
            maxA = 5e4
        elif 'L' in camera or camera == 'Stereo_0':
            minDist = 25
            minA = 100
            maxA = 1e5
        elif 'R' in camera or camera == 'Stereo_1':
            minDist = 25
            minA = 100
            maxA = 1e5
        tmp.minThreshold = 50
        tmp.thresholdStep = 20  # 10
        tmp.maxThreshold = 220  # 220
        tmp.minDistBetweenBlobs = minDist # default: 10
        tmp.minArea = minA # default: 25
        tmp.maxArea = maxA # default: 5000
        tmp.blobColor = 0
        tmp.minConvexity = 0.95
        tmp.maxConvexity = 3.4e38
        tmp.minInertiaRatio = 0.1
        tmp.maxInertiaRatio = 3.4e38
        tmp.minRepeatability = 2
        # tmp.filterByArea = True
        # tmp.filterByCircularity = False
        # tmp.filterByColor = True
        # tmp.filterByConvexity = False
        # tmp.filterByInertia = False
        self.findCirclesGridBlobDetectorParams = tmp
        self.findCirclesGridBlobDetector = cv2.SimpleBlobDetector_create(tmp)

    def read_gray_image(self,path):
        #return np.array(cv2.cvtColor(cv2.imread(path, cv2.IMREAD_ANYDEPTH), cv2.COLOR_BGR2GRAY))
        im = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        # im = cv2.resize(im, None, fx=0.4, fy=0.4)
        im = (im.astype(np.float)/255.0) # normalize 0,1
        if self.auto_norm=='1':
            # HISTOGRAM BASED CLIP
            if 'Pol' in path:
                excluded_percent_low = 10.0
                excluded_percent_high = 10.0
            elif 'ToF' in path or 'tof' in path:
                excluded_percent_low = 5.0
                excluded_percent_high = 2.0
            elif 'L' in path or 'Stereo_0' in path:
                excluded_percent_low = 5.0
                excluded_percent_high = 5.0
            elif 'R' in path or 'Stereo_1' in path:
                excluded_percent_low = 5.0
                excluded_percent_high = 5.0
            hist, bins = np.histogram(im.flatten(), 256)
            cdf = hist.cumsum()
            cdf_normalized = cdf.astype(float) / cdf.max()
            # select the 90% of the central values
            indMin = [ind for ind, val in enumerate(cdf_normalized) if val < excluded_percent_low/100]
            indMax = [ind for ind, val in enumerate(cdf_normalized) if val > 1-excluded_percent_high/100]
            if indMin:
                valMin = bins[indMin[-1]]
            else:
                valMin = bins[1] / 2
            if indMax:
                valMax = bins[indMax[0]]
            else:
                valMax = bins[-1]-0.5*(bins[-1]-bins[-2])
            imn = im.astype(np.float)
            imn = np.clip(imn, valMin, valMax)
            imn = (imn-valMin)/(valMax-valMin)*255
            #imeq = cv2.equalizeHist(imn.astype(np.uint8))
            imeq = imn.astype(np.uint8)
        # if imeq.shape[1]>2000: # if the resolution is too high, opencv doesnt work -> half the size of the images
        #     imeq = cv2.resize(imeq, None, fx=0.5, fy=0.5)
        else:
            # MANUALLY SELECTED CLIP ZONE
            if 'Pol' in path:
                valMin = 0.1
                valMax = 0.5
            elif 'ToF' in path or 'tof' in path:
                valMin = 0.0
                valMax = 1.0
            elif 'L' in path or 'Stereo_0' in path:
                valMin = 0.0
                valMax = 1.0
            elif 'R' in path or 'Stereo_1' in path:
                valMin = 0.0
                valMax = 1.0
            imn = np.clip(im, valMin, valMax)
            imn = (imn - valMin) / (valMax - valMin) * 255
            # imnq = (imn - valMin) / (valMax - valMin)
            imeq = imn.astype(np.uint8)
        # cv2.imshow('equalized input im', cv2.resize(imeq, None, fx=0.4, fy=0.4))
        # cv2.waitKey(0)
        return imeq

    def __init__(self, **kwargs):
        # static parameters
        self.nrows = 4
        self.ncols = 11
        self.displayTime = 200
        self.objectPoints = np.zeros((self.nrows * self.ncols, 3), np.float32)
        self.imgPath = '..\\img\\'
        index = 0
        # for j in reversed(range(self.ncols)):
        #     for i in range(self.nrows):
        #         if j % 2 == 0:
        #             self.objectPoints[index, :2] = [0.5 * j, i]
        #         else:
        #             self.objectPoints[index, :2] = [0.5 * j, i + 0.5]
        #         index += 1
        for i in range(self.ncols):
            for j in range(self.nrows):
                self.objectPoints[index, 0:2] = [0.5 * i, j + 0.5*(i%2>0)]
                index += 1
        # findCirclesGrid parameters
        self.findCirclesGridFlags = cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING
        self.findCirclesGridBlobDetectorParams = cv2.SimpleBlobDetector_Params()
        # calibrateCamera parameters
        self.calibrateCameraFlags = 0
        # self.calibrateCameraFlags = cv2.CALIB_FIX_ASPECT_RATIO
        # self.calibrateCameraFlags = cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_ZERO_TANGENT_DIST
        # stereoCalibrate parameters
        self.StereoCalibrateFlags = cv2.CALIB_FIX_INTRINSIC#+ cv2.CALIB_USE_EXTRINSIC_GUESS
        #self.StereoCalibrateFlags = cv2.CALIB_SAME_FOCAL_LENGTH + cv2.CALIB_ZERO_TANGENT_DIST
        self.StereoCalibrateCriteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

        # parameters passed from main
        options = {
            'ip0': '\img\Pol',
            'ip1': '\img\ToF',
            'ip2': '\img\S0',
            'ip3': '\img\S1'}
        options.update(kwargs)
        self.unit_length = options['L']
        self.objectPoints = self.objectPoints * self.unit_length
        self.auto_norm = options['N']
        self.out_path = options['out']
        self.input_prefix_0 = options['ip0']
        self.verbose = options['verbose']
        self.nCameras = 1
        if 'ip1' in kwargs:
            self.nCameras += 1
            self.input_prefix_1 = options['ip1']
        if 'ip2' in kwargs:
            self.nCameras += 1
            self.input_prefix_2 = options['ip2']
        if 'ip3' in kwargs:
            self.nCameras += 1
            self.input_prefix_3 = options['ip3']

        if self.nCameras==1:
            out0 = self.intrinsics(self.input_prefix_0)
            self.save({'out': out0}, self.out_path)
        elif self.nCameras==2:
            out0 = self.intrinsics(self.input_prefix_0)
            out1 = self.intrinsics(self.input_prefix_1)
            [R1, T1, repr1, ind1] = self.estrinsics(out0,out1)
            self.save({'out0': out0,'out1': out1,
                       'R1':R1,'T1': T1,'repr1': repr1, 'ind1':ind1},
                      self.out_path)
        elif self.nCameras == 3:
            print(self.input_prefix_0)
            out0 = self.intrinsics(self.input_prefix_0)
            print(self.input_prefix_1)
            out1 = self.intrinsics(self.input_prefix_1)
            print(self.input_prefix_2)
            out2 = self.intrinsics(self.input_prefix_2)
            [R1, T1, repr1, ind1] = self.estrinsics(out0, out1)
            [R2, T2, repr2, ind2] = self.estrinsics(out0, out2)
            R2 = np.transpose(R2)
            T2 = -T2
            self.save({'out0': out0,'out1': out1,'out2': out2,
                       'R1':R1,'T1': T1,'repr1': repr1, 'ind1':ind1,
                       'R2':R2,'T2': T2,'repr2': repr2, 'ind2':ind2},
                      self.out_path)
        elif self.nCameras == 4:
            print("Cam0")
            out0 = self.intrinsics(self.input_prefix_0)
            print("Cam1")
            out1 = self.intrinsics(self.input_prefix_1)
            print("Cam2")
            out2 = self.intrinsics(self.input_prefix_2)
            print("Cam3")
            out3 = self.intrinsics(self.input_prefix_3)

            self.R_guess, _ = cv2.Rodrigues(np.array([0.0, 0.1, 0.0]))
            print("Estrinsics between cam0 and cam1")
            self.T_guess = np.array([-53, 26, 31])
            [R1, T1, repr1, ind1] = self.estrinsics(out0, out1)
            print("Estrinsics between cam0 and cam2")
            self.T_guess = np.array([80, -27, -5])
            [R2, T2, repr2, ind2] = self.estrinsics(out0, out2)
            # R2 = np.transpose(R2)
            # T2 = -T2
            # print("Estrinsics between cam2 and cam0")
            # [R2, T2, repr2, ind2] = self.estrinsics(out2, out0)
            # print("Estrinsics between cam1 and cam2")
            # [R12, T12, repr12, ind12] = self.estrinsics(out1, out2)
            print("Estrinsics between cam0 and cam3")
            self.T_guess = np.array([-119, -26, -6])
            [R3, T3, repr3, ind3] = self.estrinsics(out0, out3)
            print("Estrinsics between cam2 and cam3")
            self.T_guess = np.array([-199, 1, -1])
            [R23, T23, repr23, ind23] = self.estrinsics(out2, out3)
            self.save({'out0': out0,'out1': out1,'out2': out2,'out3': out3,
                       'R1':R1,'T1': T1,'repr1': repr1, 'ind1':ind1,
                       'R2':R2,'T2': T2,'repr2': repr2, 'ind2':ind2,
                       'R3':R3,'T3': T3,'repr3': repr3, 'ind3':ind3,
                       'R23':R23,'T23': T23,'repr23': repr23, 'ind23':ind23},
                      self.out_path)

    def intrinsics(self,prefix):
        # from threading import Thread
        # import matplotlib.pyplot as plt
        name = prefix.split('\\')
        name = name[-1]
        self.set_blob_params(name)
        out = self.IntrinsicsOut(self, prefix)
        imgAr = []
        img = []
        ind = 0
        for imPath in out.imgPaths:
            img = self.read_gray_image(imPath)
            imgAr.append(img)
            ret, centers = cv2.findCirclesGrid(img,
                                               (self.nrows, self.ncols),# patternSize
                                               None, # centers
                                               flags=self.findCirclesGridFlags, # flags
                                               blobDetector=self.findCirclesGridBlobDetector)
            if ret:
                out.imgPoints[ind,:,:] = centers
                out.centers[:,:,ind] = np.reshape(centers,(self.nrows * self.ncols,2))
                out.imgValid[ind] = True

                if self.verbose>0:
                    img2 = cv2.drawChessboardCorners(img, (self.nrows, self.ncols), centers, ret)
                    cv2.circle(img2, tuple(centers[0][0]), 25, (128), -1)
                    if img2.shape[1] > 2000:
                        img2 = cv2.resize(img2, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
                    cv2.imshow('imgOk', img2)
                if self.verbose == 1:
                    cv2.waitKey(self.displayTime)
                if self.verbose>1:
                    cv2.waitKey(0)
                    if self.verbose > 2:
                        print(str(out.centers[0:6, :, ind]))
                        cv2.imwrite(self.imgPath+'preview_'+name+"_{0:02d}.png".format(ind), img2)

                    #cv2.destroyAllWindows()

            else:
                # print('Dot pattern not detected')
                if self.verbose>0:
                    img2 = img
                    if img.shape[1] > 2000:
                        img2 = cv2.resize(img2, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
                    if self.verbose>1:
                        cv2.imwrite(self.imgPath+'discarded_' + name + "_{0:02d}.png".format(ind), img2)
                    cv2.imshow('imgKo', img2)
                    print('Pattern not detected in img {:02d}'.format(ind))
                    if self.verbose == 1:
                        cv2.waitKey(self.displayTime)
                    elif self.verbose>1:
                        print('No dots found in '+str(ind))
                        cv2.waitKey(0)
            ind += 1

        out.CP.resolution = img.shape
        out.objPoints = self.objectPoints
        # objectPointsAr = \
        #     np.repeat(np.reshape(self.objectPoints, (1, self.nrows * self.ncols, 3)), 3, 0)
        # [_, out.CP.K, _, _, _] = cv2.calibrateCamera(objectPointsAr,
        #                                                    out.imgPoints[0:3, :, :],
        #                                                    img.shape, out.CP.K, None, None, None,
        #                                                    self.calibrateCameraFlags)
        objectPointsAr = \
            np.repeat(np.reshape(self.objectPoints, (1, self.nrows * self.ncols, 3)), len(np.where(out.imgValid)[0]), 0)
        # [out.rms, out.CP.K, _, _, _] = cv2.calibrateCamera(objectPointsAr,
        #                                                     out.imgPoints[np.where(out.imgValid)[0], :, :],
        #                                                     img.shape, out.CP.K, None, None, None,
        #                                                     cv2.CALIB_FIX_ASPECT_RATIO)



        [out.rms, out.CP.K, D, out.CP.Rot, out.CP.Tras, _, _, perViewErr] = cv2.calibrateCameraExtended(objectPointsAr,
                                                            out.imgPoints[np.where(out.imgValid)[0], :, :],
                                                            img.shape, out.CP.K, None,
                                                            None, None, # rvecs, tvecs
                                                            None, None, None, # stdDevInt, stDevExt, perViewErr
                                                            cv2.CALIB_FIX_ASPECT_RATIO)
                                                            # cv2.CALIB_USE_INTRINSIC_GUESS)



        rotMat = np.zeros((3,3,len(out.CP.Rot)))
        for i in range(len(out.CP.Rot)):
            rotMat[:,:,i] = cv2.Rodrigues(out.CP.Rot[i][:].T[0])[0]
        out.CP.Rot = rotMat
        imgSelected = np.where(out.imgValid)[0]
        # if name=='ToF':
        imgSelected = imgSelected[np.where(perViewErr<np.mean(perViewErr)+2*np.std(perViewErr))[0]]
        objectPointsAr = \
            np.repeat(np.reshape(self.objectPoints, (1, self.nrows * self.ncols, 3)), len(imgSelected), 0)
        [out.rms, out.CP.K, D, _,_, _, _, _] = cv2.calibrateCameraExtended(objectPointsAr,
                                                            out.imgPoints[imgSelected, :, :],
                                                            img.shape, out.CP.K, D,
                                                            None, None,  # rvecs, tvecs
                                                            None, None, None,
                                                            # stdDevInt, stDevExt, perViewErr
                                                            cv2.CALIB_USE_INTRINSIC_GUESS)
        imgShape = tuple(np.flip(img.shape,0))
        validInd = np.where(out.imgValid)[0]
        for i in validInd:
            tmp = np.reshape(out.imgPoints[i, :, :, :], (1, out.imgPoints.shape[1], 2))
            objPointsUndistorted = cv2.undistortPoints(tmp, out.CP.K, D)
            imgUndistorted = cv2.undistort(imgAr[i],  out.CP.K, D)
            out.imgPointsUnd[i, :, :, :] = np.reshape(objPointsUndistorted[:, :, :], (out.imgPoints.shape[1], 1, 2))
            out.imgUnd[i] = imgUndistorted

        [out.CP.newK, out.CP.roi] = cv2.getOptimalNewCameraMatrix(out.CP.K, D, imgShape, 1, imgShape)
        out.CP.invMapX, out.CP.invMapY = cv2.initUndistortRectifyMap(out.CP.K, D, None, out.CP.newK, imgShape, 5)
        out.validInd = imgSelected
        u = out.CP.resolution[1]
        v = out.CP.resolution[0]
        tmp = np.meshgrid([float(i) for i in range(u)], [float(i) for i in range(v)])
        u = tmp[0]+0.5
        v = tmp[1]+0.5
        uCol = np.reshape(u.T, [u.size, 1])
        vCol = np.reshape(v.T, [v.size, 1])
        uv = np.hstack((uCol, vCol))
        uv = np.array([uv])

        # pts = cv2.undistortPoints(np.array([[[1.0, 1.0], [944, 504]]]), out.CP.K, D)
        pts = cv2.undistortPoints(uv, out.CP.K, D)
        u = np.reshape(pts[0][:, 0], (out.CP.resolution[1], out.CP.resolution[0])).T
        v = np.reshape(pts[0][:, 1], (out.CP.resolution[1], out.CP.resolution[0])).T
        out.CP.mapX = u*out.CP.K[0,0]+out.CP.K[0,2]
        out.CP.mapY = v*out.CP.K[1,1]+out.CP.K[1,2]

        D = np.array(D[0])
        out.CP.R = D[[0, 1, 4]]
        out.CP.D = D[[2, 3]]
        if self.verbose>=0:
            print(str(out))
        return out


    def estrinsics(self,intrOut0,intrOut1):
        indexes = np.intersect1d(intrOut0.validInd,intrOut1.validInd)
        objectPointsAr = \
            np.repeat(np.reshape(self.objectPoints, (1, self.nrows * self.ncols, 3)), len(indexes), 0)
        cp0 = intrOut0.CP
        cp1 = intrOut1.CP
        # originalPoints = intrOut0.imgPoints[indexes,:,:]
        # firstRotation = np.transpose(cp0.Rot)
        # firstTranslation = -cp0.Tras
        if self.verbose>1:
            for i in range(len(indexes)):
                img0 = intrOut0.imgUnd[indexes[i]]
                img1 = intrOut1.imgUnd[indexes[i]]
                if img0.shape[1] > 2000:
                    img0 = cv2.resize(img0, None, fx=0.4, fy=0.4)
                if img1.shape[1] > 2000:
                    img1 = cv2.resize(img1, None, fx=0.4, fy=0.4)
                cv2.imshow('im0',img0)
                cv2.imshow('im1', img1)
                cv2.waitKey(self.displayTime)
        # self.R_guess, _ = cv2.Rodrigues(np.array([0.0, 0.1, 0.0]))
        # try:
        #     tmp = self.T_guess
        # except:
        #     self.T_guess = np.zeros((3,1))
        retval, intrOut0.CP.K, D0, intrOut1.CP.K, D1, R, T, _, _, perViewErrors = cv2.stereoCalibrateExtended(
        # retval, intrOut0.CP.K, D0, intrOut1.CP.K, D1, R, T, _, _ = cv2.stereoCalibrate(
            objectPointsAr,
            intrOut0.imgPoints[indexes,:,:],
            intrOut1.imgPoints[indexes,:,:],
            intrOut0.CP.K,
            np.stack((cp0.R[0],cp0.R[1], cp0.D[0],cp0.D[1], cp0.R[-1])),
            intrOut1.CP.K,
            np.stack((cp1.R[0],cp1.R[1], cp1.D[0],cp1.D[1], cp1.R[-1])),
            intrOut0.CP.resolution,
            self.R_guess,
            self.T_guess,
            criteria = self.StereoCalibrateCriteria,
            flags = self.StereoCalibrateFlags)
        if self.verbose > 0:
            print('Estrinsics:\nR:\n'+str(np.round(R,decimals=4)))
            print('Rodrigues: '+str(np.transpose(np.round(cv2.Rodrigues(R)[0]/np.pi*180,decimals=5))[0])+' deg')
            print('T:\n' + str(np.round(np.reshape(T, (1, 3))[0], decimals=3)))
            print('RMS error: ' + str(np.round(retval,decimals=3)))
            print('Valid images (' + str(len(indexes)) + '/' + str(len(intrOut0.imgPaths)) + '):\n' + str(indexes))
        return [R, T, retval, indexes]

    def save(self, out, path):
        import scipy.io as sio
        import os
        d = path.split('\\')
        d = d[0:-1]
        d = [(d[i] + '\\') for i in range(len(d))]
        d = ''.join(d)
        if not os.path.exists(d):
            os.mkdir(d)
        sio.savemat(path, mdict=out)
        return 0

if __name__ == "__main__":
    import argparse, textwrap
    parser = argparse.ArgumentParser(prog='calib',formatter_class=argparse.RawDescriptionHelpFormatter,
                             description=textwrap.dedent('''\
                                Description:
                                  Computes intrinsics [and estrinsics] of a camera system using a blob pattern.
                                    If run with only 1 prefix (P0), then it only computes intrinsics of the camera.
                                    If run with 2 [or more] prefix (up to 4), then it computes
                                        - intrinsics and distortion vector for all cameras,
                                        - estrinsics of 2nd [3rd, 4th] camera[s] w.r.t. the 1st one,
                                        - estrinsics of 3rd w.r.t. 4th'''),
                             epilog=textwrap.dedent('''\
                             The unit of measurement is mm (for T output vector)
                             A dot pattern has been used                               
                                o   o   o   o   o   o       -
                                  o   o   o   o   o         |
                                o   o   o   o   o   o       |
                                  o   o   o   o   o         |   vertical size
                                o   o   o   o   o   o       |       3.5*L mm
                                  o   o   o   o   o         |
                                o   o   o   o   o   o       |
                                  o   o   o   o   o         -
                                                            
                                |-------------------|
                                    horizontal size
                                        5*L mm
                                L mm
                                |---|
                                
                             For further info: robertocosta2501@gmail.com'''))

    parser.add_argument('ip0', metavar='P0', nargs=1,
                        default='..\img\Pol',
                        help='Prefix of the images taken with the 1st (reference) camera. Default: \'..\img\Pol\'')
    parser.add_argument('out', metavar='OutputMatPath', nargs=1,
                        default='..\mat\calib.mat',
                        help='Path where to save the .mat output file. Default: \'..\mat\calib.mat\'')
    parser.add_argument('-P1', help='Prefix of the images taken with the 2nd camera',dest='ip1')
    parser.add_argument('-P2', help='Prefix of the images taken with the 3rd camera',dest='ip2')
    parser.add_argument('-P3', help='Prefix of the images taken with the 4th camera', dest='ip3')
    parser.add_argument('-v', help='Verbosity: [0, 1, 2]',
                        default=0, type=int, dest='v')
    # NORMALIZATION TYPE:
    #   0   SINGLE IMAGE / FIXED
    #   2   MIN THRESHOLD FOR CAM 0
    #   3   MAX THRESHOLD FOR CAM 0
    #   4   MIN THRESHOLD FOR CAM 1
    #   5   MAX THRESHOLD FOR CAM 1
    #   6   MIN THRESHOLD FOR CAM 0
    #   7   MAX THRESHOLD FOR CAM 0
    #   8   MIN THRESHOLD FOR CAM 1
    #   9   MAX THRESHOLD FOR CAM 1
    # parser.add_argument('--normalization', nargs='?', help='foo help', dest='norm')
    parser.add_argument('-L', help='Distance between 2 aligned dots [mm]', default=67.8, dest='L')
    parser.add_argument('-N', help='Normalization: 0 = Manual, 1 = Auto', default=1, dest='N')

    args = parser.parse_args()
    calib = []
    if args.ip1 == None:
        calib = Calibration(ip0=args.ip0[0],
                            out=args.out[0], verbose=args.v,L=args.L, N=args.N)
    elif args.ip2 == None:
        calib = Calibration(ip0=args.ip0[0], ip1=args.ip1,
                            out=args.out[0], verbose=args.v,L=args.L, N=args.N)
    elif args.ip3 == None:
        calib = Calibration(ip0=args.ip0[0], ip1=args.ip1, ip2=args.ip2,
                            out=args.out[0], verbose=args.v,L=args.L, N=args.N)
    else:
        calib = Calibration(ip0=args.ip0[0], ip1=args.ip1, ip2=args.ip2, ip3=args.ip3,
                            out=args.out[0], verbose=args.v,L=args.L, N=args.N)
