# noinspection PyPep8
# -*- coding: utf-8 -*-
"""
    Created on Tue Aug 16 17:48:24 2016

    This class implements different operations needed to pre process medical images.

    @author: enzo
"""

import sys
import SimpleITK as sitk
import os
import numpy as np
import matplotlib.pyplot as plt
import NyulNormalizer as nyul
from ..aux.Common import ensureDir, getMedicalImageBasename

from NyulNormalizer import NyulNormalizer
import math
import Dropreg
import scipy.ndimage.filters as filters

def flipData(m, axis):
    """
    Reverse the order of elements in an array along the given axis.
    The shape of the array is preserved, but the elements are reordered.
    .. versionadded:: 1.12.0
    Parameters
    ----------
    m : array_like
        Input array.
    axis : integer
        Axis in array, which entries are reversed.
    Returns
    -------
    out : array_like
        A view of `m` with the entries of axis reversed.  Since a view is
        returned, this operation is done in constant time.
    See Also
    --------
    flipud : Flip an array vertically (axis=0).
    fliplr : Flip an array horizontally (axis=1).
    Notes
    -----
    flip(m, 0) is equivalent to flipud(m).
    flip(m, 1) is equivalent to fliplr(m).
    flip(m, n) corresponds to ``m[...,::-1,...]`` with ``::-1`` at position n.
    Examples
    --------

    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])
    array([[[4, 5],
            [6, 7]],
           [[0, 1],
            [2, 3]]])
    array([[[2, 3],
            [0, 1]],
           [[6, 7],
            [4, 5]]])
    True
    """
    if not hasattr(m, 'ndim'):
        m = np.asarray(m)
    indexer = [slice(None)] * m.ndim
    try:
        indexer[axis] = slice(None, None, -1)
    except IndexError:
        raise ValueError("axis=%i is invalid for the %i-dimensional input array"
                         % (axis, m.ndim))
    return m[tuple(indexer)]



class PreProcessor:
    """ PreProcessor class. It implements several methods to pre process medical images.

    Attributes:
        image The original imagen opened with method 'open'
        fName Complete filename corresponding to the original opened image
    """

    def __init__(self):
        self.image = None
        self.fName = ""
    
    def open(self, fileName):
        """ 
            Loads an image for pre processing.
        """
        self.image = sitk.ReadImage(fileName)
        self.fName = fileName

    @staticmethod
    def getOneHotEncodingFromHardSegmentation(hardSegm, labels):
        """
            Given a hard segmentation, it returns the 1-hot encoding.
        """
        dims = hardSegm.shape

        oneHot = np.ndarray(shape=(1, len(labels), dims[-3], dims[-2], dims[-1]), dtype=np.float32)

        # Transform the Hard Segmentation GT to one-hot encoding
        for i, labelValue in enumerate(labels):
            oneHot[0, i, :, :, :] = (hardSegm == labelValue).astype(np.int16)

        return oneHot

    @staticmethod
    def getHardSegmentationFromTensor(probMap, labels=None):
        """
            Given probMap in the form of a Theano tensor: ( shape = (N, numClasses, H, W, D), float32) or
            ( shape = (numClasses, H, W, D), float32), where the values indicates the probability for every class,
             it returns a hard segmentation per data sample where the value corresponds to the segmentation label with
             the highest probability. The label is sequential from 0 to numClasses-1, unless a labels list is specified.
             If labels is provided, the output labels are linearly mapped: the voxels whose probability is maximum in
                   the c-th channel of the input probMap will be assigned label[c].

            :param probMap: probability maps containing the per-class prediction/
            :param labels: List containing the real label value. If it is provided, then the labels in the output hard segmentation
                   will contain the value provided in 'labels'. The mapping is lineal: the voxels whose probability is maximum in
                   the c-th channel of the input probMap will be assigned label[c].

            :return: if a 5D tensor is provided (N, numClasses, H, W, D), it returns a 4D tensor with shape = (N, H, W, D).
                     if a 4D tensor is provided, it returns a 3D tensor with shape = H, W, D
        """
        if len(probMap.shape) == 5:
            hardSegm = np.argmax(probMap, axis=1).astype(np.int16)
        elif len(probMap.shape) == 4:
            hardSegm = np.argmax(probMap, axis=0).astype(np.int16)
        else:
            return None

        # If labels are provided, replace the values in the matrix by linearly mapping them to the values in the label list.
        if labels is not None:
            finalHardSegm = np.copy(hardSegm)
            for l in range (len(labels)):
                finalHardSegm[hardSegm==l] = labels[l]
            return finalHardSegm
        else:
            return hardSegm


    def fuseLabelsToMultilabelImage(self, labelsToFuse, valuesToAssign, valFalse=0):
        """
            This method fuse labels in the list of lists 'labelsToFuse', assigning the corresponding value in valuesToAssign. Any other label is set to 0.
            For example, if labelsToFuse = [[2,3], [4,5]] and valuesToAssign = [10, 20], then all labels whose original value is 2 or 3 are gonna be assigned 10,
            and those with original value 4 or 5 are gonna be assigned 20. The rest will be 0.
        """

        # Read image data
        data = sitk.GetArrayFromImage(self.image)

        # Create the matrix where we are going to store the fused image
        dataAux = np.ones(data.shape) * valFalse

        # Fuse the labels
        for i in range(0, len(labelsToFuse)):
            for j in labelsToFuse[i]:
                dataAux[data == j] = valuesToAssign[i]

        # Save edited data
        output = sitk.GetImageFromArray(dataAux)
        output.SetSpacing(self.image.GetSpacing())
        output.SetOrigin(self.image.GetOrigin())
        output.SetDirection(self.image.GetDirection())

        return output

    def fuseLabelsIBSR(self, labelsToFuse):
        """
            This method fuse all the labels in labelsToFuse to the value corresponding to their index in the list + 1.
            All the labels which are not in the map, are fused to label 0.

            :param labelsToFuse List of integers or lists. If a list is provided as element, then all the elements in this list
                    are fused to the corresponding index value in the main list. For example, if labelsToFuse = [4, [6,7], 9] then
                     - voxels in image = 4, are assigned value 1
                     - voxels in image = 6 or 7, are assigned value 2
                     - voxels in image = 9 are assgined value 3
                     - any other voxel value is assigned to 0
            :return fused image
        """
        if self.image is not None:
            print "Fusing labels for IBSR dataset..."
            # Read image data
            data = sitk.GetArrayFromImage(self.image)

            # Create the matrix where we are going to store the fused image
            dataAux = np.zeros(data.shape, dtype=np.int16)

            # Fuse the labels
            for index, value in enumerate(labelsToFuse):
                if not isinstance(value, list):
                    dataAux[data == value] = index + 1
                else:
                    # If it is a list, then all values are fused to the corresponding index
                    for v in value:
                        dataAux[data == v] = index + 1

            # Save edited data
            output = sitk.GetImageFromArray(dataAux)
            output.SetSpacing(self.image.GetSpacing())
            output.SetOrigin(self.image.GetOrigin())
            output.SetDirection(self.image.GetDirection())

            print "Done."
            return output
        else:
            print "WARNING: There is not image opened. fuseLabelsIBSR is returning None."
            return None

    def smoothBinaryMask(self, image = None, sigma = 4.0):

        g = sitk.SmoothingRecursiveGaussianImageFilter()
        g.SetSigma(sigma)

        if image is None:
            image = self.image
        imgGauss = g.Execute(image)


        data = sitk.GetArrayFromImage(imgGauss)

        dataAux = np.zeros(data.shape, dtype='uint8')
        dataAux[data >= 0.4] = 1

        output = sitk.GetImageFromArray(dataAux)
        output.SetSpacing(image.GetSpacing())
        output.SetOrigin(image.GetOrigin())
        output.SetDirection(image.GetDirection())

        return output

    def smoothMultilabelMask(self, sigma = 1.0):

        data = sitk.GetArrayFromImage(self.image)

        labels = np.unique(data)
        oneHot = self.getOneHotEncodingFromHardSegmentation(data, labels)

        for i in range(oneHot.shape[1]):
            oneHot[0,i,:,:,:] = filters.gaussian_filter(np.squeeze(oneHot[0,i,:,:,:]), sigma)

        finalSeg = self.getHardSegmentationFromTensor(np.squeeze(oneHot), labels)

        output = sitk.GetImageFromArray(finalSeg)

        output.SetSpacing(self.image.GetSpacing())
        output.SetOrigin(self.image.GetOrigin())
        output.SetDirection(self.image.GetDirection())

        return output


    def retainLargestConnectedComponent(self, image):
        """
            Retains only the largest connected component of a binary image, and returns it.
        """    
        connectedComponentFilter = sitk.ConnectedComponentImageFilter()
        objects = connectedComponentFilter.Execute(image)

        # If there is more than one connected component        
        if connectedComponentFilter.GetObjectCount() > 1:
            objectsData = sitk.GetArrayFromImage(objects)
            
            # Detect the largest connected component            
            maxLabel = 1
            maxLabelCount = 0            
            for i in range(1, connectedComponentFilter.GetObjectCount() + 1):
                componentData = objectsData[objectsData == i]
                                                
                if len(componentData.flatten()) > maxLabelCount:
                    maxLabel = i
                    maxLabelCount = len(componentData.flatten())
            
            # Remove all the values, exept the ones for the largest connected component
            
            dataAux = np.zeros(objectsData.shape, dtype=np.int8) 
    
            # Fuse the labels
            
            dataAux[objectsData == maxLabel] = 1

            # Save edited data    
            output = sitk.GetImageFromArray( dataAux )
            output.SetSpacing( image.GetSpacing() )
            output.SetOrigin( image.GetOrigin() )
            output.SetDirection( image.GetDirection() )
        else:
            output = image
            
        return output    

    def flip(self, flipArray):
        """ 
            Flips an image in the directions indicated by the boolean list flipArray = [boolX, boolY, boolZ].
            For example, flip([True, False, False]) flips and returns the image in the X direction.
        """
        if self.image is not None:
            print "Flipping..."
            flipFilter = sitk.FlipImageFilter()
            array = sitk.VectorBool(flipArray)
            
            # I had to activate FlipAboutOrigin because otherwise, when saving the images,
            # they remain in the orientation when visualized using ITK-Snap and MITK
            flipFilter.FlipAboutOriginOn()
                        
            flipFilter.SetFlipAxes(array)
            print "Done"
            return flipFilter.Execute(self.image)
            
        else:
            print "WARNING: There is not image opened. flip is returning None."
            return None

    def flipImageData(self, fileToFlip, output, axis = 1):
        img = nib.load(fileToFlip)
        flipped_image = nib.Nifti1Image(flipData(img.get_data(), axis=axis), affine=img.affine)
        nib.save(flipped_image, output)

    def resampleToMatchImage(self, targetImage, interpolator=0):
        """
            Resamples an image to match spacing and resolution of another image.

            Optional Arguments:
                newSpacing  (default = [1.,1.,1.]) New spacing in mm.
                interpolator (default 0) 0: linear interpolator, 1: nearest neighbor interpolator (useful for binary masks), 2: Bspline

            Note: this method is based on the code by: https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/blob/master/popi_utilities_setup.py
        """
        if self.image is not None:
            print "Resampling..."

            targetSpacing = targetImage.GetSpacing()
            targetSize = targetImage.GetSize()

            if interpolator == 1:
                interpolatorFunc = sitk.sitkNearestNeighbor
            elif interpolator == 2:
                interpolatorFunc = sitk.sitkBSpline
            elif interpolator == 3:
                interpolatorFunc = sitk.sitkLabelGaussian
            else:
                interpolatorFunc = sitk.sitkLinear

            resampledImg = sitk.Resample(self.image, list(targetSize), sitk.Transform(),
                                         interpolatorFunc, self.image.GetOrigin(),
                                         list(targetSpacing), self.image.GetDirection(), 0.0,
                                         self.image.GetPixelIDValue())
            print "Done."
            return resampledImg
        else:
            print "WARNING: There is not image opened. resample is returning None."
            return None

    def resample(self, newSpacing = [1.,1.,1.], interpolator = 0):
        """ 
            Resamples an image so that it has the same real size, but a new spacing given by [spXmm, spYmm, spZmm].
            By default, the new spacing is 1mm isotropic: [1mm, 1mm, 1mm].
            
            Optional Arguments:
                newSpacing  (default = [1.,1.,1.]) New spacing in mm.
                interpolator (default 0) 0: linear interpolator, 1: nearest neighbor interpolator (useful for binary masks), 2: Bspline
                
            Note: this method is based on the code by: https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/blob/master/popi_utilities_setup.py
        """
        if self.image is not None:
            print "Resampling..."

            originalSpacing = self.image.GetSpacing()
            originalSize = self.image.GetSize()
    
            newSize = [int(math.ceil(originalSize[0]*(originalSpacing[0]/newSpacing[0]))),
                    int(math.ceil(originalSize[1]*(originalSpacing[1]/newSpacing[1]))),
                    int(math.ceil(originalSize[2]*(originalSpacing[2]/newSpacing[2])))]
    
            if interpolator == 1:
                interpolatorFunc = sitk.sitkNearestNeighbor
            elif interpolator == 2:
                interpolatorFunc = sitk.sitkBSpline
            elif interpolator == 3:
                interpolatorFunc = sitk.sitkLabelGaussian
            else:
                interpolatorFunc = sitk.sitkLinear            
            
            resampledImg = sitk.Resample(self.image, newSize, sitk.Transform(), 
                                      interpolatorFunc, self.image.GetOrigin(),
                                      newSpacing, self.image.GetDirection(), 0.0, 
                                      self.image.GetPixelIDValue())
            print "Done."
            return resampledImg
        else:
            print "WARNING: There is not image opened. resample is returning None."
            return None            

    def thresholdValues(self, lower = 0.0, upper = 999999999, value = 0.0):
        """ 
            Thresholds an image, assigning the value 'value' to those pixels which are below or avobe the 'threshold' value.
            
            Arguments:
                thresholdType 0: pixels below 'threshold' are assigned to 'value', 1: pixels above 'threshold' are assigned to 'value'.
                
        """
        if self.image is not None:
            print "Thresholding..."
            
            thresholdedImage = sitk.Threshold(self.image, lower, upper, value)
            print "Done"
            return thresholdedImage
        else:
            print "WARNING: There is not image opened. thresholdValues is returning None."
            return None         


    def padVolumeToMakeItMultipleOf(self, v, multipleOf = 3, mode='symmetric'):

        padding = ((0, 0 if v.shape[0] % multipleOf == 0 else multipleOf - (v.shape[0] % multipleOf)),
                   (0, 0 if v.shape[1] % multipleOf == 0 else multipleOf - (v.shape[1] % multipleOf)),
                   (0, 0 if v.shape[2] % multipleOf == 0 else multipleOf - (v.shape[2] % multipleOf)))

        return np.pad(v, padding, mode)

    def cropRoiAndMakeItMultipleOf(self, roiImage, multipleOf=3):
            """
                Crops the image in the area corresponding to the smallest cube that contains the ROI,
                and guarantees that the size of the cropped image is multiple of a given number.
            """
            if self.image is not None:
                # Extract the stats from the ROI indicated by label 1 in the mask
                roi = sitk.GetArrayFromImage(roiImage)

                img = sitk.GetArrayFromImage(self.image)

                minCoords = np.min(np.argwhere(roi == 1), axis=0)
                maxCoords = np.max(np.argwhere(roi == 1), axis=0)

                size = maxCoords - minCoords

                extraSize = np.array([0 if size[0] % multipleOf == 0 else multipleOf - (size[0] % multipleOf),
                                      0 if size[1] % multipleOf == 0 else multipleOf - (size[1] % multipleOf),
                                      0 if size[2] % multipleOf == 0 else multipleOf - (size[2] % multipleOf)])

                maxCoords = maxCoords + extraSize

                res = img[minCoords[0]:maxCoords[0], minCoords[1]:maxCoords[1], minCoords[2]:maxCoords[2]]

                if (res.shape[0] % multipleOf != 0) or (res.shape[1] % multipleOf != 0) or (res.shape[2] % multipleOf != 0):
                    res = self.padVolumeToMakeItMultipleOf(res)

                # Save edited data
                croppedImage = sitk.GetImageFromArray(res)
                croppedImage.SetSpacing(self.image.GetSpacing())
                croppedImage.SetOrigin(self.image.GetOrigin())
                croppedImage.SetDirection(self.image.GetDirection())

                return croppedImage

    def normalize(self, roiFilename=None, backgroundValue=None):
        """
            Returns a normalized image (mean = 0, variance = 1). If a binary ROI is specified,
            the image is normalized using the mean and variance calculated from the ROI (where label is 1).

        """
        if self.image is not None:
            if roiFilename is None:
                print "Normalizing..."
                normalizedImage = sitk.Normalize(self.image)
                print "Done"
                return normalizedImage
            else:
                # Extract the stats from the ROI indicated by label 1 in the mask
                roi = sitk.ReadImage(roiFilename)

                roiStats = sitk.LabelStatisticsImageFilter()
                roiStats.Execute(self.image, roi)

                roiMean = roiStats.GetMean(1)
                roiSigma = roiStats.GetSigma(1)
                print "Mean: " + str(roiMean) + "\tSigma: " + str(roiSigma)

                data = sitk.GetArrayFromImage(self.image)

                # Create the matrix where we are going to store the fused image
                dataAux = data.astype(float)
                dataAux = dataAux - roiMean
                dataAux = dataAux / roiSigma

                # Set the given value to the background if provided
                if backgroundValue is not None:
                    dataRoi = sitk.GetArrayFromImage(roi)
                    dataAux[dataRoi == 0] = backgroundValue

                # Save edited data
                normalizedImage = sitk.GetImageFromArray(dataAux)
                normalizedImage.SetSpacing(self.image.GetSpacing())
                normalizedImage.SetOrigin(self.image.GetOrigin())
                normalizedImage.SetDirection(self.image.GetDirection())

                # Normalize the image by subsctracting the mean and dividing by sigma
                #                normalizedImage = sitk.ShiftScale(self.image, shift = -1.0 * roiMean, scale = float(1.0 / float(roiSigma)))

                return normalizedImage

        else:
            print "WARNING: There is not image opened. normalize is returning None."
            return None

    def normalizeGivenMeanStd(self, mean, std, roiFilename=None, backgroundValue=None):
        """
            Normalizes the image using the given mean and std ((img - mean) / std). If a binary ROI is specified,
            voxels where roi == 0 will be assigned "backgroundValue" if provided.

        """
        if self.image is not None:
            data = sitk.GetArrayFromImage(self.image)

            # Create the matrix where we are going to store the fused image
            dataAux = data.astype(float)
            dataAux = dataAux - float(mean)
            dataAux = dataAux / float(std)

            if roiFilename is not None:
                # Extract the stats from the ROI indicated by label 1 in the mask
                roi = sitk.ReadImage(roiFilename)

                # Set the given value to the background if provided
                if backgroundValue is not None:
                    dataRoi = sitk.GetArrayFromImage(roi)
                    dataAux[dataRoi == 0] = backgroundValue

            # Save edited data
            normalizedImage = sitk.GetImageFromArray(dataAux)
            normalizedImage.SetSpacing(self.image.GetSpacing())
            normalizedImage.SetOrigin(self.image.GetOrigin())
            normalizedImage.SetDirection(self.image.GetDirection())

            # Normalize the image by subsctracting the mean and dividing by sigma
            #                normalizedImage = sitk.ShiftScale(self.image, shift = -1.0 * roiMean, scale = float(1.0 / float(roiSigma)))

            return normalizedImage

        else:
            print "WARNING: There is not image opened. normalize is returning None."
            return None

    def getMeanStd(self, roiFilename = None):
        """
            It returns the mean and std of the image. If roiFilename is provided, only voxels where roi>0 will be considered.

        :return: mean, std
        """

        if self.image is not None:
            data = sitk.GetArrayFromImage(self.image)

            if roiFilename is not None:
                mask = sitk.ReadImage(roiFilename)
                maskData = sitk.GetArrayFromImage(mask)

                filteredData = data[maskData > 0]
            else:
                filteredData = data

            filteredMean = filteredData.mean()
            filteredStd = filteredData.std()

            return filteredMean, filteredStd
        else:
            print "WARNING: There is not image opened. getMeanStd is returning None."
            return None

    def normalizeCT(self, roiFilename = None):
        """ 
            Returns a normalized image (mean = 0, variance = 1). If a binary ROI is specified,
            the image is normalized using the mean and variance calculated from the ROI (where label is 1).
        """
        if self.image is not None:
            if roiFilename is None:
                print "Normalizing..."
                normalizedImage = sitk.Normalize(self.image)
                print "Done"
                return normalizedImage 
            else:
                # Extract the stats from the ROI indicated by label 1 in the mask
                roi = sitk.ReadImage(roiFilename)                
                roiData = sitk.GetArrayFromImage( roi )
                
#                roiStats = sitk.LabelStatisticsImageFilter()
#                roiStats.Execute(self.image, roi)
#                               
#                roiMean = roiStats.GetMean(1)
#                roiSigma = roiStats.GetSigma(1)
#                print "Mean: " + str(roiMean)
#                print "Sigma: "+ str(roiSigma)

                data = sitk.GetArrayFromImage( self.image )

                dataInROI = data[roiData > 0]
                roiMean = dataInROI.mean()                
                roiStd = dataInROI.std()           

#                plt.figure(dpi=100)
#        

                # Compute the histogram of the pixels in the ROI
                numberOfBins = 128
                histo, bins = np.histogram(dataInROI.flatten(), numberOfBins, normed=True)
                
                # Calculate the cumulative distribution function of the original histogram
                cdf = nyul.getCdf(histo)
                # Get the value corresponding to the percitile 85th
                val, b = nyul.getPercentile(cdf, bins, 85)
#        
#                #plt.hist(handRoi.flatten(),range=(min(handRoi), max(handRoi)), bins=128, normed=True)
#                plt.plot(bins[:-1],histo)
#                plt.plot([val], [max(histo)] ,'r^')    
#                
#                plt.show()
                
                filteredData = dataInROI[dataInROI < val] 
                filteredMean = filteredData.mean()
                filteredStd = filteredData.std()
                
                print "ROI mean: " + str(roiMean)
                print "ROI std: " + str(roiStd)

                print "Filtered mean: " + str(filteredMean)
                print "Filtered std: " + str(filteredStd)

                # Create the matrix where we are going to store the fused image
                dataAux = data.astype(float)
                dataAux = dataAux - filteredMean
                dataAux = dataAux / filteredStd
#                
#                # Save edited data    
                normalizedImage = sitk.GetImageFromArray( dataAux )
                normalizedImage.SetSpacing( self.image.GetSpacing() )
                normalizedImage.SetOrigin( self.image.GetOrigin() )
                normalizedImage.SetDirection( self.image.GetDirection() )
                
                # Normalize the image by subsctracting the mean and dividing by sigma
#                normalizedImage = sitk.ShiftScale(self.image, shift = -1.0 * roiMean, scale = float(1.0 / float(roiSigma)))
                
                return normalizedImage

        else:
            print "WARNING: There is not image opened. normalize is returning None."
            return None         


    def trainNyul(self, listFiles, listMasks=[], trainedModelOutput = "./nyulModel"):
        """
            It learns the standard scale for a given set of images, and stores them
            in an output file.
        """
        print "Normalizing using Nyul's method..."
            
        nyul = NyulNormalizer()
        nyul.train(listFiles, listMasks)
        nyul.saveTrainedModel(trainedModelOutput)
        
    def normalizeNyul(self, trainedModel, maskFilename = None):
        if self.image is not None:
            print "Normalizing using Nyul's method..."

            mask = sitk.ReadImage(maskFilename)
            nyul = NyulNormalizer()
            nyul.loadTrainedModel(trainedModel)

            normalizedImage = nyul.transform(self.image, mask)

            print "Done"
            return normalizedImage
        else:
            print "WARNING: There is not image opened. normalizeNyul is returning None."
            return None         

    def stripImage(self, maskFileName, backgroundValue = None):
        """
            It strips the image, leaving only the pixels whose corresponding value in the mask is 1

            :param mask: Binary mask used to indicate which pixels are going to be conserved (mask[voxel]==1) and which will be removed
            :return: It returns the filtered image
        """
        mask = sitk.ReadImage(maskFileName)
        dataImage = sitk.GetArrayFromImage(self.image)
        dataMask = sitk.GetArrayFromImage(mask)

        dataInROI = dataImage[dataMask > 0]

        if backgroundValue is None:
            backgroundValue = -4.0 * dataInROI.std()

        strippedDataMatrix = np.where(dataMask > 0, dataImage, backgroundValue)

        # Save edited data
        output = sitk.GetImageFromArray(strippedDataMatrix)
        output.SetSpacing(self.image.GetSpacing())
        output.SetOrigin(self.image.GetOrigin())
        output.SetDirection(self.image.GetDirection())

        return output

    def stripImageGivenImageMask(self, mask, backgroundValue = None):
        """
            It strips the image, leaving only the pixels whose corresponding value in the mask is 1.
            It takes as parameter an image instead of a filename

            :param mask: Binary mask used to indicate which pixels are going to be conserved (mask[voxel]==1) and which will be removed
            :return: It returns the filtered image
        """

        dataImage = sitk.GetArrayFromImage(self.image)
        dataMask = sitk.GetArrayFromImage(mask)

        dataInROI = dataImage[dataMask > 0]

        if backgroundValue is None:
            backgroundValue = -4.0 * dataInROI.std()

        strippedDataMatrix = np.where(dataMask > 0, dataImage, backgroundValue)

        # Save edited data
        output = sitk.GetImageFromArray(strippedDataMatrix)
        output.SetSpacing(self.image.GetSpacing())
        output.SetOrigin(self.image.GetOrigin())
        output.SetDirection(self.image.GetDirection())

        return output

    def skullStrippingCTHead(self, numErode = 5, numDilate = 15, thresholdMethod = 1):
        """
           It extracts a binary mask from the input CT image and it implements a basic stripping pipeline based on morphological operators.
           It returns a mask of the head.
           The CT image must be normalized with mean=0 and std=1 (using normalize() method).
           
           The pipeline is:
               - thresholding (thresholdMethod = 1 is Otsu thresholding, thresholdMethod = 0 means all values > 0 are assigned to foreground)
               - binary fill hole
               - morphological closign operation (3 times)
               - binary fill hole
               - erosion (numErode times)
               - dilate (numDilate times)
               - binary fill hole
            
            It returns the skull mask, which is usually oversegmenting the skull.
        """
        if self.image is not None:
            print "Skull Stripping CT..."
            
            if thresholdMethod == 0:
                data = sitk.GetArrayFromImage( self.image )
                
                # Create the matrix where we are going to store the fused image
                dataAux = np.zeros(data.shape, dtype='uint8')
                
                # Fuse the labels
                dataAux[data > 0.0] = 1
            
                # Save edited data    
                output = sitk.GetImageFromArray( dataAux )
                output.SetSpacing( self.image.GetSpacing() )
                output.SetOrigin( self.image.GetOrigin() )
                output.SetDirection( self.image.GetDirection() )
                mask = output
            else:
                otsuFilter = sitk.OtsuThresholdImageFilter()
                otsuFilter.SetInsideValue(0)
                otsuFilter.SetOutsideValue(1)
                mask = otsuFilter.Execute(self.image)
                mask = sitk.BinaryFillhole(mask)           
            
            for i in range(0, 3):
                mask = sitk.BinaryMorphologicalClosing(mask) 

            mask = sitk.BinaryFillhole(mask)

            for i in range(0, numErode):
                mask = sitk.ErodeObjectMorphology(mask)

#            mask = sitk.BinaryMorphologicalClosing(mask)

#            for i in range(0, numErode):
#                mask = sitk.ErodeObjectMorphology(mask)
#
#            mask = sitk.BinaryFillhole(mask)
#
            for i in range(0, numDilate):
                mask = sitk.DilateObjectMorphology(mask)
#            
            mask = sitk.BinaryFillhole(mask)
            mask = self.retainLargestConnectedComponent(mask)

            print "Done"
            return mask 
        else:
            print "WARNING: There is not image opened. skullStrippingCT is returning None."
            return None


    def skullStrippingCTBrain(self, numErode = 5, numDilate = 15, initialThresholdMethod = 1, finalThresholdValue=0, useLevelSets=True, useAtlas=True, generateIntermediateVolumes=False):
        """
           It extracts a binary mask from the input CT image and it implements a more complete stripping pipeline than skullStrippingCTHead,
           based on morphological operators and level sets.

           It returns the brain mask.
           
           The CT image must be normalized with mean=0 and std=1 (using normalize() method).
           
           IMPORTANT: if your image CT image has a bimodal histogram, set initialThresholdMethod to 1.
                 If it doesn't, set this parameter to 0.
                 
           The pipeline is:
                - Otsu thresholding
                - binary fill hole
                - morphological closign operation (3 times)
                - binary fill hole
                - erosion (4 times)
                - dilate (9 times)
                - binary fill hole
                - Skull detection through histogram cropping (85th percentile)
                - Undersegmented brain mask generation by sustracting the skull from the head and eroding/dilating the resulting image
                - Border map extraction from the original image using smoothing, gradient magnitude and a sigmoid filter
                - Level sets initialized with the undersegmented brain mask
                - I finally extract the largest connected component and keep it as the brain mask (sometimes other parts of the head, 
                  or even the arms, are also segmented, that's why extracting the largest connected component is required).

            :param generateIntermediateVolumes Indicates if the intermediate steps are stored or not
            It returns the brain mask
        """
        #atlasFile = "/vol/biomedic/users/eferrant/data/samcook_for_atlas_construction/atlas/atlas3_fliped.nii.gz"
        atlasFile = "/vol/biomedic/users/eferrant/data/atlas/atlas3.nii.gz"
        atlasMaskFile = "/vol/biomedic/users/eferrant/data/atlas/atlas3_brainmaskLV.nii.gz"

        if self.image is not None:

            imgBase = self.image

            if useAtlas:
                print "Registering atlas to complete missing skull in case of craniotomy..."

                # Configure the registration algorithm
                d = Dropreg.Dropreg()
                d.linearTransformationType = 2
                d.linearSimilarityMeasure = 1
                d.nonLinearSimilarityMeasure = 1
                d.nonLinearRegularizationFactor = 0.5

                d.linearSamplingProb = 1.0
                d.registerNonlinear = True

                # Step 1: Register image including skull in the data term computation

                imgToSegment = self.image
                atlasImg = sitk.ReadImage(atlasFile)
                atlasMaskImg = sitk.ReadImage(atlasMaskFile)

                deformedAtlas = d.register(atlasImg, imgToSegment)
                deformedMask = d.transformUsingLastResults(atlasMaskImg, referenceImage = imgToSegment, interpolation=0)
                d.deleteLastTransformationFiles()

                # Step 2: Register image only the brain of the atlas in the data term computation
                d.registerCenterOfMass = False
                d.registerLinear = True
                d.registerNonlinear = False

                deformedAtlas2 = d.register(deformedAtlas, imgToSegment, sourceMask=deformedMask)
                d.deleteLastTransformationFiles()

                imgData = sitk.GetArrayFromImage(self.image)
#                imgRegData = sitk.GetArrayFromImage(deformedAtlas)
                imgRegData = sitk.GetArrayFromImage(deformedAtlas2)

                #maxData = np.maximum(imgData, imgRegData)
                maxData = np.copy(imgData)
                #maxData[imgRegData > 0] = ((imgData + imgRegData) / 2.0)[imgRegData > 0]
                maxData[imgRegData > 0] = np.maximum(imgData, imgRegData)[imgRegData > 0]

                imgBaseAux = sitk.GetImageFromArray(maxData)
                imgBaseAux.SetSpacing(imgBase.GetSpacing())
                imgBaseAux.SetOrigin(imgBase.GetOrigin())
                imgBaseAux.SetDirection(imgBase.GetDirection())

                imgBase = imgBaseAux

                if generateIntermediateVolumes:
                    self.writeExtendingFilename(imgBase, extendingName="_withCompleteSkullTEST", subDir="stripping")

#                subFilter = sitk.SubtractImageFilter()
#                diff = subFilter.Execute(imgBase, self.image)
#                diff = sitk.Abs(diff)
#                self.writeExtendingFilename(diff, extendingName="_diffImage", subDir="")

            origData = sitk.GetArrayFromImage(self.image)
            data = sitk.GetArrayFromImage(imgBase)

            print "Skull Stripping CT..."
            print "Creating binary initalization..."

            # This is a manual hack to deal with images where the contrast between brain and no brain is not good.
            # These images usually don't have a bimodal histogram, therefore otsu doesnt work well.
            # If you are dealing with a CT image without a bimodal histogram, use initialThresholdMethod = 0
            if initialThresholdMethod == 0:
                origData = sitk.GetArrayFromImage( imgBase )
                # Create the matrix where we are going to store the fused image
                dataAux = np.zeros(data.shape, dtype='uint8')
                
                dataAux[(data > 0.0) & (data < 95.0)] = 1
            
                # Save edited data    
                output = sitk.GetImageFromArray( dataAux )
                output.SetSpacing( imgBase.GetSpacing() )
                output.SetOrigin( imgBase.GetOrigin() )
                output.SetDirection( imgBase.GetDirection() )
                mask = output
            else:
                data = sitk.GetArrayFromImage( imgBase )

                otsuFilter = sitk.OtsuThresholdImageFilter()
                otsuFilter.SetInsideValue(0)
                otsuFilter.SetOutsideValue(1)

                # IMPORTANT: The Otsu is performed on the original image (not the fused with the mask).
                mask = otsuFilter.Execute(self.image)
                mask = sitk.BinaryFillhole(mask)           
            
            print "Refining binary initialization..."            
            
            for i in range(0, 3):
                mask = sitk.BinaryMorphologicalClosing(mask) 

            mask = sitk.BinaryFillhole(mask)

            for i in range(0, numErode):
                mask = sitk.ErodeObjectMorphology(mask)

            for i in range(0, numDilate):
                mask = sitk.DilateObjectMorphology(mask)

            mask = sitk.BinaryFillhole(mask)

            if generateIntermediateVolumes:
                self.writeExtendingFilename(image=mask, extendingName="_0_headWithSkull",
                                       subDir="stripping")
            # ===  Detect the skull as those voxels whose intensity is above the 85th percentile
            print "Detecting the skull..."                                               

            roiMaskMatrix = sitk.GetArrayFromImage( mask )
            dataInROI = origData[roiMaskMatrix > 0]

            # Compute the histogram of the pixels in the ROI
            numberOfBins = 128
            histo, bins = np.histogram(dataInROI.flatten(), numberOfBins, normed=True)
                
            # Calculate the cumulative distribution function of the original histogram
            cdf = nyul.getCdf(histo)
            # Get the value corresponding to the percitile 85th
            val, b = nyul.getPercentile(cdf, bins, 85)

            
            skullMaskMatrix = np.where(data > val, roiMaskMatrix, 0)

            # Reconstruct the mask of the skull
            skullMask = sitk.GetImageFromArray( skullMaskMatrix )
            skullMask.SetSpacing( imgBase.GetSpacing() )
            skullMask.SetOrigin( imgBase.GetOrigin() )
            skullMask.SetDirection( imgBase.GetDirection() )

            # Obtain the head without the skull
            headWithoutSkullMask = sitk.And(mask, sitk.BinaryNot(skullMask))

            if generateIntermediateVolumes:
                self.writeExtendingFilename(image=headWithoutSkullMask, extendingName="_1_headWithoutSkull",
                                       subDir="stripping")

            # This is a manual hack to del with images where the contrast between brain and no brain is not good.
            # These images usually don't have a bimodal histogram, therefore otsu doesnt work well.
            if useLevelSets == 1:
                # Erode and dilate to eliminate the skull
                for i in range(0, 7):
                    headWithoutSkullMask = sitk.ErodeObjectMorphology(headWithoutSkullMask)            
    
                for i in range(0, 7):
                    headWithoutSkullMask = sitk.DilateObjectMorphology(headWithoutSkullMask)

                headWithoutSkullMask = self.retainLargestConnectedComponent(headWithoutSkullMask)

                if generateIntermediateVolumes:
                    self.writeExtendingFilename(image=headWithoutSkullMask, extendingName="_2_initialization_withAtlas", subDir="stripping")
#                print "Writing the intermediate ..."

                print "Constructing feature maps for the level sets..."                                               
                # Create a border map from the original image using smoothing, gradient magnitude and a sigmoid filter
                # This section is based on the following Python Notebook: https://nbviewer.jupyter.org/github/jon-young/medicalimage/blob/master/Liver%20Segmentation%203D.ipynb
                timeStep_, conduct, numIter = (0.04, 9.0, 5)
                imgRecast = sitk.Cast(imgBase, sitk.sitkFloat32)
                curvDiff = sitk.CurvatureAnisotropicDiffusionImageFilter()
                curvDiff.SetTimeStep(timeStep_)
                curvDiff.SetConductanceParameter(conduct)
                curvDiff.SetNumberOfIterations(numIter)
                imgFilter = curvDiff.Execute(imgRecast)
    
                sigma_ = 2.0
                imgGauss = sitk.GradientMagnitudeRecursiveGaussian(image1=imgFilter, sigma=sigma_)
                K1, K2 = 18.0, 8.0
                alpha_ = (K2 - K1)/6
                beta_ = (K1 + K2)/2
                
                sigFilt = sitk.SigmoidImageFilter()
                sigFilt.SetAlpha(alpha_)
                sigFilt.SetBeta(beta_)
                sigFilt.SetOutputMaximum(1.0)
                sigFilt.SetOutputMinimum(0.0)
                imgSigmoid = sigFilt.Execute(imgGauss)

                if generateIntermediateVolumes:
                    self.writeExtendingFilename(image=imgSigmoid, extendingName="_3_sigmoid_withoutAtlas", subDir="stripping")
#                print "Writing the intermediate ..."

                print "Segmenting through level sets..."                                               
                
                gac = sitk.GeodesicActiveContourLevelSetImageFilter()
                gac.SetPropagationScaling(1.0)
                gac.SetCurvatureScaling(0.2)
                #gac.SetCurvatureScaling(4)
                gac.SetAdvectionScaling(3.0)
                gac.SetMaximumRMSError(0.01)
                gac.SetNumberOfIterations(200)
                
                headWithoutSkullMask = sitk.Cast(headWithoutSkullMask, sitk.sitkFloat32) * -1 + 0.5
                
                gac3D = gac.Execute(headWithoutSkullMask, sitk.Cast(imgSigmoid, sitk.sitkFloat32))
                
                print "Thresholingd the level set..."

                if generateIntermediateVolumes:
                    self.writeExtendingFilename(image=gac3D, extendingName="_4_gac3D", subDir="stripping")
#                print "Writing the intermediate ..."

                levelSetResultMatrix = sitk.GetArrayFromImage( gac3D )
                    
                # Create the matrix where we are going to store the fused image
                dataAux = np.zeros(data.shape, dtype='uint8')
                    
                # Fuse the labels
                dataAux[levelSetResultMatrix < finalThresholdValue] = 1

                # Save edited data
                output = sitk.GetImageFromArray( dataAux )
                output.SetSpacing( imgBase.GetSpacing() )
                output.SetOrigin( imgBase.GetOrigin() )
                output.SetDirection( imgBase.GetDirection() )

#                print "Writing the intermediate ..."
                if generateIntermediateVolumes:
                    self.writeExtendingFilename(image=output, extendingName="_5_gac3D_afterThresholding", subDir="stripping")
            else:
                # Erode and dilate to eliminte the skull
                for i in range(0, 4):
                    headWithoutSkullMask = sitk.ErodeObjectMorphology(headWithoutSkullMask)            
    
                for i in range(0, 5):
                    headWithoutSkullMask = sitk.DilateObjectMorphology(headWithoutSkullMask)     
                
                output = headWithoutSkullMask

            print "Filling final holes..."                                               
            
            for i in range(0,3):
                output = sitk.BinaryFillhole(output)           

#            print "Writing the intermediate ..."
            if generateIntermediateVolumes:
                self.writeExtendingFilename(image=output, extendingName="_6_afterBinaryFillhole", subDir="stripping")

            print "Keeping only the largest connected component..."
            output = self.retainLargestConnectedComponent(output)

            if generateIntermediateVolumes:
                self.writeExtendingFilename(image=output, extendingName="_7_afterRetainLargeComponent", subDir="stripping")

            return output
            
            print "Done"
#            return mask 
        else:
            print "WARNING: There is not image opened. skullStrippingCT is returning None."
            return None

    def skullStrippingMRI(self, method = 0, useLevelsets=False, finalThresholdValue = 1, generateIntermediateVolumes = False, initErosionSteps=3):
        """
           It extracts a binary mask from the input MRI image by thresholding with Otsu's method
           and then filling holes.
           
               Method 0 (default): Values > 0 are marked as foreground
               Method 1: Based on morhological opeartors
               
           It returns the skull mask.
        """
        if self.image is not None:
            if method == 1:
                print "Skull Stripping MRI using morphological operators..."
                otsuFilter = sitk.OtsuThresholdImageFilter()
                otsuFilter.SetInsideValue(0)
                otsuFilter.SetOutsideValue(1)
                mask = otsuFilter.Execute(self.image)
                mask = sitk.BinaryFillhole(mask)
                mask = sitk.DilateObjectMorphology(mask)
                mask = sitk.DilateObjectMorphology(mask)
                mask = sitk.DilateObjectMorphology(mask)
                mask = sitk.ErodeObjectMorphology(mask)
                data = sitk.GetArrayFromImage(self.image)
            else:
                print "Skull Stripping MRI by thresholding values > 0..."
                data = sitk.GetArrayFromImage( self.image )
                
                # Create the matrix where we are going to store the fused image
                dataAux = np.zeros(data.shape, dtype='uint8')
                
                # Fuse the labels
                dataAux[data > 0.0] = 1
            
                # Save edited data    
                output = sitk.GetImageFromArray( dataAux )
                output.SetSpacing( self.image.GetSpacing() )
                output.SetOrigin( self.image.GetOrigin() )
                output.SetDirection( self.image.GetDirection() )
                mask = output

            if useLevelsets:
                # Erode and dilate to eliminate the skull
                for i in range(0, initErosionSteps):
                    mask = sitk.ErodeObjectMorphology(mask)

                headWithoutSkullMask = mask
                if generateIntermediateVolumes:
                    self.writeExtendingFilename(image=headWithoutSkullMask, extendingName="_2_initialization", subDir="stripping")
                    #                print "Writing the intermediate ..."

                imgBase = self.image

                print "Constructing feature maps for the level sets..."
                # Create a border map from the original image using smoothing, gradient magnitude and a sigmoid filter
                # This section is based on the following Python Notebook: https://nbviewer.jupyter.org/github/jon-young/medicalimage/blob/master/Liver%20Segmentation%203D.ipynb
                timeStep_, conduct, numIter = (0.04, 9.0, 10)
                imgRecast = sitk.Cast(imgBase, sitk.sitkFloat32)
                curvDiff = sitk.CurvatureAnisotropicDiffusionImageFilter()
                curvDiff.SetTimeStep(timeStep_)
                curvDiff.SetConductanceParameter(conduct)
                curvDiff.SetNumberOfIterations(numIter)
                imgFilter = curvDiff.Execute(imgRecast)

                if generateIntermediateVolumes:
                    self.writeExtendingFilename(image=imgFilter, extendingName="_3_curvatureAnisotropic",
                                                subDir="stripping")

                sigma_ = 1.5
                imgGauss = sitk.GradientMagnitudeRecursiveGaussian(image1=imgFilter, sigma=sigma_)
                K1, K2 = 18.0, 8.0
                alpha_ = (K2 - K1) / 6
                beta_ = (K1 + K2) / 2

                sigFilt = sitk.SigmoidImageFilter()
                sigFilt.SetAlpha(alpha_)
                sigFilt.SetBeta(beta_)
                sigFilt.SetOutputMaximum(1.0)
                sigFilt.SetOutputMinimum(0.0)
                imgSigmoid = sigFilt.Execute(imgGauss)

                if generateIntermediateVolumes:
                    self.writeExtendingFilename(image=imgSigmoid, extendingName="_3_sigmoid_withoutAtlas",
                                                subDir="stripping")
                    #                print "Writing the intermediate ..."

                print "Segmenting through level sets..."

                gac = sitk.GeodesicActiveContourLevelSetImageFilter()
                gac.SetPropagationScaling(1.0)
                gac.SetCurvatureScaling(2)
                # gac.SetCurvatureScaling(4)
                gac.SetAdvectionScaling(3.0)
                gac.SetMaximumRMSError(0.001)
                gac.SetNumberOfIterations(200)

                headWithoutSkullMask = sitk.Cast(headWithoutSkullMask, sitk.sitkFloat32) * -1 + 0.5

                gac3D = gac.Execute(headWithoutSkullMask, sitk.Cast(imgSigmoid, sitk.sitkFloat32))

                print "Thresholingd the level set..."

                if generateIntermediateVolumes:
                    self.writeExtendingFilename(image=gac3D, extendingName="_4_gac3D", subDir="stripping")
                    #                print "Writing the intermediate ..."

                levelSetResultMatrix = sitk.GetArrayFromImage(gac3D)

                # Create the matrix where we are going to store the fused image
                dataAux = np.zeros(data.shape, dtype='uint8')

                # Fuse the labels
                dataAux[levelSetResultMatrix < finalThresholdValue] = 1

                # Save edited data
                output = sitk.GetImageFromArray(dataAux)
                output.SetSpacing(imgBase.GetSpacing())
                output.SetOrigin(imgBase.GetOrigin())
                output.SetDirection(imgBase.GetDirection())

                #                print "Writing the intermediate ..."
                if generateIntermediateVolumes:
                    self.writeExtendingFilename(image=output, extendingName="_5_gac3D_afterThresholding",
                                                subDir="stripping")

                mask = output
                mask = self.retainLargestConnectedComponent(mask)
                mask = self.smoothBinaryMask(mask)

            print "Done"
            return mask
        else:
            print "WARNING: There is not image opened. skullStrippingMRI is returning None."
            return None

    def stripMalibo(self):
        """
           It extracts a binary mask from a full body image of the Malibo dataset.

               Method 0 (default): Values > 0 are marked as foreground
               Method 1: Based on morhological opeartors

           It returns the skull mask.
        """
        if self.image is not None:
            print "Skull Stripping MRI by thresholding values > 0..."
            data = sitk.GetArrayFromImage(self.image)

            # Create the matrix where we are going to store the fused image
            dataAux = np.zeros(data.shape, dtype='uint8')

            # Fuse the labels
            dataAux[data > 10.0] = 1

            # Save edited data
            output = sitk.GetImageFromArray(dataAux)
            output.SetSpacing(self.image.GetSpacing())
            output.SetOrigin(self.image.GetOrigin())
            output.SetDirection(self.image.GetDirection())
            mask = output

            mask = sitk.ErodeObjectMorphology(mask)
            mask = sitk.DilateObjectMorphology(mask)
            mask = sitk.DilateObjectMorphology(mask)
            mask = sitk.DilateObjectMorphology(mask)
            mask = sitk.DilateObjectMorphology(mask)
            mask = sitk.BinaryFillhole(mask)
            mask = sitk.DilateObjectMorphology(mask)
            mask = sitk.DilateObjectMorphology(mask)
            mask = sitk.DilateObjectMorphology(mask)
            mask = sitk.ErodeObjectMorphology(mask)
            mask = sitk.ErodeObjectMorphology(mask)


            print "Done"
            return mask
        else:
            print "WARNING: There is not image opened. skullStrippingMRI is returning None."
            return None

    def writeExtendingFilename(self, image = None, extendingName="_preproc", subDir="preproc"):
        """ 
            Writes a new image file, extending the filename with 'extendingName' and storing it 
            in a subdirectory (relative to where the original opened image was stored) specified by preproc.
            
            If an image is specified, the method acts as a procedural one (it saves the given image instead of the
            class image).

            :return It returns the filename that was finally used to write the file
        """
        if image is None:
            image = self.image
            folder, filename = os.path.split(self.fName)
        else:
            if self.fName == "":
                filename = "output.nii.gz"
                folder = "./"
            else:
                folder, filename = os.path.split(self.fName)
                
        folder = os.path.join(folder, subDir)     
            
        ensureDir(folder + os.sep)
                         
        if filename.endswith(".gz"):
            filename = os.path.splitext(os.path.splitext(filename)[0])[0] + extendingName + os.path.splitext(os.path.splitext(filename)[0])[1] + os.path.splitext(filename)[1]
        else:
            filename = os.path.splitext(filename)[0] + extendingName + os.path.splitext(filename)[1]

        print "Writing to: " + os.path.join(folder, filename) + " ..."

        sitk.WriteImage( image, os.path.join(folder, filename) )
        print "Done"
        return os.path.join(folder, filename)

    def writeReplacingPartOfFilename(self, image = None, extendingName="_preproc", originalPath="/", replacingPath="/", verbose=True):
        """ 
            Writes a new image file. The ouput path and filename are created by repalcing 'originalPath' with 'replacingPath' 
            and extending the extending the filename with 'extendingName' 
            
            If an image is specified, the method acts as a procedural one (it saves the given image instead of the
            class image). 
        """
        if image is None:
            image = self.image
            folder, filename = os.path.split(self.fName)
        else:
            if self.fName == "":
                filename = "output.nii.gz"
                folder = "./"
            else:
                folder, filename = os.path.split(self.fName)

        folder = folder.replace(originalPath, replacingPath)
        filename = filename.replace(originalPath, replacingPath)
        
        ensureDir(folder + os.sep)
                         
        if filename.endswith(".gz"):
            filename = os.path.splitext(os.path.splitext(filename)[0])[0] + extendingName + os.path.splitext(os.path.splitext(filename)[0])[1] + os.path.splitext(filename)[1]
        else:
            filename = os.path.splitext(filename)[0] + extendingName + os.path.splitext(filename)[1]

        if verbose:
            print "Writing to: " + os.path.join(folder, filename) + " ..."

        sitk.WriteImage( image, os.path.join(folder, filename) )
        if verbose:
            print "Done"


    def writeOverwrite(self):
        """ 
            Overwrites a the original image file with the current state of self.image.
        """
        print "Overwriting: " + self.fName + " ..."

        sitk.WriteImage( self.image, self.fName )
        print "Done"

    def showStats(self, maskFileName, gtFilename, labels):
        # Process image
        dataImage = sitk.GetArrayFromImage(self.image)

        allMean = dataImage.mean()
        allStd = dataImage.std()

        # Process mask
        mask = sitk.ReadImage(maskFileName)
        dataMask = sitk.GetArrayFromImage(mask)

        dataInROI = dataImage[dataMask > 0]

        roiMean = dataInROI.mean()
        roiStd = dataInROI.std()

        #Process GT
        gt = sitk.ReadImage(gtFilename)
        dataGt = sitk.GetArrayFromImage(gt)
        labelsAvgMean = []

        toPrint = self.fName + "," + str(allMean) + "," + str(allStd) + "," + str(roiMean) + "," + str(roiStd)

        for l in range(len(labels)):
            dataLabel = dataImage[(dataGt == labels[l]) & (dataMask == 1)]
            if len(dataLabel) == 0:
                labelsAvgMean.append((dataLabel.mean(), dataLabel.std()))
            else:
                labelsAvgMean.append((np.nan, np.nan))

            toPrint = toPrint + "," + str(dataLabel.mean()) + "," + str(dataLabel.std())

        print toPrint


    def getStats(self, maskFileName, gtFilename, labels):
        # Process image
        dataImage = sitk.GetArrayFromImage(self.image)

        allMean = dataImage.mean()
        allStd = dataImage.std()

        # Process mask
        mask = sitk.ReadImage(maskFileName)
        dataMask = sitk.GetArrayFromImage(mask)

        dataInROI = dataImage[dataMask > 0]

        roiMean = dataInROI.mean()
        roiStd = dataInROI.std()

        #Process GT
        gt = sitk.ReadImage(gtFilename)
        dataGt = sitk.GetArrayFromImage(gt)

        labelsMeans = np.zeros(len(labels))
        labelsStd = np.zeros(len(labels))

        for l in range(len(labels)):
            dataLabel = dataImage[(dataGt == labels[l]) & (dataMask == 1)]
            if len(dataLabel) != 0:
                labelsMeans[l] = dataLabel.mean()
                labelsStd[l] = dataLabel.std()
            else:
                labelsMeans[l] = np.nan
                labelsStd[l] = np.nan

        return labelsMeans, labelsStd

    def getVoxelIntensitiesPerLabel(self, maskFileName, gtFilename):
        # Process image
        voxelListPerLabel = []

        dataImage = sitk.GetArrayFromImage(self.image)

        # Process mask
        mask = sitk.ReadImage(maskFileName)
        dataMask = sitk.GetArrayFromImage(mask)

        dataInROI = dataImage[dataMask > 0]

        #Process GT
        gt = sitk.ReadImage(gtFilename)
        dataGt = sitk.GetArrayFromImage(gt)
        labels = np.unique(dataGt)

        for l in range(labels.shape[0]):
            dataLabel = dataImage[(dataGt == labels[l]) & (dataMask == 1)]
            voxelListPerLabel.append(list(dataLabel.flatten()))

        return voxelListPerLabel

    def showHistogram(self, title = "Histogram", saveToDir='', maskFileName=None, gtFileName=None, trimTales = 1, rangeX=None):
        """
            Shows a histogram in matplotlib given the current image.
            Parameters:

            if saveToDir is provided, it saves the image instead of showing it
            if maskFileName is provided, only voxels in the mask are analysed
            if gtFileName is provided, per label histograms are superimposed

        """
        dataImage = sitk.GetArrayFromImage(self.image)

        if maskFileName is not None:
            # Process mask
            mask = sitk.ReadImage(maskFileName)
            dataMask = sitk.GetArrayFromImage(mask)

            data = dataImage[dataMask > 0]
        else:
            data = dataImage

        max = data.max()
        min = data.min()
        numberOfBins = 128

        histo, bins = np.histogram(data.flatten(), numberOfBins, normed=True)

        # Calculate the cumulative distribution function of the original histogram
        cdf = nyul.getCdf(histo)
        # Get the value corresponding to the percitile 85th
        valMin, b = nyul.getPercentile(cdf, bins, 0 + trimTales)
        valMax, b = nyul.getPercentile(cdf, bins, 100 - trimTales)

        filteredData = data.flatten()[(data.flatten() > valMin) & (data.flatten() < valMax)]

        fig = plt.figure(dpi=100)
        plt.title(title)

        plt.hist(filteredData.flatten(), range=(valMin, valMax), bins=numberOfBins, alpha=0.5, label="All voxels")

        if gtFileName is not None:
            gt = sitk.ReadImage(gtFileName)
            gtData = sitk.GetArrayFromImage(gt)

            labels = np.unique(gtData)

            for l in range(0, labels.shape[0]):
                dataLabel = dataImage[(gtData == labels[l]) & (dataMask == 1)]
                print str(dataLabel.min())
                print str(valMin) + ", " + str(valMax)

                plt.hist(dataLabel.flatten()[(dataLabel.flatten() > valMin) & (dataLabel.flatten() < valMax)], range=(valMin, valMax), bins=numberOfBins, alpha=0.5, label='L' + str(labels[l]))

            axes = plt.gca()

            if rangeX is not None:
                axes.set_xlim(rangeX)

            axes.set_ylim([0,100000])


            plt.legend(loc='upper right')
        if saveToDir != '':
            fig.savefig(os.path.join(saveToDir, os.path.split(os.path.dirname(title))[1] + "_" + getMedicalImageBasename(title) + ".png"))
        else:
            plt.show()

    #def getBinaryBorder(self):


    def showAxial(self, img=None, sliceNumber = None):
        """ 
            Shows the axial slice the opened image. If no sliceNumber is indicated, the slice in 
            the middle is shown.
        """
        if img is None:
            imgAux = self.image
        else:
            imgAux = img    

        if imgAux is not None:        
            imageSize = imgAux.GetSize()

            if sliceNumber is None:
                sliceNumber = imageSize[2]//2

            #imagedisplay.myshow(imgAux[:,:,sliceNumber])
            
        else:
            print "WARNING: The image is None"

    def matchHistogramTo(self, referenceImage):
        refImg = sitk.ReadImage(referenceImage)

        return sitk.HistogramMatching(self.image, refImg)



# FLIP CT IMAGE:
# img = nib.load("/vol/biomedic/users/eferrant/data/samcook_for_atlas_construction/atlas/atlas3.nii.gz")
# flipped_image = nib.Nifti1Image(flip(img.get_data(), axis=1), affine=img.affine)
# nib.save(flipped_image, "/vol/biomedic/users/eferrant/data/samcook_for_atlas_construction/atlas/atlas3_fliped.nii.gz")


import nibabel as nib
if __name__ == "__main__":
    # img = nib.load("/vol/biomedic/users/eferrant/data/samcook_for_atlas_construction/atlas/atlas3.nii.gz")
    # flipped_image = nib.Nifti1Image(flip(img.get_data(), axis=1), affine=img.affine)
    # nib.save(flipped_image, "/vol/biomedic/users/eferrant/data/samcook_for_atlas_construction/atlas/atlas3_fliped.nii.gz")

    imgFname = "/vol/biomedic/users/eferrant/data/flairProblem/flairProblem.nii"

    pp = PreProcessor()
    pp.open(imgFname)
    pp.image = pp.skullStrippingMRI(method=0,useLevelsets=True,finalThresholdValue=1,generateIntermediateVolumes=True)
    pp.writeExtendingFilename(extendingName="_mask", subDir="")


