import nibabel as nib
import os.path
from subprocess import call
from ..aux import Common

class DenseCRF3D:
    """
        This class offers a Python wrapper for the binary denseCRF

        The full path to the binary CRF has to be specified in self.denseCRF3DBin

        This code is based on the 3D Dense CRF from https://github.com/Kamnitsask/dense3dCrf
        You can download the code and compile the required binary from the github repository.
    """

    def __init__(self):
        # ================ COMMON PARAMETERS ==============

        # Binary containing dropreg executable
        self.denseCRF3DBin = "/vol/biomedic/users/eferrant/code/dense3dCrf/build/applicationAndExamples/dense3DCrfInferenceOnNiis"

        # numberOfModalitiesAndFiles: (int) number of modalities to use, followed by the full paths-filenames of the corresponding Nifti (or nii.gz) files.
        # -numberOfModalitiesAndFiles
        # 1
        # ../ applicationAndExamples / example / DWI_normalized.nii.gz
        # ../ applicationAndExamples / example / Flair_normalized.nii.gz

        # List containing the channel image files (absolute path)
        self.channels = []

        # numberOfForegroundClassesAndProbMapFiles: (int) number of FOREGROUND classes, followed by full paths-filenames to the corresponding probability maps.
        # The prob-maps will be used for unary potentials!
        # -numberOfForegroundClassesAndProbMapFiles
        # 1
        # ../ applicationAndExamples / example / lesionProbMap.nii.gz

        # List containing the probability map files
        self.probMaps = []

        # imageDimensions: (int) the dimensions of the image (2 for 2D, 3 for 3D)
        # This should be followed by the dimension of the image in the corresponding R-C-Z axes (everything in separate lines, my config-file-parser is bad!)
        #-imageDimensions
        # 3
        # 230
        # 230
        # 154

        # minMaxIntensities: (float) min intensity, and followed by max intensity (in separate lines).
        # All the channels (modalities) should have already been normalised to the same intensity range. The min and max values to use should be given here.
        # Every value below or above these boundaries will be set the to min/max respectively.
        # -minMaxIntensities
        # -3.1
        # +3.1
        # Minimum and maximum values. If not provided, the minimum and maximum value of the image will be used.
        self.minValue = -3.1
        self.maxValue = 3.1

        # outputFolder: output folder, where the results should be saved (segmentation maps and probability maps generated by the CRF). NOTE: Folder should have been created beforehand!
        # -outputFolder
        # ../ applicationAndExamples / example / results /
        # Folder where the output data will be stored
        self.outputFolder = ""

        # prefixForOutputSegmentationMap: Essentially the filename for the resulting segmentation map (default is denseCrf3dOutputSegm). Will be saved as a .nii.gz automatically.
        # -prefixForOutputSegmentationMap
        # denseCrf3dSegmMap
        self.prefixForOutputSegmentationMap = "CRF_SEGM"

        # prefixForOutputProbabilityMaps: Prefix of the filenames with which to save the resulting probability maps (default is denseCrf3dProbMapClass).
        # Each probability map will be saved as "prefix" + numberOfClass + ".nii.gz" automatically.
        # -prefixForOutputProbabilityMaps
        # denseCrf3dProbMapClass
        self.prefixForOutputProbabilityMaps = "CRF_PROB"

        # pRCZandW: please provide 4 sequential floats (separate lines) for pR, pC, pZ, pW.
        # positional-std parameters and the corresponding weight. The higher the stds, the larger the neighbourhood that the pixel is influenced by the nearby pixel-labels.
        # Similary, higher positional-W means the energy function will require nearby voxels to have consistent labels.
        # -pRCZandW
        # 3.0
        # 3.0
        # 3.0
        # 3.0
        self.pRCZandW = ["3.0", "3.0", "3.0", "3.0"]

        # bRCZandW: please provide 4 floats (separate lines) for bR, bC, bZ, bW.
        # bilateral-std parameters (bRCZ) and the corresponding weight (bW). bilateral RCZ are similar to the positional RCZ parameters.
        # But these contribute to the hybrid kernel, that is also influenced by the intensities in the images (-bMods section below). See original paper for more details.
        # bW the bilateral weight that defines the influence of the hybrid kernel (distance of pixel + intensity difference).
        # (Note: Imagine this as defining how far away from a pixel we require the intensities in the values to be rather homogeneous. bW values used were between 3-30, depends on number of modalities used too.)
        # -bRCZandW
        # 17.0
        # 12.0
        # 10.0
        # 5.0
        self.bRCZandW = ["17.0", "12.0", "10.0", "5.0"]

        # -bMods
        # 4.5
        # 3.5
        self.bMods = ["3.5"]

        # -numberOfIterations
        # 2
        self.numberOfIterations = 2


    def inference(self, channelsList, probList, outputDir=""):
        """
            It runs CRF inference using the given channels and probability images.

                :param channelsList List of absolute filenames pointing to the channel (intensity) images.
                :param probList List of absolute filenames pointing to the probability map images (one per class).
                :param outputDir Directory where the resulting segmenatation and probmaps will be stored.
        """
        if outputDir == "":
            outputDir = os.path.split(probList[0])[0] + "/crfOutput/"

        Common.ensureDir(outputDir)

        command = [self.denseCRF3DBin]

        command.extend(["-numberOfModalitiesAndFiles", str(len(channelsList))])
        command.extend(channelsList)


        command.extend(["-numberOfForegroundClassesAndProbMapFiles", str(len(probList))])
        command.extend(probList)

        imgDimensions = nib.load(channelsList[0]).get_data()

        command.extend(["-imageDimensions", "3"])
        command.extend([str(imgDimensions.shape[0]), str(imgDimensions.shape[1]), str(imgDimensions.shape[2])])

        command.extend(["-minMaxIntensities", str(self.minValue), str(self.maxValue)])

        command.extend(["-outputFolder", outputDir])

        command.extend(["-prefixForOutputSegmentationMap", self.prefixForOutputSegmentationMap])

        command.extend(["-prefixForOutputProbabilityMaps", self.prefixForOutputProbabilityMaps])

        command.extend(["-pRCZandW"])
        command.extend(self.pRCZandW)

        command.extend(["-bRCZandW"])
        command.extend(self.bRCZandW)

        command.extend(["-bMods"])
        command.extend(self.bMods)

        command.extend(["-numberOfIterations", str(self.numberOfIterations)])

        print command

        call(command)

#        if os.path.exists(outputFile) and output == "":
#            outputImg = sitk.ReadImage(outputFile)
#            return outputImg
