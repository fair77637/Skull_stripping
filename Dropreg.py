import uuid
from subprocess import call
import os.path
import SimpleITK as sitk

class Dropreg:
    """
        This class offers a Python wrapper for the binary Dropreg, originally written by Ben Blocker.

        The absolute path to dropreg binary has to be specified in self.dropregBin
    """

    def __init__(self):
        # ================ COMMON PARAMETERS ==============

        # Binary containing dropreg executable
        self.dropregBin = "/vol/biomedic/users/eferrant/code/oak2/builddrop/drop/apps/dropreg/dropreg"

        # Temporal folder (the usr running the script must be able to write here)
        self.tmpFolder = "/tmp"

        # Output image interpolation (0=NEAREST, 1=LINEAR)
        self.outputImageInterpolationMode = 1

        # Prefix to the last transformation computed by Dropreg. It will be written by "registerFiles" method and used when calling "transformUsingLastResults"
        self.lastTransformationPrefix = ""

        # ======== REGISTRATION: CENTER OF MASS ===========

        # Register only by aligning the center of mass
        self.registerCenterOfMass = True

        # ============ REGISTRATION: LINEAR ===============

        # Register using linear registration (through continuous optimization)
        self.registerLinear = True

        # LINEAR: transformation type (0=RIGID, 1=SIMILARITY, 2=AFFINE)
        self.linearTransformationType = 0

        # LINEAR: similarity measure (0=MAD, 1=CC, 2=ECC)
        self.linearSimilarityMeasure = 0

        # LINEAR: image level factors
        self.linearImageLevels = ['4','4','4','2','2','2']

        # LINEAR: number of iterations per level for the Simplex optimization algorithm
        self.linearIterations = 100

        # LINEAR: image interpolation (0=NEAREST, 1=LINEAR) TODO: Check with Ben this parameter
        self.linearInterpolation =1

        # LINEAR: image sampling probability (value in range [0,1])
        self.linearSamplingProb = 0.05

        # ========== REGISTRATION: NON-LINEAR =============

        # Register using non-linear registration (through discrete optimization)
        self.registerNonlinear = False

        # NONLINEAR: MRF type (0=FIRST_ORDER, 1=SECOND_ORDER) --ntype arg (=0)
        self.nonLinearMRFType = 0

        # NONLINEAR: regularization type (0 = FLUID, 1 = ELASTIC) --nreg arg (=0)
        self.nonLinearRegularizationType = 0

        # NONLINEAR: FFD spacing on first level --nffd arg (=80)
        self.ffdSpacing = 40

        # NONLINEAR: similarity measure (0=MAD, 1=CC, 2=ECC) --nsim arg (=0)
        self.nonLinearSimilarityMeasure = 0

        # NONLINEAR: image level factors --nlevels arg
        self.nonLinearImageLevels = ['4','4','4','2','2','2']

        # NONLINEAR: number of iterations per level (number of times QPBO is called) --niters arg (=10) #TODO: Check with Ben
        self.nonLinearIterations = 10

        # NONLINEAR: image interpolation (0=NEAREST, 1=LINEAR) --ninterp arg (=1)
        self.nonLinearInterpolation = 1

        # NONLINEAR: image sampling probability [0,1] --nsampling arg (=1)
        self.nonLinearSamplingProb = 1.0

        # NONLINEAR: regularization weight --nlambda arg (=0)
        self.nonLinearRegularizationFactor =0.0

    def getTemporalFilename(self):#
        return os.path.join(self.tmpFolder, str(uuid.uuid4()))

    def ensureIsAFilename(self, image):
        """
            This method makes sure the given parameter is a filename. If it is an image, it will save the image in disk to a temporal location and return the corresponding filename.

        :param image: string or ITK image object
        :return: if an image is provided as parameter, then it will be saved in disk to a temp file and its name will be return. If a string is provided as image, the string is returned.
        """

        if not isinstance(image, basestring):
            tempId = self.getTemporalFilename() + ".nii.gz"
            sitk.WriteImage(image, tempId)
            return tempId
        else:
            return image

    def register(self, source, target, output = "", sourceMask = "", targetMask = "", transformation = "", deleteTmpVolumes = True):
        """
            Register images stored in the given files. It will save the results into the "output" filename.
            If no output filename is provided, it returns the image.

                :param source filename of source image
                :param target filename of target image
                :param output [OPTIONAL] filename of output image
                :param sourceMask [OPTIONAL] filename of source mask
                :param targetMask [OPTIONAL] filename of target mask
                :param transformation [OPTIONAL] transformation filename of transformation that will be applied as initialization to the source image

                :param deleteTmpVolumes [OPTIONAL] indicates if the temporal files created to register the images have to be erased or not.
                :return [OPTIONAL] if no output filename is provided, then it will return an ITK image with the deformed source
        """

        if output == "":
            outputFile = os.path.join(self.tmpFolder, str(uuid.uuid4()) + ".nii.gz")
        else:
            outputFile = output

        sourceFilename = self.ensureIsAFilename(source)
        targetFilename = self.ensureIsAFilename(target)

        command = [self.dropregBin, "-s", sourceFilename, '-t', targetFilename, '-o', outputFile]

        if not (sourceMask == ""):
            sourceMaskFilename = self.ensureIsAFilename(sourceMask)
            command.extend(["--smask", sourceMaskFilename])

        if not (targetMask == ""):
            targetMaskFilename = self.ensureIsAFilename(targetMask)
            command.extend(["--tmask", targetMaskFilename])

        if transformation != "":
            command.extend(["--transform", transformation])

        command.extend(["--ointerp", str(self.outputImageInterpolationMode)])

        if self.registerCenterOfMass:
            command.extend(["-c"])

        if self.registerLinear:
            command.extend(["-l"])
            command.extend(["--lsim", str(self.linearSimilarityMeasure)])
            command.extend(["--ltype", str(self.linearTransformationType)])

            command.extend(["--llevels"])
            command.extend(self.linearImageLevels)

            command.extend(["--liters", str(self.linearIterations)])
            command.extend(["--linterp", str(self.linearInterpolation)])
            command.extend(["--lsampling", str(self.linearSamplingProb)])

        if self.registerNonlinear:
            command.extend(["-n"])
            command.extend(["--ntype", str(self.nonLinearMRFType)])
            command.extend(["--nreg", str(self.nonLinearRegularizationType)])
            command.extend(["--nffd", str(self.ffdSpacing)])
            command.extend(["--nsim", str(self.nonLinearSimilarityMeasure)])
            command.extend(["--nlevels"])
            command.extend(self.nonLinearImageLevels)

            command.extend(["--niters", str(self.nonLinearIterations)])
            command.extend(["--ninterp", str(self.nonLinearInterpolation)])
            command.extend(["--nsampling", str(self.nonLinearSamplingProb)])
            command.extend(["--nlambda", str(self.nonLinearRegularizationFactor)])

        print command

        call(command)

        self.lastTransformationPrefix = outputFile.replace(".nii.gz", "")

        if (not isinstance(source, basestring)) and deleteTmpVolumes:
            os.remove(sourceFilename)

        if (not isinstance(target, basestring)) and deleteTmpVolumes:
            os.remove(targetFilename)

        if not (sourceMask == "") and (not isinstance(sourceMask, basestring)) and deleteTmpVolumes:
            os.remove(sourceMaskFilename)

        if not (targetMask == "") and (not isinstance(targetMask, basestring)) and deleteTmpVolumes:
            os.remove(targetMaskFilename)

        if os.path.exists(outputFile) and output == "":
            outputImg = sitk.ReadImage(outputFile)

            if deleteTmpVolumes:
                os.remove(outputFile)

            return outputImg

    def transform(self, image, output = "", referenceImage = "", interpolation = 1, transformation = "", fx = "", fy = "", fz = "", deleteTmpFiles=True):
        """
            It applies a transformation (that was stored in disk) to a given image, and saves the warped image.

                :param image filename of the image that will be transformed
                :param output filename of output image
                :param referenceImage [OPTIONAL] if provided, the output image size and resolution will correspond to the reference image
                :param interpolation [DEFAULT = 1] interpolation used to produce the output image (0 = NEAREST, 1 = LINEAR)
                :param transformation [OPTIONAL] filename of linear transformation
                :param fx, fy, fz [OPTIONAL] Nifti files containing the components of the deformation field
                :param deleteTmpFiles [OPTIONAL] indicates if the temporal files created to register the images have to be erased or not

                :return [OPTIONAL] if no output filename is provided, then it will return the warped ITK image. If provided, then it will write the image in "output"
        """

        if output == "":
            outputFile = os.path.join(self.tmpFolder, str(uuid.uuid4()) + ".nii.gz")
        else:
            outputFile = output

        imageFilename = self.ensureIsAFilename(image)

        command = [self.dropregBin, "-s", imageFilename, '-o', outputFile]

        if not (referenceImage == ""):
            referenceImageFilename = self.ensureIsAFilename(referenceImage)
            command.extend(["-t", referenceImageFilename])
        else:
            command.extend(["-t", imageFilename])

        if transformation != "":
            command.extend(["--transform", transformation])

        if fx != "":
            command.extend(["--fx", fx])

        if fy != "":
            command.extend(["--fy", fy])

        if fz != "":
            command.extend(["--fz", fz])

        command.extend(["--ointerp", str(interpolation)])

        print command
        call(command)

        if (not isinstance(image, basestring)) and deleteTmpFiles:
            os.remove(imageFilename)

        if not (referenceImage == "") and (not isinstance(referenceImage, basestring)) and deleteTmpFiles:
            os.remove(referenceImageFilename)

        if os.path.exists(outputFile) and output == "":
            outputImg = sitk.ReadImage(outputFile)

            if deleteTmpFiles:
                os.remove(outputFile)

            return outputImg

    def transformUsingLastResults(self, image, output="", referenceImage = "", interpolation = 1, deleteTmpFiles=True):
        """
            It applies the transformation estimated during the last registration call, and saves the transformed image to disk.

            If no registration method was used, then

                :param image filename of the image that will be transformed or ITK image
                :param output filename of output image
                :param referenceImage [OPTIONAL] if provided, the output image size and resolution will correspond to the reference image
                :param interpolation [DEFAULT = 1] interpolation used to produce the output image (0 = NEAREST, 1 = LINEAR)

                :return
        """
        print "PREFIX: " + self.lastTransformationPrefix
        return self.transformWithPrefix(image, self.lastTransformationPrefix, output, referenceImage, interpolation, deleteTmpFiles=deleteTmpFiles)

    def transformWithPrefix(self, image, prefix, output="", referenceImage = "", interpolation = 1, deleteTmpFiles=True):
        """
            It applies a transformation (that was stored in disk) to a given image, and saves it in disk.

            The difference with the method "transform" is that, here, you only need to specify the prefix of the transformation that was generated
            by Dropreg. The algorithm will then look for the linear transformation (prefix_transform.txt) and the non linear (prefix_field_x.nii.gz,
            prefix_field_y.nii.gz, prefix_field_z.nii.gz). For example,

                d = Dropreg()
                d.transform(image, "/home/x/img_warped")

            Will look for the linear transformation "/home/x/img_warped_transform.txt" and the non linear "/home/x/img_warped_field_x.nii.gz",
            "/home/y/img_warped_field_y.nii.gz", "/home/x/img_warped_field_z.nii.gz".

            If it does not find a linear or non-linear transform, then it does not apply it.

                :param image filename of the image that will be transformed
                :param output filename of output image
                :param referenceImage [OPTIONAL] if provided, the output image size and resolution will correspond to the reference image
                :param interpolation [DEFAULT = 1] interpolation used to produce the output image (0 = NEAREST, 1 = LINEAR)

                :return
        """

        if os.path.exists(prefix + "_transform.txt"):
            transform =  prefix + "_transform.txt"
        else:
            transform = ""

        if os.path.exists(prefix + "_field_x.nii.gz"):
            fx = prefix + "_field_x.nii.gz"
        else:
            fx = ""

        if os.path.exists(prefix + "_field_y.nii.gz"):
            fy = prefix + "_field_y.nii.gz"
        else:
            fy = ""

        if os.path.exists(prefix + "_field_z.nii.gz"):
            fz = prefix + "_field_z.nii.gz"
        else:
            fz = ""

        return self.transform(image, output, referenceImage, interpolation, transform, fx, fy, fz, deleteTmpFiles)

    def deleteLastTransformationFiles(self):
        """
            This method deletes the temporal files where the last transformation was stored.

            If you dont want to keep the temporal transformation files, you should call this method after every registration call.

            NOTE: After calling this method, it is not possible to run "transformUsingLastResults" unless you have called 'register'.

        :return:
        """
        if not (self.lastTransformationPrefix == ""):
            if os.path.exists(self.lastTransformationPrefix + "_transform.txt"):
                os.remove(self.lastTransformationPrefix + "_transform.txt")

            if os.path.exists(self.lastTransformationPrefix + "_field_z.nii.gz"):
                os.remove(self.lastTransformationPrefix + "_field_x.nii.gz")

            if os.path.exists(self.lastTransformationPrefix + "_field_z.nii.gz"):
                os.remove(self.lastTransformationPrefix + "_field_y.nii.gz")

            if os.path.exists(self.lastTransformationPrefix + "_field_z.nii.gz"):
                os.remove(self.lastTransformationPrefix + "_field_z.nii.gz")

            self.lastTransformationPrefix = ""
