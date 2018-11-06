import numpy as np
import nibabel as nib
import os

def rejectOutliers(data, m=2):
    dx = np.abs(data[0] - np.median(data[0]))
    dy = np.abs(data[1] - np.median(data[1]))
    dz = np.abs(data[2] - np.median(data[2]))
    mdevx = np.median(dx)
    mdevy = np.median(dy)
    mdevz = np.median(dz)
    # if else is preventing from dividing by 0. if the median is 0, all
    # the differences have to be 0 becaues dx, dy, dz don't contain
    # negative numbers
    sx = dx/mdevx if mdevx else 0. 
    sy = dy/mdevy if mdevy else 0.
    sz = dz/mdevz if mdevz else 0.
    logicalx = sx<m
    logicaly = sy<m
    logicalz = sz<m
    e = data*logicalx*logicaly*logicalz
    data_return = np.array(e[:,e[1]!=0])
    return data_return

def dicom2np(path):
    """ 
    Converts a dicom file to a numpy array. 
    Returns img_data, the numpy array and ConstPixelSpacing, 
    a list of the spacing between datapoints in mm for each dimension
    """
    pathDicom = os.path.dirname(path)

    import dicom
    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(pathDicom):
        for filename in fileList:
            if ".dcm" in filename.lower():  # check whether the file's DICOM
                lstFilesDCM.append(os.path.join(dirName,filename))

    # Get ref files
    RefDsFirst = dicom.read_file(lstFilesDCM[0])
    RefDsLast  = dicom.read_file(lstFilesDCM[-1])
    ImagePositionPatientFirst = RefDsFirst.ImagePositionPatient
    ImagePositionPatientLast  = RefDsLast.ImagePositionPatient
    # Load dimensions based on the number of rows, columns, 
    # and slices (along the Z axis)

    ConstPixelDims = (int(RefDsFirst.Rows), int(RefDsFirst.Columns), len(lstFilesDCM))
    
    # The array is sized based on 'ConstPixelDims'
    array = np.zeros(ConstPixelDims, dtype=RefDsFirst.pixel_array.dtype)
    
    # loop through all the DICOM files
    for filenameDCM in lstFilesDCM:
        # read the files
        ds = dicom.read_file(filenameDCM)
        # store the raw image data
        array[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array  
        
    array = array*RefDsFirst.RescaleSlope + RefDsFirst.RescaleIntercept
        
    T1 = np.array(ImagePositionPatientFirst).astype(float)
    TN = np.array(ImagePositionPatientLast).astype(float)
    zaffine = np.divide(T1-TN,1.-float(array.shape[2]))
    zaffinePerp = zaffine[2] #z affine resolved perpendicular to image
    
    # Load spacing values (in mm)
    ConstPixelSpacing = (float(RefDsFirst.PixelSpacing[0]), float(RefDsFirst.PixelSpacing[1]), 
                         float(zaffinePerp))
    

    F1 = np.array(RefDsFirst.ImageOrientationPatient[:3]).astype(float)
    F2 = np.array(RefDsFirst.ImageOrientationPatient[3:6]).astype(float)
    affine = np.zeros((4,4))
    affine[:,1][:3] = -F1*ConstPixelSpacing[0]
    affine[:,0][:3] = -F2*ConstPixelSpacing[1]
    affine[:,2][:3] = zaffine
    affine[:,3][:3] = [-T1[0], -T1[1], T1[2]]
    affine[3,3] = 1.    
    
    return array, ConstPixelSpacing, [-T1[0], -T1[1], T1[2]], lstFilesDCM[0], affine

def nifti2np(pathNifti):
    """ 
    Converts a NIFTI file to a numpy array. 
    Returns img_data, the numpy array and ConstPixelSpacing, 
    a list of the spacing between datapoints in mm for each dimension
    """
    img = nib.load(pathNifti)
    img_data = img.get_data()
    header = img.header
    img_data_shape = img_data.shape
    _extent = (0, img_data_shape[0] - 1, 0, img_data_shape[1] - 1, 0, img_data_shape[2] - 1)
    ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]
    affine = img.header.get_best_affine()
    spacing = header['pixdim'][1:4]
    # Load spacing values
    ConstPixelSpacing = spacing[0], spacing[1], spacing[2]
    return(img_data, ConstPixelSpacing, affine)

def nrrd2np(pathNrrd):
    """ 
    Converts a nrrd file to a numpy array. 
    Returns img_data, the numpy array and ConstPixelSpacing, 
    a list of the spacing between datapoints in mm for each dimension
    """
    import nrrd
    readdata, options = nrrd.read(pathNrrd)
    ConstPixelSpacing = []
    spaceDirections = np.array(options['space directions'])
    ImagePositionPatient = options['space origin']
    for k in range(len(spaceDirections)):
        ConstPixelSpacing.append(np.linalg.norm(spaceDirections[k]))
    spaceDir = np.vstack(options['space directions'])
    spaceOr = options['space origin']
    affine = np.zeros((4,4))
    affine[:3,:3] = spaceDir
    affine[:3, 3] = spaceOr
    affine[3,3] = 1.

    #affine[3,3] = affine[3,3]*-1
    affine[0:2,0:2] = -affine[0:2,0:2]
    affine[0:2,3] = -affine[0:2,3]

    return(readdata, ConstPixelSpacing, affine)


def thresholdnp(array, lo, hi):
    thresholded1 = np.multiply(array, (array>lo).astype(int))
    thresholded2 = np.multiply(thresholded1, (array<hi).astype(int))
    return thresholded2

def orientation(numpy_array): #numpy_array needs to be binary/thresholded
    import b2ac.preprocess
    import b2ac.fit
    import b2ac.conversion
    numpy_array[abs(numpy_array)<0.01] = 0
    numpy_array[~np.isfinite(numpy_array)] = 0
    indices = np.nonzero(numpy_array)
    if len(indices[0])>1:
        x = indices[1]
        y = indices[0]
        z = indices[0] - indices[0]
        xyz = rejectOutliers(np.array([x,y,z]))
        x = xyz[0, :]
        y = xyz[1, :]
        points = np.zeros((len(indices[1]),2))
        for i in range(x.shape[0]):
            points[i, 0] = x[i]
            points[i, 1] = y[i]
        if points.shape[0] > 1:
            try:
                points, x_mean, y_mean = b2ac.preprocess.remove_mean_values(points)
                # Fit using NumPy methods in double precision.
                conic_double = b2ac.fit.fit_improved_B2AC_double(points)
                # Convert from conic coefficient form to general ellipse form.
                general_form_double = b2ac.conversion.conic_to_general_1(conic_double)
                general_form_double[0][0] += x_mean
                general_form_double[0][1] += y_mean
            except:
                general_form_double = 0
        else: general_form_double = 0
    else: general_form_double = 0
    return general_form_double # [x, y], [x_axis, y_axis], angle
    
    
def reshape(array, ConstPixelSpacing):
    """ Reshape array to have cubic voxels of size 1mm^3 """
    '''mind the a, b, c value is adjusted to make each '''

    width = array.shape[0]
    height = array.shape[1]
    depth = array.shape[2]
    a = int((width)*ConstPixelSpacing[0])
    b = int((height)*ConstPixelSpacing[1])
    c = int((depth)*ConstPixelSpacing[2])
    reshapedArray1 = np.zeros((a,height,depth))
    xp = np.linspace(0, (width-1)*ConstPixelSpacing[0], width) #adjust xp as [0,1ConstPixelSpacing,2CPS..
    x  = np.linspace(0, a-1, a)

    for j in range(height):
        for k in range(depth):
            reshapedArray1[:,j,k] = np.interp(x, xp, array[:,j,k])
    reshapedArray2 = np.zeros((a,b,depth))
    yp = np.linspace(0,(height-1)*ConstPixelSpacing[1],height)
    y = np.linspace(0,b-1,b)
    for j in range(a):
        for k in range(depth):
            reshapedArray2[j,:,k] = np.interp(y, yp, reshapedArray1[j,:,k])
    reshapedArray3 = np.zeros((a,b,c))
    zp = np.linspace(0,(depth-1)*ConstPixelSpacing[2],depth)
    z = np.linspace(0,c-1,c)
    for j in range(a):
        for k in range(b):
            reshapedArray3[j,k,:] = np.interp(z, zp, reshapedArray2[j,k,:])
    return reshapedArray3

def ellipses(bone):
    angles             =np.zeros(bone.shape[2])
    xcentroids         =np.zeros(bone.shape[2])
    ycentroids         =np.zeros(bone.shape[2])
    xmajor_axis_points =np.zeros(bone.shape[2])
    ymajor_axis_points =np.zeros(bone.shape[2])

    for i in range(bone.shape[2]):
    #    print 'Slice number {}'.format(i)
        array_i           = bone[:,:,i] # array to pass to 'orientation' must be thresholded already
        orientation_i     = orientation(array_i)
        if orientation_i != 0: 
            angles[i]      =orientation_i[2] # pick out the angle
            xcentroids[i]  =orientation_i[0][0]
            ycentroids[i]  =orientation_i[0][1]
            
        else: pass
    #    else: print 'Slice {} contains 0'.format(i)
    return angles, xcentroids, ycentroids

def selectEllipsesRange(angles):
    # make all angles positive
    angles[angles < 0] += np.pi
    import scipy.signal as signal
    # First, design the Butterworth filter
    N  = 1   # Filter order - the higher the order, the sharper the dropoff
    pd = float(20)  # Cutoff period - the inverse is the cutoff frequency
    fs = 1 # Sample rate frequency
    nyq = 0.5*fs
    low = 1/pd
    wn = low/nyq # Cutoff frequency as a fraction
    b, a = signal.butter(N, wn, output='ba')
    filt_angles = signal.filtfilt(b,a, angles)
    angles_diff = filt_angles-angles
    indices = [] # indices holds the slice range of interest
    for i in range(len(angles)):
#       if abs(angles_diff[i]) < 0.1 and (1.45<abs(angles[i])<1.75 or 1.45-0.5*np.pi<abs(angles[i])<1.75-0.5*np.pi):
        if abs(angles_diff[i]) < 0.1 and 1.4<abs(angles[i])<1.8:
            indices.append(i)
    from itertools import groupby
    z = zip(indices, indices[1:])
    tmp = ([list(j) for i, j in groupby(z, key=lambda x: (x[1] - x[0]) <= 1)])
    tmp = sorted(tmp, key=len) # longest lists at the end
    if len(tmp)>1:
        maxtmp = np.array(max(tmp, key=len))
        slices = list(range(maxtmp[0][0], maxtmp[-1][-1]+1))
        slice_angles = [angles[i] for i in slices]
    else:
        slices = []
        slice_angles = [np.pi/2]
#    if not 0<abs(np.mean(slice_angles))<20/360*2*np.pi or not np.pi/2-20/360*2*np.pi<abs(np.mean(slice_angles))<20/360*2*np.pi+np.pi/2:
#        slice_angles = ['use eyes']
#        slices = ['use eyes']
    return slices, slice_angles


def findPlaneFromEllipses(bone, c1, c2, slices, head_x, head_y, head_angles): 
    import findeyes as fer
    bone[abs(bone)<0.01]    = 0
    bone[~np.isfinite(bone)]= 0
    coord = (c1+c2)/2.0
    coord = np.array([coord[1], coord[0], coord[2]])
    if slices != []:
        print '\n','Using ellipses to find midplane','\n'
        x=[]
        y=[]
        remaining_z = range(bone.shape[2]-1, slices[-1],-1)
        for i in remaining_z:
            x.append(np.mean(np.nonzero(bone[:,:,i])[1]))
            y.append(np.mean(np.nonzero(bone[:,:,i])[0]))
        centroids_array = np.concatenate((np.array([head_x,head_y,slices]).T, np.array([x,y,remaining_z]).T),axis=0).T
        centroids_array[~np.isfinite(centroids_array)]=0
        centroids_array[abs(centroids_array)<0.01] = 0
        centroids_array = rejectOutliers(centroids_array)
        from scipy.optimize import curve_fit
        def f(x,A,B):
            return A*x + B
        A_x1,B_x1 = curve_fit(f, centroids_array[2], centroids_array[0])[0] # your data x, y to fit
        A_y1,B_y1 = curve_fit(f, centroids_array[2], centroids_array[1])[0] # your data x, y to fit

        z = np.median(slices)
        x = A_x1*z + B_x1
        y = A_y1*z + B_y1
#        point that the plane goes through, p
        p = np.array([x, y, z]) # coordinates when z = mean(indices)
        v1 = np.array([A_x1/2, A_y1/2, 1])
        v1 = v1/np.sqrt((np.sum(np.power(v1,2))))
#        print 'v1 = ', v1
        mean_angle = np.mean(head_angles)
        v2 = np.array([np.cos(mean_angle+np.pi/2), np.sin(mean_angle+np.pi/2), 0])
#         v2b = (p - coord)
#         v2b = v2b/(np.sqrt(np.sum(np.power(v2b,2))))
#         v2b = np.nan_to_num(v2b)
#         print v2a, v2b
#         v2 = (v2a+v2b)/2.0
        normal = np.cross(v1,v2)
        normal = normal/np.sqrt((np.sum(np.power(normal,2))))
#         print 'normal before invalid = ',normal
        normal = np.nan_to_num(normal)
        # a plane is ax+by+cz = d - find d
        print coord, v1, v2
        d = np.dot(coord,normal)
        # superpose plane onto slices
        a = normal[0]
        b = normal[1]
        c = normal[2]
        # plane equation is ax+by+cz = d
        # for each slice substitute slice number into z to get equation
    else:

        print '\n','Using eyes to find midplane, as ellipses were unreliable','\n'

        v1 = np.array([0,0,1])
        v2 = np.array([1,0,0])
        normal = np.cross(v1,v2)
        d = np.dot(coord,normal)
        a = normal[0]
        b = normal[1]
        c = normal[2]
    return a,b,c,d,normal


def visualiseSingle(rotatedBone, a,b,c,d, slice_no):
    import plotly as py
    import plotly.graph_objs as go
    py.offline.init_notebook_mode()
    slice_ = rotatedBone[:,:,slice_no]
    heatmap = go.Heatmap(
            z = slice_,
            colorscale=[[0.0, 'rgb(160,160,160)'], 
                        [0.1111111111111111, 'rgb(20,20,20)'], 
                        [0.2222222222222222, 'rgb(40,40,40)'], 
                        [0.3333333333333333, 'rgb(100,100,100)'], 
                        [0.4444444444444444, 'rgb(120,120,120)'], 
                        [0.5555555555555556, 'rgb(140,140,140)'], 
                        [0.6666666666666666, 'rgb(160,160,160)'], 
                        [0.7777777777777778, 'rgb(170,170,170)'], 
                        [0.8888888888888888, 'rgb(250,250,250)'], 
                        [1.0, 'rgb(250,250,250)']]
            )
    x = np.linspace(0,slice_.shape[0], 2)
    y = (d-c*slice_no-a*x)/b
    midline = go.Scatter(
        x = x,
        y = y,
        mode = 'lines'
    )
    layout = go.Layout(
        width = 600,
        height= 600,
    #     title='Fig. 10. Slice number %i' % slice_no,
        title='Fig. 10. Slice of the midplane found', # title for report
        showlegend = False,
            margin= go.Margin(
            l=50,
            r=50,
            b=100,
            t=50,
            pad=4
            ),
#        yaxis=dict(
#            range=[50, 220]
#        ),
#        xaxis=dict(
#            range=[0, 220]
#        )
    )

    data = [heatmap, midline]
    fig = go.Figure(data=data, layout=layout)
    py.offline.iplot(fig)
    return None


def correctPlaneParams(angle1, angle2, n, c1Reshaped, c2Reshaped, pixelspacing):
    import math
            
    ang1rad = angle1/360.*2*np.pi
    ang2rad = angle2/360.*2*np.pi
            
    rotAngle = np.sin(ang2rad)*angle1-90

    while rotAngle<=-45: #skews are never that big
        rotAngle = rotAngle+90

    rotAngle = rotAngle/360.*2*np.pi
    
    a = rotAngle
    
    yrotM = np.array([[np.cos(a),  0,-np.sin(a) ], #scipy's definition on y is the cartesian x axis
                      [0,          1,        0  ],
                      [np.sin(a),  0, np.cos(a)]])
    
#    yrotM = np.array([[1,          0,        0  ], #scipy's definition on y is the cartesian x axis
#                      [0,  np.cos(a), -np.sin(a)],
#                      [0,  np.sin(a), np.cos(a)]])
    a = ang2rad
    
    zrotM = np.asarray([[np.cos(a), -np.sin(a), 0.],
                      [np.sin(a),  np.cos(a), 0.],
                      [0.,          0.      , 1.]])  
    
    R = np.dot(zrotM, yrotM)
    n1 = np.dot(yrotM, n)
    normal = np.dot(R, n)
    n2 = normal
    normal = np.true_divide(normal, pixelspacing)
    normal = normal/np.linalg.norm(normal)

    c = 0.5*(c1Reshaped+c2Reshaped)
    mideyes = np.array([c[1], c[0], c[2]])
    mideyes = np.true_divide(mideyes, pixelspacing)

    d = np.dot(mideyes, normal)
    a = normal[0]
    b = normal[1]
    c = normal[2]
    
    return a,b,c,d,normal, n1, n2


def hist_control(data): # optimise number of bins to get the lowest threshold fraction. A thresholdfraction of 0.1 means that the bin with the 2nd highest frequency will have 0.1xthe frequency of the first highest bin.
    # (1) get rid of frequencies outside a standard deviation of 1. 
    """ 
    @ return: Return array. 1st number is the number of bins, 2nd number is the fraction achieved (where fraction is the 2nd highest frequency divided by the highest frequency, 3rd number is the midbin value of the bin corresponding to the highest frequency. 
    """
    from scipy.stats import norm
    hist, bins= np.histogram(data, bins=20)
    mu,sd = norm.fit(data) #mean and standard deviation
    for i in range(len(hist)):
        if bins[i] < (mu-sd) or bins[i] > (mu+sd):
            hist[i] = 0
    # (2) optimise number of bins to get largest differences between 1st and second highest frequences
    fractions = np.zeros((50,3))
    for num_bins in range(10,60):
        hist, bins = np.histogram(data, bins=num_bins)
        midbins=[(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
        data = np.transpose(np.array([hist,midbins]))
        sorted_ = data[data[:,0].argsort()]
        fractions[num_bins-10][0]=num_bins # number of bins 
        fractions[num_bins-10][1]=sorted_[-2][0]/sorted_[-1][0] # fraction 
        fractions[num_bins-10][2]=sorted_[-1][1] # midbin value
###shall we transpose fractions to find ind?? or else fractions[1] gives the second 
###array in fractions
        fractions_n = np.transpose(fractions)
    ind = np.where(fractions_n[1]==min(fractions_n[1]))[0][0]
    #ind = np.where(fractions[1]==min(fractions[1]))[0][0]
    return fractions[ind]

def locateForamen(Skull_Contour,SliceNumber):
    w = (skull_contour[SliceNumber,:,:]!=False).astype(int)
    sx = ndimage.sobel(w, axis=0, mode='constant')
    sy = ndimage.sobel(w, axis=1, mode='constant')
    sob = np.hypot(sx, sy)
    
    SagCon = dmask[SliceNumber,:,:]
    from skimage.transform import (hough_line, hough_line_peaks,
                                   probabilistic_hough_line)
    import matplotlib.pyplot as plt
    from matplotlib import cm
    image = SagCon
    edges = sob
    lines = probabilistic_hough_line(edges, threshold=10, line_length=20,
                                     line_gap=3)
    cen = ndimage.measurements.center_of_mass(SagCon)
    #Only interested in lines lower than the center of mass
    LowerLines = []
    for line in lines:
        p0,p1 = line
        if p0[0]<cen[1] and p1[0]<cen[1]:
            LowerLines.append(line)
    VLines = []
    for line in LowerLines:
        p0,p1 = line
        if p0[1]<cen[0] and p1[1]<cen[0]:
            VLines.append(line)
    HLines = []
    for line in LowerLines:
        p0,p1 = line
        if p0[1]>cen[0] and p1[1]>cen[0]:
            HLines.append(line)
    # Locate foramen magnum in sag view
    Vlength = []
    for line in VLines:
        p0,p1 = line
        Vlength.append(np.abs(p0[0]-p1[0]))
    VInd = np.argmax(Vlength)
    Vpoints = VLines[VInd]
    if Vpoints[0][0] < Vpoints[1][1]:
        F_1 = [SliceNumber,Vpoints[0][1],Vpoints[0][0]]
    else:
        F_1 = [SliceNumber,Vpoints[1][1],Vpoints[1][0]]

    HInd = np.argmin(np.transpose(HLines)[0])

    if HInd > len(HLines):
        F_2 = (SliceNumber,np.transpose(HLines)[1][1][HInd-len(HLines)],np.transpose(HLines)[0][1][HInd-len(HLines)])
    else:
        F_2 = (SliceNumber,np.transpose(HLines)[1][0][HInd],np.transpose(HLines)[0][0][HInd])
    return F_1,F_2
    