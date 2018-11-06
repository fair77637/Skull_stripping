import numpy as np
import core
import matplotlib.pyplot as plt
import matplotlib.image

def single_slice(axis_no, thresholded_np, slice_no):
    if axis_no == 0:
        matplotlib.image.imsave('img.png', thresholded_np[slice_no,:,:]) 
    elif axis_no == 1:
        matplotlib.image.imsave('img.png', thresholded_np[:,slice_no,:]) 
    elif axis_no == 2:
        matplotlib.image.imsave('img.png', thresholded_np[:,:,slice_no]) 
    else:
        print 'axis_no must be 0, 1 or 2'
        return None
    img = cv2.imread('img.png')
    img = cv2.resize(img, None, fx=2, fy=2)
    img = cv2.medianBlur(img,3)
    cimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(cimg,cv.CV_HOUGH_GRADIENT,1,50, param1=50,param2=30,minRadius=5,maxRadius=30)
    try:
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    except:
        pass
    plt.clf()
    plt.imshow(cimg)
    plt.axis('equal')
    plt.grid('on')
    return plt.show()

def allSlices(axis_no, softtissue):
    import matplotlib
    import cv2
    import cv2.cv as cv
    circlesData = np.zeros((softtissue.shape[axis_no], 2, 3))
    for i in range(softtissue.shape[axis_no]):
        # to change dimension, change where i is 
        if axis_no == 0:
            matplotlib.image.imsave('img.png', softtissue[i,:,:]) 
            img = cv2.imread('img.png')
        elif axis_no == 1:
            matplotlib.image.imsave('img.png', softtissue[:,i,:]) 
            img = cv2.imread('img.png')
        elif axis_no == 2:
            matplotlib.image.imsave('img.png', softtissue[:,:,i]) 
            img = cv2.imread('img.png')
        else:
            print 'axis_no must be 0, 1 or 2'
            return None            
        img = cv2.resize(img, None, fx=2, fy=2)
        img = cv2.medianBlur(img,3)
        cimg= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(cimg,cv.CV_HOUGH_GRADIENT,1,50, 
                               param1=50,param2=30,minRadius=5,maxRadius=30)
        try:
            circlesData[i,:,:] = circles[0][0:2]
        except:
            pass
    return circlesData


def indexedData(circlesData):
    # hist_data is the same size as circles_data but has an additional column (for the explicit 
    # slice number). hist_data is x,y,z,r

    histData2c = np.zeros((circlesData.shape[0], circlesData.shape[1], circlesData.shape[2]+1))
    histData3c = histData2c
    histData4c = histData2c
    
    numCircles = 2
    for i in range(histData2c.shape[0]):
        histData2c[i,0:numCircles, 0:2] = circlesData[i,0:numCircles, 0:2] 
        #first 2 cols of every slice in circles data assigned to first 2 cols of 
        # every slice in hist_data
        histData2c[i,0:numCircles, 3] = circlesData[i,0:numCircles, 2] 
        # 3rd col of every slice in circles data assigned to 4rd col of every slice 
        # in hist_data (radii data)
        histData2c[i,0:numCircles,2]= i*2 
        # fill in index and stretch by factor of 2
        
    numCircles = 3
    for i in range(histData3c.shape[0]):
        histData3c[i,0:numCircles, 0:2] = circlesData[i,0:numCircles, 0:2] 
        histData3c[i,0:numCircles, 3] = circlesData[i,0:numCircles, 2] 
        histData3c[i,0:numCircles,2]= i*2 
    numCircles = 4
    for i in range(histData4c.shape[0]):
        histData4c[i,0:numCircles, 0:2] = circlesData[i,0:numCircles, 0:2] 
        histData4c[i,0:numCircles, 3] = circlesData[i,0:numCircles, 2] 
        histData4c[i,0:numCircles,2]= i*2 

    return histData2c, histData3c, histData4c

def reshape(hist_data0, hist_data1, hist_data2):
        # if axis_no is 0: 1st column is z, 2nd column is y, 3rd column is x. 
        # if axis_no is 1: 1st column is z, 2nd column is x, 3rd column is y.
        # if axis_no is 2: 1st column is y, 2nd column is x, 3rd column is z.
    x0 = hist_data0[:,:,2].ravel()
    y0 = hist_data0[:,:,1].ravel()
    z0 = hist_data0[:,:,0].ravel()
    r0 = hist_data0[:,:,3].ravel()

    x1 = hist_data1[:,:,1].ravel()
    y1 = hist_data1[:,:,2].ravel()
    z1 = hist_data1[:,:,0].ravel()
    r1 = hist_data1[:,:,3].ravel()

    x2 = hist_data2[:,:,1].ravel()
    y2 = hist_data2[:,:,0].ravel()
    z2 = hist_data2[:,:,2].ravel()
    r2 = hist_data2[:,:,3].ravel()

    x = np.append(x0,np.append(x1,x2))
    y = np.append(y0,np.append(y1,y2))
    z = np.append(z0,np.append(z1,z2))
    r = np.append(r0,np.append(r1,r2))

    hist_data = np.array([x,y,z,r]).T
    return hist_data

def circlesData(softtissue, numCircles):
    """ Fixes hist_data dimensions """
    
    circlesData0 = allSlices(0, softtissue) # axis 0 circle detection
    histData02c, histdata03c, histdata04c = indexedData(circlesData0)

    circlesData1 = allSlices(1, softtissue) # axis 1 circle detection
    histData12c, histdata13c, histdata14c = indexedData(circlesData1)

    circlesData2 = allSlices(2, softtissue) # axis 1 circle detection
    histData22c, histdata23c, histdata24c = indexedData(circlesData2)
    
    histData2 = reshape(histData02c,histData12c,histData22c)
    histData3 = reshape(histdata03c, histdata13c, histdata23c)
    histData4 = reshape(histdata04c, histdata14c, histdata24c)
 
    return histData2, histData3, histData4

def hist3d(hist_data):
    H, edges = np.histogramdd(hist_data[:, 0:3]) 

    # remove all the data points on the axes 
    H[0,:,:] = 0
    H[:,0,:] = 0
    H[:,:,0] = 0

    flat_H = H.flatten()

    mid_edges_x = np.zeros((len(edges[0])-1))
    mid_edges_y = np.zeros((len(edges[1])-1))
    mid_edges_z = np.zeros((len(edges[2])-1))

    for i in range(len(mid_edges_x)):
        mid_edges_x[i] = (edges[0][i]+edges[0][i+1])/2
    for i in range(len(mid_edges_y)):
        mid_edges_y[i] = (edges[1][i]+edges[1][i+1])/2
    for i in range(len(mid_edges_z)):
        mid_edges_z[i] = (edges[2][i]+edges[2][i+1])/2

    z = np.tile(mid_edges_z,len(mid_edges_y)*len(mid_edges_x))
    y = np.tile(np.repeat(mid_edges_y, len(mid_edges_z)), len(mid_edges_x))
    x = np.repeat(mid_edges_x, len(mid_edges_x)*len(mid_edges_z))

    data = np.array([x,y,z,flat_H])

    for i in range(data.shape[0]):
        data = np.array(data[:,data[i]!=0])
    return data, H, edges

def hist3dAll(softtissue):
    
    # circle_num should always be greater than 1. circle_num = 1 means 2 circles being picked out. 
    
    histData2, histData3, histData4 = circlesData(softtissue, 1) # 2c = 2 circles
#    print '\n','Circle detection for 2 circles complete','\n'

#    histData3 = circlesData(softtissue, 2) 
#    print '\n','Finding eyes part 2 of 3 complete','\n'

#    histData4 = circlesData(softtissue, 3)
#    print '\n','Finding eyes part 3 of 3 complete','\n'

    data2c, H2c, edges2c = hist3d(histData2)
    data3c, H3c, edges3c = hist3d(histData3)
    data4c, H4c, edges4c = hist3d(histData4)
    H = H2c+H3c+H4c
    return H, edges2c, histData2

def ranges(H,edges):
    ind = np.dstack(np.unravel_index(np.argsort(H.ravel()), H.shape))
    index_1 = ind[:,-1,:][0] # x, y, z indices of 1st eye socket 
    index_2 = ind[:,-2,:][0] # x, y, z indices of 2nd eye socket
    certainty = (H[ind[:,-3,:][0][0], ind[:,-3,:][0][1], ind[:,-3,:][0][2]])/(
        H[ind[:,-2,:][0][0], ind[:,-2,:][0][1],ind[:,-2,:][0][2]])
    firstEyeRange = np.array([[edges[0][index_1[0]], edges[0][index_1[0]+1]], 
                       [edges[1][index_1[1]], edges[1][index_1[1]+1]],
                       [edges[2][index_1[2]], edges[2][index_1[2]+1]]])
    secondEyeRange = np.array([[edges[0][index_2[0]], edges[0][index_2[0]+1]], 
                       [edges[1][index_2[1]], edges[1][index_2[1]+1]],
                       [edges[2][index_2[2]], edges[2][index_2[2]+1]]])
    return firstEyeRange, secondEyeRange, certainty

def mask_data(d, ranges):
    logicals = [d[j,0]>=ranges[0,0] and 
                d[j,0]<=ranges[0,1] and 
                d[j,1]>=ranges[1,0] and 
                d[j,1]<=ranges[1,1] and
                d[j,2]>=ranges[2,0] and 
                d[j,2]<=ranges[2,1]
                for j in range(d.shape[0])]
    e = np.array([np.multiply(d[:,j], logicals) for j in range(d.shape[1])])
    socket = np.array(e[:,e[1]!=0]) # columns are x,y,z,r
    return socket

def coords(histData2, firstEyeRange, secondEyeRange):
    import scipy
    from scipy.optimize import curve_fit
    socket_1 = mask_data(histData2, firstEyeRange) # can be any of the hist_datas because only using 
                                                 # the first 3 cols
    socket_2 = mask_data(histData2, secondEyeRange)
    socket_1 = core.rejectOutliers(socket_1)
    socket_2 = core.rejectOutliers(socket_2)
   
    def max_z_r(socket):
        p = np.polyfit(socket[2], socket[3], deg=2)
        def f(z): return p[0]*z**2 + p[1]*z + p[2]
        max_z = scipy.optimize.fmin(lambda r: -f(r), 0)
        p = np.poly1d(p)
        max_r = p(max_z)
        return max_z, max_r
    
    maxz1, maxr1 = max_z_r(socket_1)
    maxz2, maxr2 = max_z_r(socket_2)
    if maxz1 > np.amax(socket_1[2]) or maxz1 < np.amin(socket_1[2]):
        maxz1 = np.array([np.mean(socket_1[2])])
    if maxz2 > np.amax(socket_2[2]) or maxz2 < np.amin(socket_2[2]):
        maxz2 = np.array([np.mean(socket_2[2])])
    
    p = np.polyfit(socket_1[2], socket_1[0], deg=1)
    def f(z): return p[0]*z + p[1]
    x1=f(maxz1)
    p = np.polyfit(socket_1[2], socket_1[1], deg=1)
    def f(z): return p[0]*z + p[1]
    y1=f(maxz1)
    
    p = np.polyfit(socket_2[2], socket_2[0], deg=1)
    def f(z): return p[0]*z + p[1]
    x2=f(maxz2)
    p = np.polyfit(socket_2[2], socket_2[1], deg=1)
    def f(z): return p[0]*z + p[1]
    y2=f(maxz2)

#    ind1 = np.argmax(socket_1[3])
#    ind2 = np.argmax(socket_2[3])
#    coord1 = socket_1[0][ind1]/2, socket_1[1][ind1]/2, socket_1[2][ind1]/2 # get rid of the factor of 2
#    coord2 = socket_2[0][ind2]/2, socket_2[1][ind2]/2, socket_2[2][ind2]/2
#    c1 = np.array([coord1[0], coord1[1], coord1[2]]) 
#    c2 = np.array([coord2[0],coord2[1],coord1[2]])
    
    c1 = np.array([x1/2,y1/2,maxz1/2])[:,0]
    c2 = np.array([x2/2,y2/2,maxz2/2])[:,0]
    
    return c1,c2

def checkcoords(c1, c2, softtissue):

    """Plots figures checking that the coordinates were correctly chosen for the 2 eye sockets"""
    c1 = c1.astype(int)
    c2 = c2.astype(int)

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3)

    a,b,c = softtissue.shape

    ax1.imshow(softtissue[:,:,c1[2]])
    ax1.plot(c1[1],c1[0], marker='o', markersize=2,markeredgewidth=25, markeredgecolor='r')
    ax1.set_xlim([0,b])
    ax1.set_ylim([0,a])
    ax1.set_aspect('equal')

    ax2.imshow(softtissue[:,c1[1],:])
    ax2.plot(c1[2],c1[0], marker='o', markersize=2,markeredgewidth=25, markeredgecolor='r')
    ax2.set_xlim([0,c])
    ax2.set_ylim([0,a])
    ax2.set_aspect('equal')

    ax3.imshow(softtissue[c1[0],:,:])
    ax3.plot(c1[2],c1[1], marker='o', markersize=2,markeredgewidth=25, markeredgecolor='r')
    ax3.set_xlim([0,c])
    ax3.set_ylim([0,b])
    ax3.set_aspect('equal')
    
    ax4.imshow(softtissue[:,:,c2[2]])
    ax4.plot(c2[1],c2[0], marker='o', markersize=2,markeredgewidth=25, markeredgecolor='r')
    ax4.set_xlim([0,b])
    ax4.set_ylim([0,a])
    ax4.set_aspect('equal')

    ax5.imshow(softtissue[:,c2[1],:])
    ax5.plot(c2[2],c2[0], marker='o', markersize=2,markeredgewidth=25, markeredgecolor='r')
    ax5.set_xlim([0,c])
    ax5.set_ylim([0,a])
    ax5.set_aspect('equal')

    ax6.imshow(softtissue[c2[0],:,:])
    ax6.plot(c2[2],c2[1], marker='o', markersize=2,markeredgewidth=25, markeredgecolor='r')
    ax6.set_xlim([0,c])
    ax6.set_ylim([0,b])
    ax6.set_aspect('equal')
    plt.show()
    
    return None

def anglesFromEyes(c1,c2, arrayShape):
    # point that the plane goes through, p
    c = 0.5*(c1+c2)
#    c1 = np.array([c1[1], c1[0], c1[2]])
#    c2 = np.array([c2[1], c2[0], c2[2]])
#    dist1 = np.linalg.norm(np.array(arrayShape)-cc1)
#    dist2 = np.linalg.norm(np.array(arrayShape)-cc2)
#    if dist1<dist2:
#        c1 = np.roll(c1, 1)
#        c2 = np.roll(c2, 1)
#    else:
#        c1 = np.roll(c1, -1)
#        c2 = np.roll(c2, -1)    
    normal = (c1-c2)
    normal = normal/np.linalg.norm(normal)
    
    zaxis = np.array([0,0,1])
    cosangle = np.dot(normal, zaxis)
    angle = np.arccos(cosangle)
    angle1 = angle*360/np.pi/2.

    xaxis = np.array([0,1,0])
    cosangle = np.dot(normal, xaxis)
    angle = np.arccos(cosangle)
    angle2 = angle*360/np.pi/2.
    
    return angle1, angle2 #angles are in degrees

def correctSkews(angle1, angle2, array):
    from scipy.ndimage.interpolation import rotate
    rotated1 = rotate(array, angle1, mode='nearest', axes=(0,1))
    angle1rad = angle1/360*2*np.pi
    rotated2 = rotate(rotated1, angle2 ,mode='nearest', axes=(2,0))
    return rotated1, rotated2