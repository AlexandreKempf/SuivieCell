import cv2
import h5py
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from PIL import Image
from skimage import io, transform
from skimage.feature import register_translation
from scipy.ndimage import fourier_shift
from scipy.io import loadmat

def exist(dir, file, multiple_answer=False, iter=10):
    """
    Is there FILE in DIR or in its ITER subfolder
    """
    answer = ""
    for i in np.arange(iter):
        if len(glob.glob(dir + i * "*/" + file)) > 0:
            if multiple_answer==False:
                answer = glob.glob(dir + i * "*/" + file)[0]
            else:
                answer = np.sort(glob.glob(dir + i * "*/" + file))
    return answer


def shift(img, shiftlist):
    offset_image = fourier_shift(np.fft.fftn(img), shiftlist)
    offset_image = np.fft.ifftn(offset_image).real
    return offset_image


def align_images(im1, im2, show=True):
    # Convert images to grayscale
    im1_g = np.copy(im1)
    im2_g = np.copy(im2)

    im1_g = cv2.cvtColor(im1_g, cv2.COLOR_BGR2GRAY)
    im2_g = cv2.cvtColor(im2_g, cv2.COLOR_BGR2GRAY)

    im1_g = gaussian_filter(im1_g, 2)
    im2_g = gaussian_filter(im2_g, 2)
    sz = im1_g.shape

    warp_mode = cv2.MOTION_EUCLIDEAN
    # warp_mode = cv2.MOTION_AFFINE
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    number_of_iterations = 5000
    termination_eps = 1e-10
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                number_of_iterations,  termination_eps)

    (cc, warp_matrix) = cv2.findTransformECC(
        im1_g, im2_g, warp_matrix, warp_mode, criteria)
    im2_aligned = cv2.warpAffine(
        im2, warp_matrix, (sz[1], sz[0]), flags=cv2.WARP_INVERSE_MAP)
    if show == True:
        ax1 = plt.subplot(131)
        ax1.imshow(im1)
        ax2 = plt.subplot(132, sharex=ax1, sharey=ax1)
        ax2.imshow(im2)
        ax3 = plt.subplot(133, sharex=ax1, sharey=ax1)
        ax3.imshow(im2_aligned)
        plt.show()

    return im2_aligned, warp_matrix


def loadROI(filepath):
    try:
        data = loadmat(filepath, mat_dtype=True)
    except:
        data = h5py.File(filepath, 'r')
    ny = int(data["ny"][0])
    nx = int(data["nx"][0])
    try:
        matr = data["A"].toarray()
    except:
        matr = sparse.csc_matrix(
            (data["A"]["data"], data["A"]["ir"], data["A"]["jc"])).toarray()
    size = matr.shape[1]
    loc = [np.where(matr[i, :] > 0)[0] for i in np.arange(matr.shape[0])]

    vide = np.zeros(nx * ny)
    for i in np.arange(len(loc)):
        vide[loc[i]] = i + 1

    vide = np.reshape(vide, (ny, nx))
    return vide, loc


def ROIcenter_gtransform(loc,mat,nx):
    # loc is the localization of the cell as a list of array in the original geometry space
    # mat is the (2x3) transformation matrix_align
    # nx is the number of column of the image (second dimension)
    loc_after = []
    matr_transform = cv2.invertAffineTransform(mat)
    matr_transform = np.concatenate([matr_transform, [[0, 0, 1]]], 0)
    for i in np.arange(len(loc)):
        y = loc[i] // (nx)
        x = loc[i] % (nx)
        z = np.ones(len(loc[i]))
        a = np.dstack([y, x, z])
        a = np.dot(a, matr_transform)[0, :, :2]
        loc_after.append(a)

    loc_centre = np.array([np.median(i, 0) for i in loc_after])
    loc_centre[:, 1] += matr_transform[0,2]
    loc_centre[:, 0] += matr_transform[1,2]
    return loc_centre


def give_coord(center,dist,angle):
    x = dist * np.cos(angle) + center[0];
    y = dist * np.sin(angle) + center[1];
    return np.array([x,y])

def mindist(center,loc_center):
    diff = (loc_center-center)
    d = np.sqrt(np.sum(diff**2,1))
    # calcul l'angle pondéré des points proches (<5px)
    angl = np.arctan2(diff[:,1], diff[:,0])
    minCenter = np.argmin(d)
    minDistance = d[minCenter]
    minAngle = angl[minCenter]
    return minCenter,minDistance,minAngle

def align_cells(listregfile,mat,n1,n2):
    vide1, loc1 = loadROI(listregfile[n1])
    vide2, loc2 = loadROI(listregfile[n2])
    loc_center1 = ROIcenter_gtransform(loc1,mat[n1],vide1.shape[1])
    # vide_aligned1 = cv2.warpAffine(vide1, mat[ii[0]], (res[ii[0]].shape[0],res[ii[0]].shape[1]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
    loc_center2 = ROIcenter_gtransform(loc2,mat[n2],vide2.shape[1])
    # vide_aligned2 = cv2.warpAffine(vide2, mat[ii[1]], (res[ii[1]].shape[0],res[ii[1]].shape[1]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
    CEN=np.zeros(len(loc_center1))-1
    for i, center in enumerate(loc_center1):
        #select cells from the same day around to estimate the position of the putative cell the next day
        diff = (loc_center1-center)
        d = np.sqrt(np.sum(diff**2,1))
        idx = np.where(d<100)[0]
        neiMat = np.array([mindist(loc_center1[ii], loc_center2) for ii in idx])
        neiMat = neiMat[neiMat[:,1]<3.5]
        _, meanDist, meanAngle = np.median(neiMat,0)
        estimate = give_coord(center,meanDist,meanAngle)
        # calcul la distance entre un points du jour 1 et les autres du jour 2
        diff_inter = (loc_center2-estimate)
        d_inter = np.sqrt(np.sum(diff_inter**2,1))
        if np.min(d_inter)<3.5:
            CEN[i] = np.argmin(d_inter)
    return CEN




# PARAM
PATH = "/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=workdata/ANALYSIS/Alex/EXPSEB/BEHAV/"
# PATH = "/home/alexandre/data/2p/expseb/"
MOUSE = "M1"
# TAKE ALL DIRECTORY ALREADY ALIGN
listfile = np.array(glob.glob(PATH + "*" + MOUSE + "*"))
booldir = np.array([os.path.isdir(ii) for ii in listfile])
listfile = np.sort(listfile[booldir])
reffile = np.array([exist(ii, "*stdimg.png", 3) for ii in listfile])
ref2file = np.array([exist(ii, "*meanimg.png", 3) for ii in listfile])
reffile = reffile[np.where(reffile != '')[0]]
ref2file = ref2file[np.where(ref2file != '')[0]]
refday = [ii.split('/')[-2] for ii in reffile]

regfile = np.array([exist(ii, "*regions*.mat", 2) for ii in listfile])
regfile = regfile[np.where(regfile != '')[0]]
regday = [ii.split('/')[-2] for ii in regfile]

refday_bool = np.array([ii in regday for ii in refday])
reffile = reffile[refday_bool]
ref2file = ref2file[refday_bool]
#

# CREATE THE MEAN AND THE STD IMAGE IN EACH FOLDER

# # REFFILE IS THE FIRST OF THE REG MATRICES FILE HERE
# # ACTIVATE ONLY ONE TIME AT THE VERY BEGINNING
# for ii in np.arange(len(reffile)):
#     m = []
#     nloop = len(glob.glob('/'.join(reffile[ii].split('/')[:-1])+"/*"))
#     nloop = np.min([100,nloop])
#     for i in np.arange(1,nloop):
#         name = reffile[ii]
#         if i <10:
#             name = name[:-9] + str(i) + name[-8:]
#         elif i<100:
#             name = name[:-10] + str(i) + name[-8:]
#         elif i>=100:
#             name = name[:-11] + str(i) + name[-8:]
#         a=h5py.File(name, 'r')
#         a=np.array(a["data"])
#         m.append(a.mean(0))
#         print(i/nloop)
#     s = np.array(m).std(0)
#     s /= s.max()
#     s *=1000
#     m2 = np.array(m).mean(0)
#     m2 /= m2.max()
#     m2 *=500
#     cv2.imwrite('/'.join(reffile[ii].split('/')[:-2])+"/stdimg.png",s)
#     cv2.imwrite('/'.join(reffile[ii].split('/')[:-2])+"/meanimg.png",m2)


# # # REFFILE IS THE MEAN IMG HERE
# # # REGISTRATION OF THE IMAGES
# im1 = cv2.imread(reffile[0])
# im2 = cv2.imread(ref2file[0])
# im = cv2.add(cv2.divide(im1, 2), cv2.divide(im2, 2))
# res = []
# res.append(im)
# mat = []
# mat.append(np.eye(2, 3))
# for i in np.arange(1, len(reffile[:])):
#     im1 = cv2.imread(reffile[i])
#     im2 = cv2.imread(ref2file[i])
#     im = cv2.add(cv2.divide(im1, 2), cv2.divide(im2, 2))  # align on the sum of the mean and the std
#     im_aligned, matrix_align = align_images(res[0], im, show=False)
#     res.append(im_aligned)
#     mat.append(matrix_align)

# np.save("mat.npy",(mat,res))
mat,res = np.load("mat.npy")




ax1 = plt.subplot(2, np.ceil(len(res) / 2), 1)
img = cv2.cvtColor(res[0], cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.equalizeHist(img))

for i in np.arange(0, len(res)):
    plt.subplot(2, np.ceil(len(res) / 2), i + 1, sharex=ax1, sharey=ax1)
    img = cv2.cvtColor(res[i], cv2.COLOR_BGR2GRAY)
    plt.imshow(cv2.equalizeHist(img))
plt.show()




# LOAD REGIONS FOR A FILE AND MATCH THEM
# & FOUND THE APPROPRIATE MASK WITH THE GEOMETRIC TRANSFORM
listregions = [exist(path, "*regions.mat") for path in listfile]
listregions = [i for i in listregions  if len(i) !=0]
nbcell = []
for i in listregions:
    vide,loc = loadROI(i)
    nbcell.append(len(loc))

# # FOR COMPARISON BETWEEN ONE EXP AND THE REST
# iexp=0
# matiti = np.zeros((nbcell[iexp],len(listregions)))
# for i in np.arange(len(listregions)):
#     matiti[:,i] = align_cells(listregions,mat,iexp,i)
# plt.imshow(np.transpose(matiti>=0),interpolation="None", aspect='auto');plt.show()

# # # FOR ALL THE COMPARISON
# RESUME=[]
# for i in np.arange(len(listregions)):
#     smallresume = []
#     for j in np.arange(len(listregions)):
#         smallresume.append(align_cells(listregions, mat, i, j).astype(int))
#     RESUME.append(smallresume)
# pickle.dump(RESUME,open("RESUME2.p","wb"))

# RELOADING DATA
import pickle
import copy
mat = np.load("mat.npy")
RESUME = pickle.load(open("RESUME2.p","rb"))

def merge(RESUME, form, to, by):
    """
    Trouve les cellules identiques entre deux expériences très espacées temporellement grâce à une experience intermediaire.
    RESUME est la matrice qui associe à chaque pair d'expérience un vector qui indique les cellules identiques (-1 if cell not found)
    form, to and by sont les index des expériences:
    Par exemple form=0, to=2, by=1 calcul les cellules identiques entres les expériences 0 et 2 en passant par l'experience 1 pour comparer.
    """
    # 0 means from, 1 means by and 2 means to
    c01 = RESUME[form][by]
    c12 = RESUME[by][to]
    c02 = RESUME[form][to]
    for i in np.where(c02 == -1)[0]:
        if c01[i] != -1 :
            c02[i] = c12[c01[i]]
    RESUME[form][to] = c02
    return RESUME

for i in np.arange(2,len(RESUME)):
    for ii in np.arange(1,i):
        RESUME = merge(RESUME,0,i,ii)


#take only the first 6 because after that there is a movement and the cells changed
resume = np.concatenate([[RESUME[0][0]], [RESUME[0][1]], [RESUME[0][2]], [RESUME[0][3]], [RESUME[0][4]], [RESUME[0][5]]],0)
plt.imshow(resume,interpolation="None", aspect='auto');plt.show()
np.sum(np.sum(resume>0,0)>=6)
np.save("RESUMEPOST.npy",resume)

##### Load the data from this experiments
import os,sys
sys.path.append('/home/alexandre/docs/code/pkg/imgca/new/')
from imgca import imgdata, elphy_read
paths = [os.path.dirname(i) for i in listregions[:6]]

a1 = imgdata()
a1,_ =a1.create(paths[0])
a1.deconvolve()
a1.smooth('time',0.05)

a2 = imgdata()
a2,_ =a2.create(paths[1])
a2.deconvolve()
a2.smooth('time',0.05)

a3 = imgdata()
a3,_ =a3.create(paths[2])
a3.deconvolve()
a3.smooth('time',0.05)

a4 = imgdata()
a4,_ =a4.create(paths[3])
a4.deconvolve()
a4.smooth('time',0.05)

a5 = imgdata()
a5,_ =a5.create(paths[4])
a5.deconvolve()
a5.smooth('time',0.05)

a6 = imgdata()
a6,_ =a6.create(paths[5])
a6.deconvolve()
a6.smooth('time',0.05)

plt.plot(np.nanmean(a1.data,3).mean(1)[:,0], color="#000000", lw= 2);
plt.plot(np.nanmean(a2.data,3).mean(1)[:,0], color="#666666", lw= 2);
plt.plot(np.nanmean(a1.data,3).mean(1)[:,1], color="#ff0000", lw= 2);
plt.plot(np.nanmean(a2.data,3).mean(1)[:,1], color="#ff6666", lw= 2);
plt.show()

plt.plot(np.nanmean(a3.data,3).mean(1)[:,1], color="#000000", lw= 2);
plt.plot(np.nanmean(a4.data,3).mean(1)[:,1], color="#666666", lw= 2);
plt.plot(np.nanmean(a5.data,3).mean(1)[:,1], color="#bbbbbb", lw= 2);
plt.plot(np.nanmean(a3.data,3).mean(1)[:,2], color="#ff0000", lw= 2);
plt.plot(np.nanmean(a4.data,3).mean(1)[:,2], color="#ff6666", lw= 2);
plt.plot(np.nanmean(a5.data,3).mean(1)[:,2], color="#ffbbbb", lw= 2);

plt.plot(np.nanmean(a3.data,3).mean(1)[:,3], color="#00ff00", lw= 2);
plt.plot(np.nanmean(a4.data,3).mean(1)[:,3], color="#66ff66", lw= 2);
plt.plot(np.nanmean(a5.data,3).mean(1)[:,3], color="#bbffbb", lw= 2);
plt.plot(np.nanmean(a3.data,3).mean(1)[:,4], color="#0000ff", lw= 2);
plt.plot(np.nanmean(a4.data,3).mean(1)[:,4], color="#6666ff", lw= 2);
plt.plot(np.nanmean(a5.data,3).mean(1)[:,4], color="#bbbbff", lw= 2);
plt.show();

datfiles = np.sort(np.array([exist(file,"*.DAT") for file in listfile]))
recordings, dates, vectors, menupar, xpar, epinfo = elphy_read.Read(open(datfiles[5],'rb'))







#
# # TEAM MEETING ON ALIGNEMENT OF IMG AND ROI
#
# # Sum of the mean and the std image
# im1 = cv2.imread(reffile[0])
# im2 = cv2.imread(ref2file[0])
# im = cv2.add(cv2.divide(im1, 2), cv2.divide(im2, 2))
# im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# ax1 = plt.subplot(131)
# plt.imshow(im1, interpolation=None, cmap=plt.cm.Greys_r)
# plt.subplot(132,sharex=ax1,sharey=ax1)
# plt.imshow(im2, interpolation=None, cmap=plt.cm.Greys_r)
# plt.subplot(133,sharex=ax1,sharey=ax1)
# plt.imshow(im, interpolation=None, cmap=plt.cm.Greys_r)
# plt.show()
#
#
# # Alignement of the mean+std images
# ax1 = plt.subplot(2, np.ceil(len(res) / 2), 1)
# img = cv2.cvtColor(res[0], cv2.COLOR_BGR2GRAY)
# plt.imshow(cv2.equalizeHist(img),interpolation=None)
#
# for i in np.arange(0, len(res)):
#     plt.subplot(2, np.ceil(len(res) / 2), i + 1, sharex=ax1, sharey=ax1)
#     img = cv2.cvtColor(res[i], cv2.COLOR_BGR2GRAY)
#     plt.imshow(cv2.equalizeHist(img), interpolation=None)
# plt.show()
#
# # Transform the ROI accordingly and take the centers
# vide1, loc1 = loadROI(listregions[0])
# vide2, loc2 = loadROI(listregions[1])
# loc_center1 = ROIcenter_gtransform(loc1,mat[0],vide1.shape[1])
# vide_aligned1 = cv2.warpAffine(vide1, mat[0], (res[0].shape[0],res[0].shape[1]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
# loc_center2 = ROIcenter_gtransform(loc2,mat[1],vide2.shape[1])
# vide_aligned2 = cv2.warpAffine(vide2, mat[1], (res[1].shape[0],res[1].shape[1]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
#
# ax1 = plt.subplot(121)
# plt.imshow(vide_aligned1>0, origin='lower', interpolation=None, cmap=plt.cm.Greys_r)
# plt.plot(loc_center1[:,1],loc_center1[:,0], '+')
# plt.subplot(122,sharex=ax1,sharey=ax1)
# plt.imshow(vide_aligned2>0, origin='lower', interpolation=None, cmap=plt.cm.Greys_r)
# plt.plot(loc_center2[:,1],loc_center2[:,0], '+')
# plt.show()
#
#
# # Problem of fine alignment
# plt.plot(loc_center1[:,1],loc_center1[:,0], 'or')
# plt.plot(loc_center2[:,1],loc_center2[:,0], 'ob')
# plt.show()
#
# # Final Alignement
# ali2 = align_cells(listregions,mat,0,1)
# ali1 = np.arange(len(ali2))
# ali1 = ali1[ali2>=0].astype(int)
# ali2 = ali2[ali2>=0].astype(int)
# plt.plot(loc_center1[ali1,1],loc_center1[ali1,0], 'or')
# plt.plot(loc_center2[ali2,1],loc_center2[ali2,0], 'ob')
# plt.show()
#
# # ROI view
# data3 = loadmat(listregions[0], mat_dtype=True)
# ny3 = int(data3["ny"][0])
# nx3 = int(data3["nx"][0])
# matr3 = data3["A"].toarray()
# size3 = matr3.shape[1]
# loc3 = [np.where(matr3[i, :] > 0)[0] for i in np.arange(matr3.shape[0])]
# vide3 = np.zeros(nx3 * ny3)
# for i in np.arange(len(loc3))[ali1]: # WARNING HERE CHANGE ALI1 IF YOU CHANGE THE LISTREGIONS IDX
#     vide3[loc3[i]] = i + 1
# vide3 = np.reshape(vide3, (ny3, nx3))
#
# data4 = loadmat(listregions[1], mat_dtype=True)
# ny4 = int(data4["ny"][0])
# nx4 = int(data4["nx"][0])
# matr4 = data4["A"].toarray()
# size4 = matr4.shape[1]
# loc4 = [np.where(matr4[i, :] > 0)[0] for i in np.arange(matr4.shape[0])]
# vide4 = np.zeros(nx4 * ny4)
# for i in np.arange(len(loc4))[ali2]: # WARNING HERE CHANGE ALI1 IF YOU CHANGE THE LISTREGIONS IDX
#     vide4[loc4[i]] = i + 1
# vide4 = np.reshape(vide4, (ny4, nx4))
#
# ax1 = plt.subplot(121)
# plt.imshow(cv2.warpAffine(vide3, mat[0], (res[0].shape[0],res[0].shape[1]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP), origin='lower', interpolation=None)
# plt.subplot(122,sharex=ax1,sharey=ax1)
# plt.imshow(cv2.warpAffine(vide4, mat[1], (res[1].shape[0],res[1].shape[1]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP), origin='lower', interpolation=None)
# plt.show()
#






















































# simi = np.zeros((np.max(nbcell),len(nbcell)), dtype='int')
# ref = 2
# for i in np.arange(len(listregions)):
#     loc_filepath2 = listregions[i]
#     dist1_2=align_cells(listregions,mat,[ref,i])
#     simi[dist1_2[:,0],i] = dist1_2[:,1]
#
# # plt.imshow(np.transpose(simi>0), aspect='auto', interpolation="None");plt.show()
# # HOW TO COMPARE MATRICE WITH DIFF REF
# np.sum(np.sum(simi>0,1)==4)








############## NOPE NOPE NOPE
# R = []
# for i in np.arange(len(listregions)):
#     S=[]
#     for j in np.arange(len(listregions)):
#         dist=align_cells(listregions,mat,[i,j])
#         S.append(dist)
#     R.append(S)
#
# exp=0
# nbcell = np.array(nbcell)
# result = (np.zeros((nbcell[0], len(R[0])))-1).astype(int)
#
# for i in np.arange(result.shape[0]): # cell
#     selection = np.arange(len(R[0]))[np.arange(len(R[0])) != exp] # remove exp de la selection pour les expériences sinon on n'est sur d'avoir ce numéro de cellule à cause de la diagonale de R
#     for j in selection : # exp
#         if i in R[0][j][:,0]:
#             result[i,exp]=i
#             result[i,j]=R[0][j][R[0][j][:,0]==i,1]
# i=5
# j=1




# # VISUALIZE THE REGIONS SELECTED
# vide1, loc1 = loadROI(loc_filepath1)
# vide2, loc2 = loadROI(loc_filepath1)
# loc1 = [loc1[i] for i in dist1_2[:,0]]
# loc2 = [loc2[i] for i in dist1_2[:,1]]
# nx1,ny1 = vide1.shape
# nx2,ny2 = vide2.shape
# vide1 = np.zeros(nx1*ny1)
# vide2 = np.zeros(nx2*ny2)
# colo = np.random.random(len(dist1_2))*255
#
# for i in np.arange(len(dist1_2)):
#    vide1[loc1[i]]=colo[i]
#    vide2[loc2[i]]=colo[i]
#
# ax1 = plt.subplot(121)
# plt.imshow(vide1.reshape(nx1,ny1))
# plt.subplot(122, sharex=ax1, sharey = ax1)
# plt.imshow(vide2.reshape(nx2,ny2))
# plt.show()
