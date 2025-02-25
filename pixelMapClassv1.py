import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import pydicom as pydicom
import skimage as skimage
from skimage.draw import disk
from skimage.measure import label, regionprops
from skimage.morphology import binary_erosion, disk
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import datetime
import cv2 as cv

#path = "/Users/merlinowens/Documents/STP/MRI/QA/Measuring_T1/14082024/DICOM/00008C3A/AA36A35A/AAFA8C00"
path = "H:/STP/MRI/Measuring_T1/14082024/DICOM/00008C3A/AA36A35A/AAFA8C00"

def load_dicom_images(folder_path):
    
    dicom_files = glob.glob(os.path.join(folder_path, '**', '*'), recursive=True)
    print(f"Scanning folder: {folder_path}")
    print(f"Found {len(dicom_files)} files")
    print(f"Number of files found in nested directories: {len(dicom_files)}")
    if not dicom_files:
        print('No DICOM files found.')
    dicom_images = []
    for file in dicom_files:
        try:
            dicom_image = pydicom.dcmread(file, force = True)
            series_description = dicom_image.get('SeriesDescription', '')
            protocol_name = dicom_image.get('ProtocolName', '')
            prefix = "t1_se_tra"
            if series_description.startswith(prefix) or protocol_name.startswith(prefix):
                dicom_images.append(dicom_image)
        except Exception:
            pass
    return dicom_images

# Sorts the images by their inversion time
def inversion_(dicom_image):
    return dicom_image.get((0x0018, 0x0082), "Not Available").value

def sort_by_inversion(dicom_images):
    inversions = []
    images = sorted(dicom_images, key=lambda img: inversion_(img) or float('inf'))
    for image in images:
        inversions.append(image.get((0x0018, 0x0082), "Not Available").value)
    #print(f'inversion times: {inversions}')
    return images, inversions

images_ = load_dicom_images(path)

images, inversions = sort_by_inversion(images_)
inversions = np.array(inversions)

def applyMask(image, threshold):
    mask = image > threshold
    return mask

print(f'shapeImages: {np.shape(images)}')

# A binary search algorithm for finding a local minimum in discrete function.

def localMinUtil(arr, low, high, n):
    if low == high:
        return low
    
    if low + 1 == high:
        return low if arr[low] <=arr[high] else high
    
    # Find index of middle element
    mid = low + (high - low) // 2  # (low + high) // 2

    #debugging
    #print(f'arrShape: {np.shape(arr)} arrPrint: {arr}')
    #
    # Compare middle element with its neighbours (if neighbours exist)
    if ((mid == 0 or arr[mid-1] > arr[mid]) and
            (mid == n-1 or arr[mid+1] > arr[mid])):
        return mid

    # If middle element is not minima and its left neighbour is smaller than it, then left half must have a local minima.
    if mid > 0 and arr[mid-1] <= arr[mid]:
        return localMinUtil(arr, low, mid, n)

    # If middle element is not minima and its right neighbour is smaller than it, then right half must have a local minima.
    return localMinUtil(arr, (mid + 1), high, n)


# A wrapper over recursive function localMinUtil()
def localMin(arr, n):
    return localMinUtil(arr, 0, n-1, n)

# A function to invert sign of every data point to the left of the minimum, as the images 
# were magnitude reconstructed.

def magnitude_corr(array):
    min_idx = localMin(array, len(array))
    for i in range(min_idx):
        array[i] = -array[i]
    return array, min_idx

# Defining the exponential T1 recovery solution to the Bloch equation.

def model_function(TI, T1, I0):
    return I0 * (1 - 2 * np.exp(-TI / T1))

def magnitude_minima_corr(array, idx_min):
    array[idx_min] = -array[idx_min]
    return array

def residuals(params, TI, observed_data):
    T1, I0 = params
    return observed_data - model_function(TI, T1, I0)

def t1_fit_least_squares(array, inversions, idx_min):
    initial_guess = [1000, np.max(array)]  # Initial guess for T1 and I0
    
    # Perform least squares optimization
    result = least_squares(residuals, initial_guess, args=(inversions, array))
    
    # Extract the optimized parameters
    estimated_T1, estimated_I0 = result.x
    
    # Calculate fitted intensities using the optimized parameters
    fitted_intensities = model_function(inversions, estimated_T1, estimated_I0)
    
    # Calculate R^2
    residuals_array = array - fitted_intensities
    ss_res = np.sum(residuals_array ** 2)
    ss_tot = np.sum((array - np.mean(array)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # This if statement detects whether the point at the minimum needs to be magnitude 
    # corrected by checking if the fit has a R^2 of over 0.999. Without the correction,
    # R^2 will not be as high, as the least squares fitting will attempt to go through
    # this erroneous point. To illustrate this, change condition below to only execute 
    # if r_squared  >100.

    if r_squared <0.999:
        array = magnitude_minima_corr(array, idx_min)
        initial_guess = [1000, np.max(array)]  # Initial guess for T1 and I0
    
        # Perform least squares optimization
        result = least_squares(residuals, initial_guess, args=(inversions, array))
        
        # Extract the optimized parameters
        estimated_T1, estimated_I0 = result.x
        
        # Calculate fitted intensities using the optimized parameters
        fitted_intensities = model_function(inversions, estimated_T1, estimated_I0)
        
        # Calculate R^2
        residuals_array = array - fitted_intensities
        ss_res = np.sum(residuals_array ** 2)
        ss_tot = np.sum((array - np.mean(array)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
    
    return fitted_intensities, estimated_T1, r_squared, estimated_I0

###

class pixelMapping:
    def __init__(self, images, dependentVariable, patchsize, threshold):
        self.images = images
        self.patchsize = patchsize
        self.dependentVariable = dependentVariable
        self.threshold = threshold
        self.results = None
    
    def thresholding(self):
        for i in range(len(self.images)):
            if isinstance(self.images[i], pydicom.dataset.FileDataset):
                if "PixelData" in self.images[i]:
                    self.images[i] = self.images[i].pixel_array
                else:
                    raise ValueError(f'DICOM at self.images[{i}] does not contain PixelData')
            print(f"Type of self.images[{i}]: {type(self.images[i])}")


            image_array = np.array(self.images[i])
            self.images[i] = np.where(image_array > self.threshold, image_array, 0)
        
    def averagingGrid(self, image):
        patchsize = self.patchsize
        shape = np.shape(image)
        #print(f'currentImageShape: {shape}')
        grid = np.zeros((shape[0], shape[1]))
        for j in range(0,np.shape(image)[0], patchsize):
            #print(f'currentIndexJ: {j}')
            for i in range(0,np.shape(image)[0], patchsize):
                #print(f'currentIndexI: {i}')
                sum_ = 0
                avg = 0
                for step in range(patchsize): #gridsize -1?
                    #print(f'currentStep: {step}')
                    sum_ += sum(image[j + step][i:i + patchsize])
                    if step == patchsize-1:
                        avg = sum_/patchsize**2
                        #print(f'patchAverage: {avg}')
                for step in range(patchsize):
                    grid[j+step][i:i + patchsize] = avg
        print(f'gridShape: {np.shape(grid)}')
        return grid
    
    # Going to compute fits for one pixel per patch then apply that value to the rest of the pixels to minimise
    # the number of computations needed.

    def patchBasedFit(self, grids, dependentVariable):
        patchsize = self.patchsize
        print(f'gridsGridShape: {np.shape(grids[0])}, gridsGridShapeIdx0: {np.shape(grids[0])}')
        shape = np.shape(grids[0])
        mapping = np.zeros((shape[0],shape[1])) #np.shape(0) ?? was this a mistake
        for j in range(0, shape[0], patchsize):
            for i in range(0, shape[0], patchsize):
                arr = []
                arrToFit = []
                idx_min = []
                #print(f'lenGrids: {len(grids)}')
                for idx in range(len(grids)):
                    #print(f'currentGridPosition (i, j): {i},{j}')
                    arr.append(grids[idx][j][i])
                if 60<sum(arr):
                    print(f'arrPrint: {arr}')
                    arrToFit, idx_min = magnitude_corr(arr)
                    print(f'arrToFit: {arrToFit}')
                    print(f'arrToFitShape: {np.shape(arrToFit)}')
                    print(f'dependentVariable: {np.shape(dependentVariable)}')
                    fitted_intensities, estimated_T1, r_squared, estimated_I0 = t1_fit_least_squares(arrToFit, dependentVariable, idx_min)
                    for step in range(patchsize):
                        mapping[j + step][i:i + patchsize] = estimated_T1
                else:
                    for step in range(patchsize):
                        mapping[j + step][i:i + patchsize] = 0
        return mapping
    
    def averageExtraction(self, thresholds, mapping, erosionStrength):
        patchsize = self.patchsize
        mask = (mapping > thresholds[0]) & (mapping < thresholds[1])
        # Need to erode the mask to remove high T1 estimations at vial boundaries
        selem = disk(erosionStrength)
        mask = binary_erosion(mask, selem)
        labelling = label(mask)
        Regions = regionprops(labelling)
        
        ValidRegions = [region for region in Regions if (region.area<500/patchsize)]
        
        regionProps = regionprops(labelling, intensity_image=mapping)

        # MAYBE REPLACE NUMBER THRESHOLDS WITH VARIABLES EG 2200 REPLACED WITH 'upperBound'

        regionData = [(region.intensity_mean, region.centroid) for region in regionProps if (region.area<500/patchsize) 
                          and (region.area>5)]
        regionAverages, regionCentroids = zip(*regionData) if regionData else ([],[])
        return regionCentroids, regionAverages, mask

    def t1MapDisplay(self, mapping, vials, plot_mask, erosionStrength):
        patchsize = self.patchsize
        #norm = mcolors.LogNorm(vmin=mapping.min() + 1e-6, vmax=mapping.max())
        min = 100
        max = 3000
        mapping = np.where((mapping >= min) & (mapping <= max), mapping, 0)
        cmap = 'inferno'
        fig, ax = plt.subplots()
        if vials == True:
            regionCentroids, regionAverages, mask = self.averageExtraction(thresholds, mapping, erosionStrength)
            for i, centroid in enumerate(regionCentroids):
                ax.scatter(centroid[1], centroid[0], s=2, marker='x')
                ax.text(centroid[1], centroid[0], f'Vial {i+1}', fontsize=5, color = 'white')
        img = ax.imshow(mapping, cmap = cmap)
        cbar = plt.colorbar(img, ax=ax)
        cbar.set_label(f'T1 [ms]')
        plt.show()
        if plot_mask:
            plt.imshow(mask)
            plt.title(f'ErodedMask')
            plt.show()

        if vials:
            return regionCentroids, regionAverages
    


myPixelMapping = pixelMapping(images, inversions, patchsize = 2, threshold = 50)

grids = []
myPixelMapping.thresholding()
for i in range(len(images)):
    grids.append(myPixelMapping.averagingGrid(images[i]))
mapping = myPixelMapping.patchBasedFit(grids, dependentVariable=inversions)
thresholds = [100,2200]
#Centroids, regionAverages = myPixelMapping.averageExtraction(thresholds, mapping, patchsize = 1)
regionCentroids, regionAverages = myPixelMapping.t1MapDisplay(mapping, True, True, 2)
#myPixelMapping.t1MapDisplay(mapping, False, patchsize)
for i in range(len(regionAverages)):
    print(f'Vial {i+1}: {regionAverages[i]}')
