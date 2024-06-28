# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 13:19:05 2024

@author: 20202215
"""

import numpy as np
from scipy import ndimage
from scipy.io import loadmat
import matplotlib.pyplot as plt
from model import Model
from PIL import Image, ImageOps
from skimage.morphology import binary_closing
from skimage.morphology import disk
#import cv2
import math

def load_data(path_to_data):
    '''Load the data from the .mat file and return a dictionary of the fields in the .mat file. The .mat 
    file contains a 1x1 struct called VObj (stands for Virtual Object). VObj contains the 16 fields described 
    in https://mrilab.sourceforge.net/manual/MRiLab_User_Guide_v1_3/MRiLab_User_Guidech3.html#x8-120003.1
    
    Args:
    path_to_data (str): The path to the .mat file containing the data
    
    Returns:
    data_dict (dict): A dictionary containing the 16 fields in the VObj struct'''

    # Load the .mat file. 
    mat_contents = loadmat(path_to_data)

    # Access the VObj struct
    VObj = mat_contents['VObj']

    # Create a dictionary of the fields
    data_dict = {}
    for field_name in VObj.dtype.names:
        data_dict[field_name] = VObj[0, 0][field_name]

    return data_dict


def scan(scan_parameters, model):
    '''Simulate a scan of the model using the scan parameters. The scan parameters include the TR (repetition 
    time), TE (echo time), flip_angle. The function should return the simulated MRI image.
    
    Args:
    model (Model): The model to scan
    scan_parameters (dict): A dictionary containing the scan parameters
    
    Returns:
    simulated_image (np.ndarray): The simulated MRI image'''

    # Simulate the MRI image
    signal_array = calculate_signal(scan_parameters, model)

    # Add image artifacts
    Artefact = scan_parameters.get('Artefact')
    
    if Artefact == 'Wrap-around':
        simulated_image = add_aliasing(signal_array, scan_parameters)       # Wrap around artefact
    if Artefact == 'Chem_shift_1':
        simulated_image = fat_shift_1(signal_array, scan_parameters)        # Chemical shift of the first kind
    if Artefact == 'Chem_shift_2':
        simulated_image = fat_shift_2(signal_array, **scan_parameters)      # Chemical shift of the second kind
    if Artefact == 'Chem_shift_1_and_2':
        simulated_image = fat_shift_1_and_2(signal_array, scan_parameters)  # Chemical shift of the first AND second kind combined
    if Artefact == None:
        simulated_image = signal_array                                      # No artefacts
    
    return simulated_image


def calculate_signal(scan_parameters, model):
    '''Calculate the signal intensity given the scan parameters for each voxel of the model. Note that the 
    signal intensity is calculated using the signal equation for a spin echo sequence. 
    
    Args: 
    scan_parameters (dict): A dictionary containing the scan parameters
    model (Model): The model to scan'''

    TE = scan_parameters['TE']
    TR = scan_parameters['TR']
    TI = scan_parameters['TI']

    PD = model.PDmap[:, :] 
    T1 = model.T1map[:, :]
    T2 = model.T2map[:, :]

    signal_array = np.abs(PD * np.exp(np.divide(-TE,T2)) * (1 - 2 * np.exp(np.divide(-TI, T1)) + np.exp(np.divide(-TR, T1)))) # calculate the signal intensity using the signal equation for a spin echo sequence.  
    signal_array = np.nan_to_num(signal_array, nan=0) # replace all nan values with 0. This is necessary because the signal_array can contain nan values, for example if both TI and T1 are 0. 

    return signal_array


def axes_image(model, **scan_parameters):
    '''Shows image in the chosen axial direction (plane) 
    
    Args:
    Scan_parameters (dict): contains the chosen settings and should contain at least the following parameters
        plane (str): either Transverse, Sagittal or Coronal, based on what plane should be visible in the simulator    
        slice_nr (int): number of the slice that should be shown in the simulator
        
    Returns:
    Image based on the chosen plane and slice number'''
   
    # Unpack chosen scan parameters
    plane = scan_parameters.get('plane')
    slice_nr = scan_parameters.get('slice_nr')
    waterfat_phantom = scan_parameters.get('WaterFat phantom')
    
    # In case they forget to change their slice number between the brain phantom and this phantom. This can be deleted if an input is used that cannot go above the max layers of the phantom.
    if waterfat_phantom == True:
        plane = "Transverse"
        if slice_nr >= 32:
            slice_nr = 16
    
    title = plane + ' plane'
    
    if plane == 'Transverse':
        plt.figure()
        plt.imshow(model[:, :, slice_nr], cmap='gray', vmin=0, vmax=1)
        plt.title(title)
        plt.show()
        
    elif plane == 'Sagittal':
        plt.figure()
        plt.imshow(np.rot90(model[:, slice_nr, :], k=3), cmap='gray', vmin=0, vmax=1)
        plt.title(title)
        plt.show()
        
    elif plane == 'Coronal':
        plt.figure()
        plt.imshow(np.rot90(model[slice_nr, :, :], k=3), cmap='gray', vmin=0, vmax=1)
        plt.title(title)
        plt.show()
        
    else:
        print("Error: Invalid plane specified. Must be 'Transverse', 'Sagittal', or 'Coronal'.")


def masking(image):
    '''Makes a mask of the image and makes sure it does not contain any unnecessary holes. 
    
    Args:
    image (np.ndarray): array of an image that needs a mask
        
    Returns:
    mask_image (np.ndarray): array containing a mask of the chosen image'''
    
    # Used for closing of the mask
    stamp = ([[1,1,1,1,1],
              [1,1,1,1,1],
              [1,1,1,1,1]],
             [[1,1,1,1,1],
              [1,1,1,1,1],
              [1,1,1,1,1]],
             [[1,1,1,1,1],
              [1,1,1,1,1],
              [1,1,1,1,1]])
    
    # Make mask of image
    mask_image = np.where(image != 0, 1, image)
    
    # Apply closing using aforementioned stamp
    mask_image = ndimage.binary_closing(mask_image, structure = stamp, border_value=1)
    
    return mask_image


def FOV_padding(part_array, direction, FOV_AP, FOV_LR, FOV_FH, FOV_center_AP, FOV_center_LR, FOV_center_FH):
    '''Shift the cut-off sides of the field of view by adding padding to the outside, making them the same size as the field of view. 
    
    Args:
    part_array (np.ndarray): array of the cut-off side that is going to be padded
    direction (str): in which direction the padding should be added
    FOV_AP (int): length of the field of view in anterior-posterior direction
    FOV_LR (int): length of the field of view in left-right direction
    FOV_FH (int): length of the field of view in feet-head direction
    FOV_center_AP (int): positioning of the center of the field of view in anterior-posterior direction
    FOV_center_LR (int): positioning of the center of the field of view in left-right direction
    FOV_center_FH (int): positioning of the center of the field of view in feet-head direction
      
    Returns:
    FOV_pad (np.ndarray): array of the cut-off side with padding'''
    
    # Adds the padding in the correct directions
    def FOV_pad(a, b, c, d, e, f, g, h, i, j):
        dimension_FOV = part_array.shape[a]
        needed_dimension = FOV_LR*d + FOV_AP*b + FOV_FH*c - dimension_FOV

        # Ensure that the padding width is correctly formatted
        padding_width = [(needed_dimension*e, needed_dimension*f), (needed_dimension*g, needed_dimension*h), (needed_dimension*i, needed_dimension*j)]

        FOV_padded = np.pad(part_array.copy(), padding_width, mode='constant')
        return FOV_padded
    
    # Assigns parameter values based on the direction the padding should go
    if direction == 'FOV_left':
        param = (1,0,0,1,0,0,1,0,0,0)
    elif direction == 'FOV_right':
        param = (1,0,0,1,0,0,0,1,0,0)
    elif direction == 'FOV_head':
        param = (0,1,0,0,1,0,0,0,0,0)
    elif direction == 'FOV_feet':
        param = (0,1,0,0,0,1,0,0,0,0)
    elif direction == 'FOV_anterior':
        param = (2,0,1,0,0,0,0,0,1,0)
    elif direction == 'FOV_posterior':
        param = (2,0,1,0,0,0,0,0,0,1)

    return FOV_pad(*param)


def add_aliasing(signal_array, **scan_parameters):
    '''Simulate a wrap around artefact based on the chosen settings of the simulator. 
    
    Args:
    Signal_array (np.ndarray): contains the signal intensity of each voxel
    Scan_parameters (dict): contains the chosen settings and should contain at least the following parameters
        plane (str): either Transverse, Sagittal or Coronal, based on what plane should be visible in the simulator    
        PE_direction (str): phase encoding direction, either horizontal or vertical, based on the chosen plane
        
        FOV_unit (str): if the FOV is given in metre or centimeter for some reason, the values will be scaled based on the needed FOV input
        FOV_AP (int): length of the field of view in anterior-posterior direction
        FOV_LR (int): length of the field of view in left-right direction
        FOV_FH (int): length of the field of view in feet-head direction
        FOV_center_AP (int): positioning of the center of the field of view in anterior-posterior direction
        FOV_center_LR (int): positioning of the center of the field of view in left-right direction
        FOV_center_FH (int): positioning of the center of the field of view in feet-head direction
        
    Returns:
    I_OUT (np.ndarray): The simulated MRI image with a wrap-around artefact (if applicable)'''
    
    # Unpack mandatory scan parameters
    PE_direction = scan_parameters.get["PE_direction"]
    if PE_direction is None:
        raise ValueError("Please choose a PE direction")
    plane = scan_parameters["plane"]
    if plane is None:
        raise ValueError("Please choose a plane")
    
    # Unpack optional parameters (based on chosen direction and plane)
    FOV_unit = scan_parameters.get["FOV_unit", 'Pixels']
    FOV_AP = scan_parameters.get["FOV_AP", 216]
    FOV_LR = scan_parameters.get["FOV_LR", 180]
    FOV_FH = scan_parameters.get["FOV_FH", 180]
    FOV_center_AP = scan_parameters.get["FOV_center_AP", 108]
    FOV_center_LR = scan_parameters.get["FOV_center_LR", 90]
    FOV_center_FH = scan_parameters.get["FOV_center_FH",90]

    scaling_area = False # Does work, but is not convenient. Needs extra work to see if you can get it working. Can be added to variables later if it works
    
    # Rescale values in case FOV is taken in meters or centimeters (not necessary if FOV in pixels or millimeters)
    if FOV_unit == 'metre':
        FOV_AP = FOV_AP*1000
        FOV_LR = FOV_LR*1000
        FOV_FH = FOV_FH*1000
        FOV_center_AP = FOV_center_AP*1000
        FOV_center_LR = FOV_center_AP*1000
        FOV_center_FH = FOV_center_AP*1000
    
    if FOV_unit == 'centimetre':
        FOV_AP = FOV_AP*10
        FOV_LR = FOV_LR*10
        FOV_FH = FOV_FH*10
        FOV_center_AP = FOV_center_AP*10
        FOV_center_LR = FOV_center_AP*10
        FOV_center_FH = FOV_center_AP*10
    
    # Calculate the starting and ending indices for the FOV
    start_AP = FOV_center_AP - FOV_AP // 2
    start_LR = FOV_center_LR - FOV_LR // 2
    start_FH = FOV_center_FH - FOV_FH // 2
    end_AP = start_AP + FOV_AP
    end_LR = start_LR + FOV_LR
    end_FH = start_FH + FOV_FH

    # Check if the FOV is out of bounds, else create I_FOV
    if start_AP < 0 or start_LR < 0 or end_AP > signal_array.shape[0] or end_LR > signal_array.shape[1] or start_FH < 0 or end_FH > signal_array.shape[2]:
        if start_AP < 0 or end_AP > signal_array.shape[0]:
            print("Warning: FOV is out of bounds in the AP direction. Choose a smaller FOV in AP direction")
        if start_LR < 0 or end_LR > signal_array.shape[1]:
            print("Warning: FOV is out of bounds in the LR direction. Choose a smaller FOV in LR direction")
        if start_FH < 0 or end_FH > signal_array.shape[2]:
            print("Warning: FOV is out of bounds in the FH direction. Choose a smaller FOV in FH direction")
    else:
        I_FOV = signal_array[start_AP:end_AP, start_LR:end_LR, start_FH:end_FH]
        I_mask_FOV = masking(I_FOV)
    
    # Find cut-off sides outside FOV in PE direction
    if (plane == 'Transverse' and PE_direction == 'Horizontal') or (plane == 'Coronal' and PE_direction == 'Horizontal'):
        I_A = signal_array[start_AP:end_AP, :start_LR, start_FH:end_FH]
        I_B = signal_array[start_AP:end_AP, end_LR:, start_FH:end_FH]
        I_direction_A = 'FOV_left'
        I_direction_B = 'FOV_right'
        
    elif (plane == 'Transverse' and PE_direction == 'Vertical') or (plane == 'Sagittal' and PE_direction == 'Horizontal'):
        I_A = signal_array[:start_AP, start_LR:end_LR, start_FH:end_FH]
        I_B = signal_array[end_AP:, start_LR:end_LR, start_FH:end_FH]
        I_direction_A = 'FOV_head'
        I_direction_B = 'FOV_feet'
    
    elif (plane == 'Sagittal' and PE_direction == 'Vertical') or (plane == 'Coronal' and PE_direction == 'Vertical'):
        I_A = signal_array[start_AP:end_AP, start_LR:end_LR, :start_FH]
        I_B = signal_array[start_AP:end_AP, start_LR:end_LR, end_FH:]
        I_direction_A = 'FOV_anterior'
        I_direction_B = 'FOV_posterior'
    
    else:
        print('The combination of chosen plane and phase encoding direction will not show a wrap around artifact in the simulation')
    
    # Shift cut-off sides to other side of FOV by adding padding and create masks
    I_A = I_A[:FOV_AP, :FOV_LR, :FOV_FH]    
    I_A_padded = FOV_padding(I_A, I_direction_A, FOV_AP, FOV_LR, FOV_FH, FOV_center_AP, FOV_center_LR, FOV_center_FH)  
    I_mask_A = masking(I_A_padded)

    I_B = I_B[:FOV_AP, :FOV_LR, :FOV_FH]  
    I_B_padded = FOV_padding(I_B, I_direction_B, FOV_AP, FOV_LR, FOV_FH, FOV_center_AP, FOV_center_LR, FOV_center_FH)   
    I_mask_B = masking(I_B_padded)
    
    # Combine FOV and cut-off sides to create unscaled wrap-around image
    I_wrap = I_FOV + I_A_padded + I_B_padded
    
    # Seperate "single layer" (XOR) and "double layer" (AND)
    I_mask_A_B = I_mask_A + I_mask_B
    I_mask_XOR = np.logical_xor(I_mask_FOV,I_mask_A_B)
    I_mask_AND = np.logical_and(I_mask_FOV,I_mask_A_B)
    
    I_XOR = I_wrap*I_mask_XOR
    I_AND = I_wrap*I_mask_AND
    
    # Scale "double layer" depending on chosen setting
    if scaling_area == True:
        """
        This scaling is based on the relative area of the double layer compared to the single layer.
        The shift is currently too harsh to use with FOVs close to the entire image.
        """
        value_overlap = I_mask_A + I_mask_B
        sum_overlap = np.sum(value_overlap)
    
        value_main = I_mask_XOR
        sum_main = np.sum(value_main)
        
        I_AND_scaled = I_AND*float((sum_overlap/sum_main))
    
        print('The wrap around artefact has been scaled by ',sum_overlap/sum_main)
    
    else:
        I_AND_scaled = I_AND*0.5
    
    # Final scaled image 
    I_OUT = I_XOR + I_AND_scaled
    
    return I_OUT



def fat_shift_1(signal_array, scan_parameters):
    '''Simulate a chemical shift artefact of the first kind based on the chosen settings of the simulator. 
    
    Args:
    Signal_array (np.ndarray): contains the signal intensity of each voxel
    Scan_parameters (dict): contains the chosen settings and should contain at least the following parameters
        gyromagnetic ratio (float): the gyromagnetic ratio used in the simulation
        field strength (int): the field strenght of the the MRI (most likely 1.5 or 3 T) [T]
        bandwidth (int): range of frequencies involved in the reception of the signal [Hz]
        resolution (int): amount of voxels along the frequency encoding axis
        FE direction (str): frequency encoding direction, either horizontal or vertical, based on the chosen plane
            
    Returns:
        shift (int): will tell you the amount of pixels the fat shifted, based on the chosen settings
        Chemical_shift_2 (np.ndarray): The simulated MRI image with a chemical shift artefact of the first kind'''
    
    
    # Unpack the scan parameters
    gamma = scan_parameters.get("gyromagnetic ratio", 2.6754*10**8)
    B_0 = scan_parameters.get("field strength")
    BW = scan_parameters.get("bandwidth")
    N = scan_parameters.get("resolution")
    FE_direction = scan_parameters.get("FE_direction")
    
    if FE_direction == 'Horizontal':
        axis_shift = 1
    elif FE_direction == 'Vertical':
        axis_shift = 0

    # Calculate the fat shift in voxels
    shift = -((3.5 * 10**-6) * gamma * B_0) / (BW / N)
    shift = int(round(shift))
    print("The fat was shifted by", abs(shift), "voxels in the", FE_direction, "direction")

    # Shift the fat signal
    chosen_fat = signal_array[:,:,:, 1]     
    shifted_fat = np.roll(chosen_fat, shift, axis=axis_shift)
    
    # Combine the shifted fat signal with the water signal
    chosen_water = signal_array[:,:,:,0]        
    shifted_waterfatshift =  shifted_fat + chosen_water
    
    return shifted_waterfatshift



def fat_shift_1_and_2(signal_array, **scan_parameters):
    '''Simulate a chemical shift artefact of the first kind based on the chosen settings of the simulator. 
    
    Args:
    Signal_array (np.ndarray): contains the signal intensity of each voxel
    Scan_parameters (dict): contains the chosen settings and should contain at least the following parameters
        gyromagnetic ratio (float): the gyromagnetic ratio used in the simulation
        field strength (int): the field strenght of the the MRI (most likely 1.5 or 3 T) [T]
        bandwidth (int): range of frequencies involved in the reception of the signal [Hz]
        resolution (int): amount of voxels along the frequency encoding axis
        FE direction (str): frequency encoding direction, either horizontal or vertical, based on the chosen plane
        TE (float): echo time in [s]
        in phase (bool): if water and fat should be in phase
        out of phase (bool): if water and fat should be out of phase
     
    Returns:
        shift (int): will tell you the amount of pixels the fat shifted, based on the chosen settings
        TE (flaot): if "in phase" or "out of phase" were chosen to be true, the optimal TE time respectively will be printed
        Chemical_shift_2 (np.ndarray): The simulated MRI image with a chemical shift artefact of the second kind'''
    

    # Unpack the scan parameters
    gamma = scan_parameters.get("gyromagnetic ratio", 2.6754*10**8)
    B_0 = scan_parameters.get("field strength")
    FE_direction = scan_parameters.get("FE_direction")
    TE = scan_parameters.get("TE")
    in_phase = scan_parameters.get("in phase")
    out_of_phase = scan_parameters.get("out of phase")
    BW = scan_parameters.get("bandwidth")
    N = scan_parameters.get("resolution")
    
    # Values of the fat shift (ppm) and resonance frequency (RF)
    ppm = 3.5*10**(-6)
    RF = 62*10**6 * (B_0/1.5)
    
    if FE_direction == 'Horizontal':
        axis_shift = 1
    elif FE_direction == 'Vertical':
        axis_shift = 0

    # Calculate the fat shift in voxels
    shift = -((3.5 * 10**-6) * gamma * B_0) / (BW / N)
    shift = int(round(shift))
    print("The fat was shifted by", abs(shift), "voxels in the", FE_direction, "direction")

    # Shift the fat signal
    chosen_fat = signal_array[:,:,:, 1]     
    shifted_fat = np.roll(chosen_fat, shift, axis=axis_shift)
    
    # Combine the shifted fat signal with the water signal
    chosen_water = signal_array[:,:,:,0]        
    shifted_waterfatshift =  shifted_fat + chosen_water
    
    # Time for the shift of the second kind
    # Frequency of water and fat 
    f_water = gamma*B_0
    f_fat = f_water + 2*math.pi*ppm*RF
    
    # Determine optimal TE if out of phase
    if out_of_phase == True:
        TE = 0.5/(ppm*RF)
        print('Optimal TE time:', round(TE,4),'s')
    
    # Determine optimal TE if in phase
    if in_phase == True:
        TE = 1/(ppm*RF)
        print('Optimal TE time:', round(TE,4),'s')
    
    # Based on the TE (either chosen or calculated), find the average signal in the border voxels  
    average_phase = abs(np.sin(f_water*TE) + np.sin(f_fat*TE))*0.5
    
    # Create masks and find the overlapping pixels
    fat_mask = shifted_fat > 0   
    water_mask = chosen_water > 0   
    fat_border = np.logical_and(fat_mask, water_mask)
    fat_border = np.invert(fat_border).astype(float)

    # Multiply the overlap pixels (border around the fat) with the average phase in the voxels (in phase = 1, out of phase = 0)
    fat_border[fat_border == 0] = average_phase
    chemical_shift_1_and_2 =  np.multiply(shifted_waterfatshift,fat_border)
    
    return chemical_shift_1_and_2



def fat_shift_2(signal_array, **scan_parameters):
    '''Simulate a chemical shift artefact of the second kind based on the chosen settings of the simulator. 
    
    Args:
    Signal_array (np.ndarray): contains the signal intensity of each voxel
    Scan_parameters (dict): contains the chosen settings and should contain at least the following parameters
        gyromagnetic ratio (float): the gyromagnetic ratio used in the simulation
        field strength (int): the field strenght of the the MRI (most likely 1.5 or 3 T)
        TE (float): echo time in [s]
        in phase (bool): if water and fat should be in phase
        out of phase (bool): if water and fat should be out of phase
            
    Returns:
        TE (flaot): if "in phase" or "out of phase" were chosen to be true, the optimal TE time respectively will be printed
        Chemical_shift_2 (np.ndarray): The simulated MRI image with a chemical shift artefact of the second kind'''
    
    # Unpack the scan parameters
    gamma = scan_parameters.get("gyromagnetic ratio", 2.6754*10**8)
    B_0 = scan_parameters.get("field strength")
    TE = scan_parameters.get("TE")
    in_phase = scan_parameters.get("in phase")
    out_of_phase = scan_parameters.get("out of phase")
    
    # Values of the fat shift (ppm) and resonance frequency (RF)
    ppm = 3.5*10**(-6)
    RF = 62*10**6 * (B_0/1.5)
    
    # Frequency of water and fat 
    f_water = gamma*B_0
    f_fat = f_water + 2*math.pi*ppm*RF
    
    # Determine optimal TE if out of phase
    if out_of_phase == True:
        TE = 0.5/(ppm*RF)
        print('Optimal TE time:', round(TE,4),'s')
    
    # Determine optimal TE if in phase
    if in_phase == True:
        TE = 1/(ppm*RF)
        print('Optimal TE time:', round(TE,4),'s')
    
    # Based on the TE (either chosen or calculated), find the average signal in the border voxels  
    average_phase = abs(np.sin(f_water*TE) + np.sin(f_fat*TE))*0.5
    
    fat_signal = signal_array[:,:,:,1]
    water_signal = signal_array[:,:,:,0]
    signal = water_signal + fat_signal
    
    # Create the water-fat border by creating masks of both water and fat. Dilating the fat to get overlaying voxels and then finding the overlapping voxels.
    fat_mask = fat_signal > 0   
    water_mask = water_signal > 0   
    dilated_fat_signal = ndimage.grey_dilation(fat_mask, size=(4,4,4)) 
    fat_border = np.logical_and(dilated_fat_signal, water_mask)
    fat_border = np.invert(fat_border).astype(float)

    # Multiply the overlap pixels (border around the fat) with the average phase in the voxels (in phase = 1, out of phase = 0)
    fat_border[fat_border == 0] = average_phase
    
    chemical_shift_2 =  np.multiply(signal,fat_border)
    
    return chemical_shift_2


def main():
    
    Brain_phantom = False
    WaterFatShift_phantom = True
    
    if Brain_phantom == True:
        #This functions loads the anatomical brain model from the .mat file and returns a dictionary of the data contained in the anatomical model. The 16 data fields contained in data_dict are described in https://mrilab.sourceforge.net/manual/MRiLab_User_Guide_v1_3/MRiLab_User_Guidech3.html#x8-120003.1
        data_dict = load_data("BrainHighResolution.mat") 
        model = Model("BrainHighResolution", data_dict["T1"], data_dict["T2"], data_dict["T2Star"], data_dict["Rho"])
        
    if WaterFatShift_phantom == True:
        #This functions loads the waterfatshift phantom from the .mat file and returns a dictionary of the data contained in the model.
        data_dict = load_data("WaterFatPhantom.mat")
        model = Model("WaterFatPhantom.mat", data_dict["T1"], data_dict["T2"], data_dict["T2Star"], data_dict["Rho"])
        
    gyro_ratio = data_dict["Gyro"][0][0].item()
        
    # Simulate a scan
    scan_parameters = {
        "TE": 0.014,
        "TR": 0.044,
        "TI": 0.140,
        "FOV_AP" : 150,
        "FOV_LR" : 180,
        "FOV_FH" : 180,
        "FOV_center_AP" : 108,
        "FOV_center_LR" : 90,
        "FOV_center_FH" : 90,
        "PE_direction" : 'Horizontal',
        "FE_direction" : 'Vertical',
        "plane" : 'Transverse',
        "slice_nr" : 16,
        "field strength" : 1.5,
        "in phase" : False,
        "out of phase" : True,
        "bandwidth" : 50*10**3,
        "resolution" : 256,
        "gyromagnetic ratio" : gyro_ratio,
        "WaterFat phantom" : WaterFatShift_phantom,
        "Artefact" : 'Chem_shift_1'
        }
        
    # Make sure to select the correct artefact (Wrap_around, Chem_shift_1, Chem_shift_2)
    simulated_image = scan(scan_parameters, model)
    
    #Show image
    axes_image(simulated_image, **scan_parameters)

    
if __name__ == "__main__":
    main()
    