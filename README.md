# BEP-Medical-Imaging-TUe
Temporary repository for the code of F. van Veens BEP in Medical Imaging at the TU/e

This is a Python script for simulating MRI scans with various artifacts. The script defines several functions to load data, calculate signal intensities, add artifacts, and visualize the results. Here is a brief overview of the main functions:

- **load_data(path_to_data)**: Loads data from a .mat file and returns a dictionary containing the fields in the .mat file. The .mat file contains a 1x1 struct called VObj, which contains 16 fields describing the data.

- **scan(scan_parameters, model)**: Simulates a scan of the model using the given scan parameters. The scan parameters include TR, TE, and flip angle. The function returns the simulated MRI image.

- **calculate_signal(scan_parameters, model)**: Calculates the signal intensity for each voxel in the model, given the scan parameters. The signal intensity is calculated using the signal equation for a spin echo sequence.

- **axes_image(model, scan_parameters)**: Shows an image of the model in the chosen axial direction (plane) and slice number.

- **masking(image)**: Creates a mask of the image and ensures that it does not contain any unnecessary holes.

- **FOV_padding(part_array, direction, FOV_AP, FOV_LR, FOV_FH, FOV_center_AP, FOV_center_LR, FOV_center_FH)**: Shifts the cut-off sides of the field of view by adding padding to the outside, making them the same size as the field of view.

- **add_aliasing(signal_array, scan_parameters)**: Simulates a wrap-around artifact based on the chosen settings of the simulator.

- **fat_shift_1(signal_array, scan_parameters)**: Simulates a chemical shift artifact of the first kind based on the chosen settings of the simulator.

- **fat_shift_1_and_2(signal_array, scan_parameters)**: Simulates a chemical shift artifact of the first and second kind combined based on the chosen settings of the simulator.

- **fat_shift_2(signal_array, scan_parameters)**: Simulates a chemical shift artifact of the second kind based on the chosen settings of the simulator.

The script also includes a main() function that loads the anatomical brain model or the water-fat shift phantom, sets the scan parameters, simulates the scan with the chosen artifact, and visualizes the results. The user can choose which model to load, which artifact to add, and the scan parameters to use. The script also includes several print statements to provide feedback to the user.
