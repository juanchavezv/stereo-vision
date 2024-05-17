"""
Sparse 3D reconstruction using stereo vision - This script processes two rectified infrared 
images, permitting the user to choose up to 30 pixels in each for three-dimensional position 
calculation.

python stereo-vision.py --l_img left_infrared_image.png --r_img right_infrared_image.png
"""

import argparse
import cv2
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys

def parse_user_data() :

    parser = argparse.ArgumentParser(prog='Sparse 3D reconstruction using stereo vision',
                                    description='Select 2D points and generate a basic 3D reconstruction', 
                                    epilog='Juan Carlos Chávez Villarreal - 2024')
    parser.add_argument('-left_image',
                        '--l_img',
                        type=str,
                        required=True,
                        help="Path to the left image")
    parser.add_argument('-right_image',
                        '--r_img',
                        type=str,
                        required=True,
                        help="Path to the right image")
    
    args = parser.parse_args()
    return args

def image_load(img):
    """
    load an image from a specified file path.
    
    input: 
        img - string path to the image file
    output: 
        img - loaded image object or None if file not found or format unsupported
    """
    img = cv2.imread(img)
    if img is None:
        print(f"File not found or unsupported format: {img}")
        return None
    return img

def image_resize(img):
    """
    resize an image to its original size.
    
    input: 
        img - image object
    output: 
        img_resize - resized image object
    """
    width = int(img.shape[1] * 1) 
    height = int(img.shape[0] * 1)
    img_resize = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return img_resize

def image_visualise(img, title):
    """
    display an image with a specified title.

    input: 
        img - image object
        title - string title for the image window
    output: 
        none
    """
    img_resize = image_resize(img) 
    cv2.imshow(title, img_resize)
    if True:
        cv2.waitKey(0)

def load_calibration_data(calibration_file):
    """
    load camera calibration parameters from a json file.
    
    input: 
        calibration_file - string path to the json file containing calibration data
    output: 
        calibration_data - dictionary containing calibration parameters
    """
    try:
        with open(calibration_file, 'r') as file:
            calibration_data = json.load(file)
        return calibration_data
    except FileNotFoundError:
        print("Calibration file not found.")
        return None
    except json.JSONDecodeError:
        print("Error decoding the calibration file.")
        return None

def point_selection(img):
    """
    allow user to manually select points on an image.
    
    input: 
        img - image object on which points are to be selected
    output: 
        points - array of selected points
    """
    title = 'Point Selection on Left Image'
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    points = np.asarray(plt.ginput(30, timeout=-1))
    plt.close(fig)
    return points

def match_and_select_points(left_image, left_points , right_image):
    """
    display selected points from the left image on the right image and allow 
    additional point selection.
    
    input: 
        l_img - left image object
        left_points - array of points selected on the left image
        r_img - right image object
    output: 
        right_points - array of points selected on the right image corresponding 
        to the left points
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(left_image, cmap='gray')
    axs[0].plot(left_points[:, 0], left_points[:, 1], 'ro', markersize=1.5)
    axs[0].set_title('Left Image with Selected Points')
    axs[1].imshow(right_image, cmap='gray')
    axs[1].set_title('Select points on Right Image')

    right_points = np.asarray(plt.ginput(30, timeout=-1)) 
    
    plt.close(fig)
    
    if right_points.size == 0:
        print("Error: No points were selected on the right image.")
        return None
    return right_points

def compute_3d_coordinates(left_pts, right_pts):
    """
    calculate 3d coordinates from stereo image points based on disparity.

    input: 
        left_pts - array of points from left image
        right_pts - array of points from right image
    output: 
        coordinates - array of calculated 3d coordinates
    """
    try:
        disparity = left_pts[:, 0] - right_pts[:, 0]
        Z = np.where(disparity != 0, (f * B) / disparity, np.inf)
        X = np.where(disparity != 0, (left_pts[:, 0] - cx) * Z / f, 0)
        Y = np.where(disparity != 0, (left_pts[:, 1] - cy) * Z / f, 0)
        return np.column_stack((X, Y, Z))
    except Exception as e:
        print(f"Failed to calculate 3D coordinates: {e}")
        return None

def close_windows() -> None:

    cv2.destroyAllWindows()

def pipeline():
    global B, f, cx, cy 

    calibration_data = load_calibration_data('calibration-parameters.txt')

    B = abs(float(calibration_data['baseline']))
    f = float(calibration_data['rectified_fx'])
    cx = float(calibration_data['rectified_cx'])  
    cy = float(calibration_data['rectified_cy'])
    
    user_input =  parse_user_data()
    l_img = image_load(user_input.l_img)
    r_img = image_load(user_input.r_img)

    image_visualise(l_img,"Left Image")
    image_visualise(r_img,"Right Image")
    close_windows()

    left_points = point_selection(l_img)
    right_points = match_and_select_points(l_img, left_points, r_img)
    coordinates_3d = compute_3d_coordinates(left_points, right_points)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coordinates_3d[:, 0], coordinates_3d[:, 2], -coordinates_3d[:, 1], c='purple', edgecolor='black', s=50, alpha=0.6)
    plt.axis('Equal')
    ax.set_xlabel('X (mm)', fontweight='bold')
    ax.set_ylabel('Z (mm)', fontweight='bold')
    ax.set_zlabel('Y (mm)', fontweight='bold')
    ax.set_title('3D Reconstruction of Selected Points', fontsize=14, fontweight='bold')
    ax.view_init(elev=25, azim=90)
    ax.grid(False)

    plt.show()

if __name__ == "__main__":
    pipeline()