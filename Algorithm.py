

# Imports
import matplotlib.pyplot as plt

import matplotlib

matplotlib.use('Agg')

import matplotlib.patches as patches

import base64

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from io import BytesIO

import math

import numpy as np

from Algorithm_RIS import run_mse_esti


def show_objects(coordinates = None, esti_theta = None, esti_d = None): 

    # true position coordinates
    detect_objects = [(1,1), (3,1), (8,5), (2,2), (3,5)] 

    # estimated position coordinates
    estimated_objects = [(1,2), (3,1.2), (3,4.5), (1,1.5), (8,4)]

    base_st = [(3,0)] # base station position 

    reflective_st = [(0,4)] # RIS position

    RIS_pos = np.array([0, 4, 0])

    # Size of figure
    plt.figure(figsize=(7,7))

    # Plot true objects coordinates
    for dots in detect_objects:
        plt.scatter(*zip(*detect_objects), color='red', label = 'True objects')
        plt.text(dots[0], dots[1], f'({dots[0]}, {dots[1]})', fontsize=9)

    
    for i in range(len(esti_theta)):

        theta = esti_theta[i]

        d = esti_d[i]

        deltaX_r = d * np.cos(np.deg2rad(theta))

        deltaY_r = d * np.sin(np.deg2rad(theta))

        esti_obj_pos = [RIS_pos[0] + deltaX_r, RIS_pos[1] + deltaY_r, 0]

        # Plot estimated object positions based on theta and d estimates
        plt.scatter(*esti_obj_pos, color='red', label = 'True objects')
        plt.text(esti_obj_pos[0], esti_obj_pos[1], f'({esti_obj_pos[0]}, {esti_obj_pos[1]})', fontsize=9)
        


    # Limits of x and y-axis
    plt.xlim(0, 12)
    plt.ylim(0, 8)

    # Title of the Graph
    plt.title("Room setting with objects")

    # Plot estimated objects coordinates
    for rect in estimated_objects:
        detect = patches.Rectangle((rect[0] - 0.25, rect[1] - 0.25), 0.2, 0.2, edgecolor='blue', facecolor = 'none')
        plt.gca().add_patch(detect)
        plt.text(rect[0], rect[1], f'({rect[0]}, {rect[1]})', fontsize = 9)


    # Plot base station
    for base in base_st:
        # Base station is red
        b = patches.Rectangle((base[0] - 0.3, base[1] - 0.2), 0.3, 0.5, edgecolor = 'red', facecolor = 'red')
        # Adding the base station to the plot
        plt.gca().add_patch(b)
        # Adding text paragraph to graph
        plt.text(base[0] + 0.2, base[1], f'({base[0]}, {base[1]})', fontsize = 9)
    
    # Plot reflective station
    for reflective in reflective_st:
        # Reflective station is blue
        ref = patches.Rectangle((reflective[0] - 0.3, reflective[1] - 0.2), 0.6, 0.5, edgecolor = 'blue', facecolor = 'blue')
        # Adding the reflective station to the plot
        plt.gca().add_patch(ref)
        # Adding text paragraph to graph
        plt.text(reflective[0] + 0.4, reflective[1], f'({reflective[0]}, {reflective[1]})', fontsize = 9)
    
    # Add description of what is true and estimated objects on plot
    plt.legend(handles = [
        # Adding legend for true objects
        plt.Line2D([0], [0], marker = 'o', color='w', label = 'True objects', markerfacecolor ='red', markersize = 10),
        # Adding legend for estimated objects
        patches.Patch(color = 'blue', label = "Estimated objects", fill = False),
        # Adding legend for base station 
        patches.Patch(color ='red', label = "Base station", fill = True),
        # Adding legend for reflective station
        patches.Patch(color = 'blue', label = "RIS", fill = True)
    ])

    # Plot coordinates of objects based on input
    if coordinates is not None:
        for x, y in coordinates:
            plt.scatter(x, y, color='red')
            plt.text(x, y, f'({x}, {y})', fontsize = 9)

    # Save figure as png 
    img = BytesIO()

    # Save figure for the html main page
    plt.savefig(img, format='png')

    img.seek(0)

    # Encode image as base64
    plot_url = base64.b64encode(img.read()).decode('utf-8')

    #plot_url = base64.b64encode(img.getvalue()).decode()

    # Close figure
    img.close()

    # Return plot url
    return f"data:image/png;base64,{plot_url}"




