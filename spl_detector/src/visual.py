# Zweck: Hilfsfunktionen zur Visualisierung von Bildern + Labels
# Author: David Kostka
# Datum: 05.11.2020

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import util
import numpy

def plot_bbox(img, bbox):
    '''
    Plotte ein Bild mit eingezeichneter BBOX
    Input: Bild, BBOX
    Output: Kein Return, aber ausgabe des Plots
    '''
    plt.imshow(img, cmap='gray')

    if len(bbox) is 4:
        x, y, xm, ym = bbox.astype('int')
    else: 
        x, y, xm, ym, is_obj = bbox.astype('int')
        if is_obj:
            bbox = Rectangle((x, y), xm - x, ym - y, linewidth=1, edgecolor='red', facecolor='none')
            plt.gca().add_patch(bbox)
            #plt.text(x, y, str([x, y, xm, ym]), color='red')
        else:
            plt.text(0, 0, 'No Object found', color='red')

    plt.show()

'''
def visualize_dataset(dataset, model, target_size, count=1):
    pred = model.predict(dataset)
    i = 0

    for image, _ in dataset:
        if i > count: break
        print(pred[i])
        #plot_bbox(image, util.unnormalize_bbox(np.array(label), target_size))
        plot_bbox(image, util.unnormalize_bbox(pred[i], target_size))
        #plot_bbox(image, pred[i])
        i += 1
'''