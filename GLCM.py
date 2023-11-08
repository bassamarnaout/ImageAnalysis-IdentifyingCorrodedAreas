#!/usr/bin/env python


import random
import math
import functools

import numpy as np
import copy



"""
###################################################################################################
function : imageTextureMeasuresFromGLCM()
###################################################################################################
"""
def imageTextureMeasuresFromGLCM(image, texture_type='none', gray_levels=256, displacement=[0, 1]):


    if texture_type == 'random':
      image = [[random.randint(0,gray_levels-1)
                        for _ in range(IMAGE_SIZE)] for _ in range(IMAGE_SIZE)]
    elif texture_type == 'diagonal':
        image = [[gray_levels - 1 if (i+j)%2 == 0 else 0
                        for i in range(IMAGE_SIZE)] for j in range(IMAGE_SIZE)]
    elif texture_type == 'vertical':
        image = [[gray_levels - 1 if i%2 == 0 else 0
                        for i in range(IMAGE_SIZE)] for _ in range(IMAGE_SIZE)]
    elif texture_type == 'horizontal':
        image = [[gray_levels - 1 if j%2 == 0 else 0
                        for i in range(IMAGE_SIZE)] for j in range(IMAGE_SIZE)]
    elif texture_type == 'checkerboard':
        image = [[gray_levels - 1 if (i+j+1)%2 == 0 else 0
                        for i in range(IMAGE_SIZE)] for j in range(IMAGE_SIZE)]
    elif texture_type == 'low_contrast':
        image[0] = [random.randint(0,gray_levels-1) for _ in range(IMAGE_SIZE)]
        for i in range(1,IMAGE_SIZE):
            image[i][0] = random.randint(0,gray_levels-1)
            for j in range(1,IMAGE_SIZE):
                image[i][j] = image[i-1][j-1]
    else:
        # grab the image dimensions
        IMAGE_SIZE_HEIGHT = image.shape[0]  # Height if the image
        IMAGE_SIZE_WIDTH = image.shape[1]  # Width of the image


             
    glcm = [[0 for _ in range(gray_levels)] for _ in range(gray_levels)]

    rowmax = IMAGE_SIZE_HEIGHT - displacement[0] if displacement[0] else IMAGE_SIZE_HEIGHT -1
    colmax = IMAGE_SIZE_WIDTH - displacement[1] if displacement[1] else IMAGE_SIZE_WIDTH -1

    for i in range(rowmax):
        for j in range(colmax):
            m, n =  image[i][j], image[i + displacement[0]][j + displacement[1]]
            glcm[m][n] += 1
            glcm[n][m] += 1

    # CALCULATE ATTRIBUTES OF THE GLCM MATRIX:

    entropy = energy = contrast = \
        homogeneity = dissimilarity = correlation = None
    mu_m = mu_n = segma_m = segma_n = None

    normalizer = functools.reduce(lambda x,y: x + sum(y), glcm, 0)

    if normalizer != 0:
        for m in range(len(glcm)):
            for n in range(len(glcm[0])):
                prob = (1.0 * glcm[m][n]) / normalizer
                if (prob >= 0.0001) and (prob <= 0.999):
                    log_prob = math.log(prob,2)
                if prob < 0.0001:
                    log_prob = 0
                if prob > 0.999:
                    log_prob = 0
                if entropy is None:
                    entropy = -1.0 * prob * log_prob
                    continue
                entropy += -1.0 * prob * log_prob
                if energy is None:
                    energy = prob ** 2
                    continue
                energy += prob ** 2
                if contrast is None:
                    contrast = ((m - n)**2 ) * prob
                    continue
                contrast += ((m - n) ** 2) * prob
                if dissimilarity is None:
                    dissimilarity = abs(m - n) * prob
                    continue
                dissimilarity += abs(m - n) * prob
                if (mu_m is None) or (mu_n is None) or (segma_m is None) or (segma_n is None):
                    mu_m = m * prob
                    segma_m = ((m-mu_m)**2) * prob
                    mu_n = n * prob
                    segma_n = ((n-mu_n)**2) * prob
                    continue
                mu_m += m * prob
                segma_m += ((m - mu_m) ** 2) * prob
                mu_n += n * prob
                segma_n += ((n - mu_n) ** 2) * prob
                if segma_m !=0 and segma_n !=0:
                    if correlation is None:
                        correlation = ((m-mu_m)*(n-mu_n)*prob) / (segma_m * segma_n)
                        continue
                    correlation += ((m-mu_m)*(n-mu_n)*prob) / (segma_m * segma_n)
                if homogeneity is None:
                    homogeneity = prob / ( ( 1 + abs(m - n) ) ** 2.0 )
                    continue
                homogeneity += prob / ( (1  + abs(m - n) ) ** 2.0 )

        if abs(entropy) < 0.0000001: entropy = 0.0
        # print("\nTexture attributes: ")
        # print("    entropy: %f" %  entropy)
        # print("    contrast: %f" % contrast)
        # print("    homogeneity: %f" % homogeneity)
        # print("    energy: %f" %  energy)
        # print("    dissimilarity: %f" % dissimilarity)
        # print("    correlation: %f" % correlation)

    return [glcm, entropy, energy, contrast, homogeneity, dissimilarity, correlation]






"""
###################################################################################################
function : imageColouredTextureMeasuresFromGLCM()
###################################################################################################
"""
def imageColouredTextureMeasuresFromGLCM(imageColoured, gray_levels_per_channel=256, displacement=[0, 1]):

    ratio_For_downSampledImage = int(256 / gray_levels_per_channel)

    imageColoured = np.floor_divide(copy.copy(imageColoured), ratio_For_downSampledImage)

    # INITALIZE ATTRIBUTES OF THE GLCM MATRIX:
    entropy = []
    energy = []
    contrast = []
    homogeneity = []
    dissimilarity = []
    correlation = []
    mu_m = []
    mu_n = []
    segma_m = []
    segma_n = []


    for rgbChannel in range(3):

        imageColoured_Channel = imageColoured[:, :, rgbChannel]

        # grab the image dimensions
        IMAGE_SIZE_HEIGHT = imageColoured.shape[0]  # Height if the image
        IMAGE_SIZE_WIDTH = imageColoured.shape[1]  # Width of the image


        # CALCULATE THE GLCM MATRIX:
        glcm = [[0 for _ in range(gray_levels_per_channel)] for _ in range(gray_levels_per_channel)]

        rowmax = IMAGE_SIZE_HEIGHT - displacement[0] if displacement[0] else IMAGE_SIZE_HEIGHT - 1
        colmax = IMAGE_SIZE_WIDTH - displacement[1] if displacement[1] else IMAGE_SIZE_WIDTH - 1

        for i in range(rowmax):
            for j in range(colmax):
                m, n = imageColoured_Channel[i][j], imageColoured_Channel[i + displacement[0]][j + displacement[1]]
                glcm[m][n] += 1
                glcm[n][m] += 1


        # INITIALIZE ATTRIBUTES OF THE GLCM MATRIX:

        entropy.append(None)
        energy.append(None)
        contrast.append(None)
        homogeneity.append(None)
        dissimilarity.append(None)
        correlation.append(None)
        mu_m.append(None)
        mu_n.append(None)
        segma_m.append(None)
        segma_n.append(None)

        # CALCULATE ATTRIBUTES OF THE GLCM MATRIX:
        normalizer = functools.reduce(lambda x, y: x + sum(y), glcm, 0)

        if normalizer != 0:
            for m in range(len(glcm)):
                for n in range(len(glcm[0])):
                    prob = (1.0 * glcm[m][n]) / normalizer
                    if (prob >= 0.0001) and (prob <= 0.999):
                        log_prob = math.log(prob, 2)
                    if prob < 0.0001:
                        log_prob = 0
                    if prob > 0.999:
                        log_prob = 0
                    if entropy[rgbChannel] is None:
                        entropy[rgbChannel] = -1.0 * prob * log_prob
                        continue
                    entropy[rgbChannel] += -1.0 * prob * log_prob
                    if energy[rgbChannel] is None:
                        energy[rgbChannel] = prob ** 2
                        continue
                    energy[rgbChannel] += prob ** 2
                    if contrast[rgbChannel] is None:
                        contrast[rgbChannel] = ((m - n) ** 2) * prob
                        continue
                    contrast[rgbChannel] += ((m - n) ** 2) * prob
                    if dissimilarity[rgbChannel] is None:
                        dissimilarity[rgbChannel] = abs(m - n) * prob
                        continue
                    dissimilarity[rgbChannel] += abs(m - n) * prob
                    if (mu_m[rgbChannel] is None) or (mu_n[rgbChannel] is None) or (segma_m[rgbChannel] is None) or (segma_n[rgbChannel] is None):
                        mu_m[rgbChannel] = m * prob
                        segma_m[rgbChannel] = ((m - mu_m[rgbChannel]) ** 2) * prob
                        mu_n[rgbChannel] = n * prob
                        segma_n[rgbChannel] = ((n - mu_n[rgbChannel]) ** 2) * prob
                        continue
                    mu_m[rgbChannel] += m * prob
                    segma_m[rgbChannel] += ((m - mu_m[rgbChannel]) ** 2) * prob
                    mu_n[rgbChannel] += n * prob
                    segma_n[rgbChannel] += ((n - mu_n[rgbChannel]) ** 2) * prob
                    if segma_m[rgbChannel] != 0 and segma_n[rgbChannel] != 0:
                        if correlation[rgbChannel] is None:
                            correlation[rgbChannel] = ((m - mu_m[rgbChannel]) * (n - mu_n[rgbChannel]) * prob) / (segma_m[rgbChannel] * segma_n[rgbChannel])
                            continue
                        correlation[rgbChannel] += ((m - mu_m[rgbChannel]) * (n - mu_n[rgbChannel]) * prob) / (segma_m[rgbChannel] * segma_n[rgbChannel])
                    if homogeneity[rgbChannel] is None:
                        homogeneity[rgbChannel] = prob / ((1 + abs(m - n)) ** 2.0)
                        continue
                    homogeneity[rgbChannel] += prob / ((1 + abs(m - n)) ** 2.0)

            if abs(entropy[rgbChannel]) < 0.0000001: entropy[rgbChannel] = 0.0
            # print("\nTexture attributes: ")
            # print("    entropy: %f" %  entropy[rgbChannel])
            # print("    contrast: %f" % contrast[rgbChannel])
            # print("    homogeneity: %f" % homogeneity[rgbChannel])
            # print("    energy: %f" %  energy[rgbChannel])
            # print("    dissimilarity: %f" % dissimilarity[rgbChannel])
            # print("    correlation: %f" % correlation[rgbChannel])

    return [np.average(np.array(entropy)), np.average(np.array(energy)), np.average(np.array(contrast)),
            np.average(np.array(homogeneity)), np.average(np.array(dissimilarity)), np.average(np.array(correlation))]
