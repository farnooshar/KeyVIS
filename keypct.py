from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import cv2
#import fire
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from PIL import Image
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm

import extract_utils as utils

import numpy as np
import matplotlib.pyplot as plt
import cv2

import sys
import warnings

import torch.nn as nn

import cv2
import numpy as np
from scipy.signal import convolve2d

class KEYPCT(nn.Module):
    def __init__(self,kernel_size=8):
        super(KEYPCT, self).__init__()
        
        params = [ (0, 0), (0, 1), (0, 2),(0,3), (1, 0), (1, 1), (1, 2),(1,3), (2, 0), (2, 1), (2,2),(2,3), (3, 0), (3, 1), (3,2),(3,3)]
        
        params = params[4:12]
        
        self.h_stars=[];
        self.kernel_size = kernel_size
        for i, (n, l) in enumerate(params):
            self.h_stars.append(self.get_h_star(n, l, self.kernel_size))

        

        
    def gaussian_curve_samples(self,num_samples=16, mean=0, std=1, x_min=-3, x_max=3):
        """
        Generate `num_samples` from a Gaussian curve and return them as a normalized probability array.

        Parameters:
        - num_samples: Number of samples to take from the Gaussian curve (default is 16)
        - mean: Mean of the Gaussian distribution (default is 0)
        - std: Standard deviation of the Gaussian distribution (default is 1)
        - x_min: Minimum x-value for sampling (default is -3, approximately 3 standard deviations left of the mean)
        - x_max: Maximum x-value for sampling (default is 3, approximately 3 standard deviations right of the mean)

        Returns:
        - A normalized array of probabilities summing to 1.
        """
        # Define a Gaussian (normal) distribution function
        def gaussian(x, mu, sigma):
            return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

        # Generate `num_samples` evenly spaced points between `x_min` and `x_max`
        x_values = np.linspace(x_min, x_max, num_samples)

        # Evaluate the Gaussian function at each of these points
        gaussian_values = gaussian(x_values, mean, std)

        # Normalize the values so they sum to 1 (to represent a probability distribution)
        probability_array = gaussian_values / gaussian_values.sum()

        return probability_array

    def polar_cosine_transform(image, n, kernel):
        Z = omega(n) * np.sum(kernel * image)
        Amp = np.abs(Z)
        Phi = np.angle(Z) * 180 / np.pi
        return Amp, Phi

    def omega(n):
        # Define omega based on your specific requirements (this can vary depending on the application).
        # For now, I assume it's a simple function that returns 1.
        return 1

    def get_h_star(self,n, l, size):
        x = np.linspace(1, size, size)
        y = np.linspace(1, size, size)
        X, Y = np.meshgrid(x, y)

        # Center
        c = (1 + size) / 2
        X = (X - c) / (c - 1)
        Y = (Y - c) / (c - 1)

        # Polar coordinates
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)

        # Create mask for R <= 1 (unit circle)
        mask = (R <= 1)

        # Compute the polar cosine transform base
        result = mask * np.cos(np.pi * n * R**2) * np.exp(-1j * l * Theta)
        return result
    
    def apply_pct(self,h_stars,patch,kernel_real=8):

        convs=[]
        for h_star in h_stars:
            kernel_real = np.real(h_star)
            convs.append(np.mean(convolve2d(patch, kernel_real, mode='same', boundary='wrap')))

        return np.asarray(convs)

    


    def get_object_motion(self,f1, f2, f3, keypoints,maskclip):

        f1 = self.prepare_image(f1)
        f2 = self.prepare_image(f2)
        f3 = self.prepare_image(f3)

        A,B,C,D,E,F = keypoints
        Nins,T,H,W = maskclip.shape

        distance_forward_1  = [];
        distance_forward_2  = [];
        distance_backward = [];

        #if len(A)!=0:
        distance_forward_1,keysmask_1 = self.core_object(A,B,Nins,maskclip,0,1)

        #if len(C)!=0:
        distance_forward_2,keysmask_2 = self.core_object(C,D,Nins,maskclip,1,2)

        #if len(E)!=0:
        distance_backward,keysmask_3  = self.core_object(E,F,Nins,maskclip,2,0)


        dfw_1=1.0;dfw_2=1.0;dbw=1.0
        if len(distance_forward_1)!=0:
            dfw_1 = np.mean(distance_forward_1);

        if len(distance_forward_2)!=0:
            dfw_2 = np.mean(distance_forward_2);

        if len(distance_backward)!=0:
            dbw = np.mean(distance_backward);


        return dfw_1,dfw_2,dbw,[keysmask_1,keysmask_2,keysmask_3]

    def select_pixels_from_mask(self,mask, num_pixels=8):
        """
        Select `num_pixels` random pixels from areas where the mask value is 1.

        Parameters:
        - mask: 2D numpy array (binary mask with 0 and 1 values)
        - num_pixels: Number of pixels to select (default is 8)

        Returns:
        - A list of numpy arrays in the format [array([[x, y]], dtype=float32), ...]
        """
        # Find coordinates where mask == 1
        coordinates = np.column_stack(np.where(mask == 1))

        # If less than num_pixels available, adjust the number of pixels to select
        if len(coordinates) < num_pixels:
            coordinates = np.column_stack(np.where(mask != 1))

            #raise ValueError(f"Not enough pixels with value 1 in the mask. Found {len(coordinates)}, required {num_pixels}.")

        # Select random pixel coordinates
        random_indices = np.random.choice(len(coordinates), size=num_pixels, replace=False)
        selected_pixels = coordinates[random_indices]

        # Format the selected pixels
        formatted_pixels = [np.array([[float(y), float(x)]], dtype=np.float32) for x, y in selected_pixels]

        return formatted_pixels

    def core_object(self,keysrc,keydes,Nins,maskclip,id1,id2):

        distance_list=[];keysmask=[]
        for Nin in range(0,Nins):

            mask1 = maskclip[Nin,id1]
            mask2 = maskclip[Nin,id2]

            # Concatenate both images horizontally for visualization
            #img_combined = np.hstack((img1, img2))

            # Draw lines between the keypoints
            srcmask=[];dstmask=[]
            for (src, dst) in zip(keysrc, keydes):
                try:
                    if mask1[int(src[0][1]),int(src[0][0])]==0:
                        continue
                except:
                    continue

                try:
                    if mask2[int(dst[0][1]),int(dst[0][0])]==0:
                        continue
                except:
                    continue

                #pt1 = tuple(np.int32(src[0]))
                #pt2 = tuple(np.int32(dst[0] + np.array([img1.shape[1], 0])))  # Adjust x-coordinate for the combined image
                #cv2.line(img_combined, pt1, pt2, color=(0, 255, 0), thickness=1)
                srcmask.append(src);dstmask.append(dst);

            distances = 1;
            if len(srcmask)!=0:
                distances = np.linalg.norm(np.asarray(srcmask) - np.asarray(dstmask), axis=1)/ np.sqrt(self.H**2+self.W**2)

            if len(srcmask)<=7:
                #print('---------------------XXX ',len(srcmask))
                ln = 8 - len(srcmask)
                srcmask = self.select_pixels_from_mask(mask1,ln)
                dstmask = self.select_pixels_from_mask(mask2,ln)

                #for (src, dst) in zip(srcmask, dstmask):
                    #pt1 = tuple(np.int32(src[0]))
                    #pt2 = tuple(np.int32(dst[0] + np.array([img1.shape[1], 0])))  # Adjust x-coordinate for the combined image
                    #cv2.line(img_combined, pt1, pt2, color=(0, 255, 0), thickness=1)


                #plt.imshow(np.hstack((mask1, mask2)))
                #plt.figure()

            keysmask.append([srcmask,dstmask])

            mean_distance = np.mean(distances)
            distance_list.append(mean_distance)
   
        return distance_list,keysmask



    def prepare_image(self,img):
        """
        Prepare image by normalizing and converting it to uint8.

        Parameters:
        img: Input image (could be float32 or other format).

        Returns:
        uint8_image: Image converted to uint8 format, suitable for feature detection.
        """
        # Normalize the image to the range 0-255 if it contains values outside that range
        img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

        # Convert to uint8 format
        img_uint8 = img_normalized.astype(np.uint8)

        return img_uint8


    def find_homography(self,img1, img2):
        """
        Find the homography matrix to align img1 to img2.

        Parameters:
        img1, img2: Input images for which homography is to be found.

        Returns:
        homography_matrix: The homography matrix to warp img1 to img2.
        """
        # Prepare images by converting them to uint8
        img1 = self.prepare_image(img1)
        img2 = self.prepare_image(img2)

        # Initialize SIFT detector
        sift = cv2.SIFT_create()

        # Detect keypoints and descriptors
        keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

        # Match features using FLANN-based matcher
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        # Store only good matches (using Lowe's ratio test)
        #print(len(matches))
        good_matches = [];
        for m, n in matches:
            if m.distance < 0.85 * n.distance:
                good_matches.append(m)

        #print(len(good_matches),len(matches),len(good_matches)/len(matches))
        # Extract location of good matches
        if len(good_matches) > 10:
            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Compute homography matrix using RANSAC
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            return H,src_pts, dst_pts
        else:
            pass
            #raise ValueError("Not enough matches found to compute homography.")





    def get_camera_motion(self,f1, f2, f3):

        A=[];B=[];C=[];D=[];E=[];F=[];
        motion_backward = 0.0; motion_forward_1 = 0.0; motion_forward_2 = 0.0;
        try:
            H12,A,B = self.find_homography(f1, f2)
        except:
            motion_forward_1 = 1.0;

        try:
            H23,C,D = self.find_homography(f2, f3)
        except:
            motion_forward_2 = 1.0

        try:
            H31,E,F = self.find_homography(f3, f1)
        except:
            motion_backward = 1.0

        # Get dimensions of the images
        h, w = f2.shape[:2]
        f1 = np.zeros_like(f1)+1
        f2 = np.zeros_like(f2)+1
        f3 = np.zeros_like(f3)+1

        # Warp f1 and f3 to align them with f2 using the respective homography matrices

        if motion_backward<1:
            aligned_f3 = cv2.warpPerspective(f3, H31, (w, h))
            motion_backward = len(np.where(aligned_f3[:,:,0]==0)[0])/(h*w)

        if motion_forward_1<1:
            aligned_f1 = cv2.warpPerspective(f1, H12, (w, h))
            motion_forward_1 = len(np.where(aligned_f1[:,:,0]==0)[0])/(h*w)

        if motion_forward_2<1:
            aligned_f2 = cv2.warpPerspective(f2, H23, (w, h))
            motion_forward_2 = len(np.where(aligned_f2[:,:,0]==0)[0])/(h*w)

        return motion_forward_1,motion_forward_2,motion_backward,[A,B,C,D,E,F]


        
    def core_pct_loss(self,feature,Nins,weight,ikeysmask,id1,id2,rh,rw,HF,WF):
    
        loss_list=[]

        for Nin in range(0,Nins):

            loss_mle=0;
            A = ikeysmask[Nin]
            keysrc = A[0]; keydes = A[1]

            feats1 = feature[Nin,id1]
            feats2 = feature[Nin,id2]

            qq=0;

            for (src, dst) in zip(keysrc, keydes):
                #print(src,dst)

                a = int(src[0][1]/rh)
                b = int(src[0][0]/rw)

                u1 = a-8; u2 = a+8; v1 = b-8; v2 = b+8;

                if u1<0:
                    u1=0;

                if v1<0:
                    v1=0;

                if u2>(HF-1):
                    u2=HF-1;

                if v2>(WF-1):
                    v2=WF-1;

                patch1 = feats1[u1:u2,v1:v2]
                patch2 = feats2[u1:u2,v1:v2]

                convs1 = torch.tensor(self.apply_pct(self.h_stars,patch1,kernel_real=8)*weight)
                convs2 = torch.tensor(self.apply_pct(self.h_stars,patch2,kernel_real=8)*weight)

                loss_mle = loss_mle +  F.l1_loss(convs1, convs2).requires_grad_()
                qq+=1;
                
                if qq>=32:
                    break

            loss_list.append(loss_mle/qq)

        return loss_list 
        
    def get_patch_pct_loss(self,feature,Nins,HF,WF,rh,rw,camera_motion_forward_1,camera_motion_forward_2,\
                           camera_motion_backward,object_motion_forward_1,object_motion_forward_2,object_motion_backward,keysmask):

        loss_mle = 0

        keysmask_1,keysmask_2,keysmask_3 = keysmask

        mean1   = (object_motion_forward_1 + camera_motion_forward_1)/2
        weight1 = self.gaussian_curve_samples(num_samples=8, mean=mean1)

        mean2   = (object_motion_forward_2 + camera_motion_forward_2)/2
        weight2 = self.gaussian_curve_samples(num_samples=8, mean=mean2)

        mean3   = (object_motion_backward  + camera_motion_backward)/2
        weight3 = self.gaussian_curve_samples(num_samples=8, mean=mean3)


        loss_list_forward_1 = self.core_pct_loss(feature,Nins,weight1,keysmask_1,0,1,rh,rw,HF,WF)
        loss_list_forward_2 = self.core_pct_loss(feature,Nins,weight2,keysmask_2,1,2,rh,rw,HF,WF)
        loss_list_backward =  self.core_pct_loss(feature,Nins,weight3,keysmask_3,2,0,rh,rw,HF,WF)

        return loss_list_forward_1,loss_list_forward_2,loss_list_backward


    import torch

    def replace_inf_nan(self,tensor):
        
        if torch.isinf(tensor).any() or torch.isnan(tensor).any():
            tensor = torch.where(
                torch.isinf(tensor) | torch.isnan(tensor),
                torch.tensor(1.0, dtype=tensor.dtype, device=tensor.device),
                tensor
            )
        return tensor

    
    def forward(self,feats,images,targets):
        
        N,C,self.H,self.W = images.size()
        I,T,HF,WF = feats.size()
        images = images.detach().cpu().numpy()
        feats  =  feats.detach().cpu().numpy()

        
        rh = self.H/HF; rw = self.W/WF;
                
        B=0;loss_forward_1=[];loss_forward_2=[];loss_backward=[];
        for n in range(0,N,3):
            
            f1 = images[n].transpose(1,2,0);f2 = images[n+1].transpose(1,2,0);f3 = images[n+2].transpose(1,2,0)
            maskclip =  targets[B]['masks'].detach().cpu().numpy(); Nins = maskclip.shape[0]


            camera_motion_forward_1,camera_motion_forward_2,camera_motion_backward,keypoints = self.get_camera_motion(f1, f2, f3)
            object_motion_forward_1,object_motion_forward_2,object_motion_backward,keysmask  = self.get_object_motion(f1, f2, f3, keypoints, maskclip)
            loss_list_forward_1, loss_list_forward_2, loss_list_backward = self.get_patch_pct_loss(feats,Nins,HF,WF,rh,rw,camera_motion_forward_1,\
                                 camera_motion_forward_2,camera_motion_backward,object_motion_forward_1,object_motion_forward_2,object_motion_backward,keysmask)

            loss_forward_1 =  loss_forward_1  + loss_list_forward_1
            loss_forward_2 =  loss_forward_2  + loss_list_forward_2
            loss_backward  =  loss_backward   + loss_list_backward
            B+=1;

    
        loss_forward_1 = torch.stack(loss_forward_1).view(I, 1, 1, 1)
        loss_forward_2 = torch.stack(loss_forward_2).view(I, 1, 1, 1)
        loss_backward  = torch.stack(loss_backward).view(I, 1, 1, 1)
        
        
        loss_forward_1 = self.replace_inf_nan(loss_forward_1)
        loss_forward_2 = self.replace_inf_nan(loss_forward_2)
        loss_backward  = self.replace_inf_nan(loss_backward)

        
        loss_total = torch.stack([loss_forward_1,loss_forward_2,loss_backward])

        
        return loss_total.cuda();

