import cv2
import numpy as np
import glob
import sys

"""
This code is an assignment for the subject Computer Vision from the degree Robotics Engineering at UC3M.
Authors: Ignacio González Meléndez (100522976) & Lucas Beltrán Rúa (100523114)
Professor: Abdulla Al Kaff
/images folder has some images to try with this code
Images must be given in order, thus i1 is the first and ix (with x <= 8) is the last one.
This last statement is due to computational constraints in order to avoid large convergence times.
"""

def trim_black_borders(image): # Function to trim black borders of the panorama

    if image is None: return None
    
    # Convert to grayscale and threshold to find non-black areas
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    # Find all non-zero points (actual image content)
    coords = cv2.findNonZero(thresh)
    if coords is not None: # Get the tightest bounding box around the content
        x, y, w, h = cv2.boundingRect(coords)
        return image[y:y+h, x:x+w]
    return image

def get_homography(img1, img2): # Detects SIFT main features and computes Homography using RANSAC
    sift = cv2.SIFT_create() # SIFT is selected over ORB because it is more robust to the scale and rotation changes
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    bf = cv2.BFMatcher()    # Brute-Force Matcher for feature matching
    matches = bf.knnMatch(des2, des1, k=2)
    
    good = [m for m, n in matches if m.distance < 0.7 * n.distance]
    # Lowe's Ratio Test to only keep distinct, high-quality matches
    # by comparing the distance of the best match to the second-best
    if len(good) > 10:
        src_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)    # Use RANSAC to compute the homography matrix
        return H
    return None

def stitch_pair(base_img, next_img): # Create a canvas and blend images
    H = get_homography(base_img, next_img)
    if H is None: return base_img
    
    h1, w1 = base_img.shape[:2]
    h2, w2 = next_img.shape[:2]
    
    # Calculate corners
    corners_next = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners_next, H)
    corners_base = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    all_corners = np.concatenate((corners_base, transformed_corners), axis=0)
    # Determine canvas size by finding the min/max coordinates 
    # of both the base image and the warped image corners
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    # Translation to keep result in the positive coordinate frame
    translation = [-x_min, -y_min]
    H_trans = np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])
    
    # Warp image to align with the previous one
    warped = cv2.warpPerspective(next_img, H_trans.dot(H), (x_max - x_min, y_max - y_min))
    
    # Blend current panorama with the new warped image
    roi = warped[translation[1]:h1 + translation[1], translation[0]:w1 + translation[0]]
    base_gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(base_gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    
    bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    fg = cv2.bitwise_and(base_img, base_img, mask=mask)
    warped[translation[1]:h1 + translation[1], translation[0]:w1 + translation[0]] = cv2.add(bg, fg)
    
    # Trim borders after each stitch to keep the canvas clean
    return trim_black_borders(warped)

def main():
    # Load input images from the folder
    paths = sorted(glob.glob("images/*.jpg")) + sorted(glob.glob("images/*.png"))
    
    # Check if folder has at leats 2 images and a maximum of 8
    num_images = len(paths)
    if num_images < 2 or num_images > 8:
        print(f"Error: 2 to 8 images required. Found {num_images} images.")
        sys.exit()

    images = []
    for i, p in enumerate(paths):
        img = cv2.imread(p)
        if img is not None:
            # Handle various resolutions by resizing to standard width
            h, w = img.shape[:2]
            img_resized = cv2.resize(img, (500, int(h * (500/w))))
            images.append(img_resized)
            # Display original input images
            cv2.imshow(f"Input Image {i+1}", img_resized)

    # Use Center-Out logic for better stability and a smoother panorama
    mid = len(images) // 2
    panorama = images[mid]

    for i in range(mid - 1, -1, -1):    # Stitching images to one side
        print(f"Stitching image {i+1}...")
        panorama = stitch_pair(panorama, images[i])

    for i in range(mid + 1, len(images)):   # Stitching images to the other side
        print(f"Stitching image {i+1}...")
        panorama = stitch_pair(panorama, images[i])

    # Show final panorama in a resizable window
    cv2.namedWindow("Final Panorama", cv2.WINDOW_NORMAL)
    cv2.imshow("Final Panorama", panorama)
    
    print("Process complete. Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()