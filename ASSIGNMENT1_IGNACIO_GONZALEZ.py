import cv2
import numpy as np
import time

    ###     EVERY COMMENT IN THIS CODE HAS BEEN DONE BY ME: IGNACIO GONZÁLEZ MELÉNDEZ   ###
    ###     THIS CODE IS AN ASSIGNMENT FOR THE SUBJECT COMPUTER VISION FROM ROBOTICS ENGINEERING (UC3M)  ###

def nothing(x):     #Empty function used as trackbars argument
    pass

def get_histogram_img(frame, channels_count):       #Function to compute and render live histograms for the dashboard
    """Computes and draws a histogram image. Always returns a 3-channel BGR image."""
    hist_h, hist_w = 150, 320 
    hist_img = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    
    if channels_count == 1:
        hist = cv2.calcHist([frame], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist, 0, hist_h, cv2.NORM_MINMAX)
        for i in range(1, 256):
            cv2.line(hist_img, (i-1, hist_h - int(hist[i-1].item())), 
                     (i, hist_h - int(hist[i].item())), (255, 255, 255), 1)
    else:
        for i, col in enumerate(colors):
            hist = cv2.calcHist([frame], [i], None, [256], [0, 256])
            cv2.normalize(hist, hist, 0, hist_h, cv2.NORM_MINMAX)
            for j in range(1, 256):
                cv2.line(hist_img, (j-1, hist_h - int(hist[j-1].item())), 
                         (j, hist_h - int(hist[j].item())), col, 1)
    return hist_img

cv2.namedWindow('UC3M CV Dashboard')        #Create a single window dashboard showing all outputs at the same time
cv2.createTrackbar('Gauss K', 'UC3M CV Dashboard', 1, 10, nothing)      #Trackbar to control Gaussian Kernel size, only accepts odd values
cv2.createTrackbar('Sobel/Canny K', 'UC3M CV Dashboard', 1, 3, nothing)     #Trackbar to control Sobel and Canny Kernel sizes
cv2.createTrackbar('Canny Low', 'UC3M CV Dashboard', 50, 255, nothing)      #Trackbar to control Canny Low Treshold
cv2.createTrackbar('Canny High', 'UC3M CV Dashboard', 150, 255, nothing)    #Trackbar to control Canny High Treshold

cap = cv2.VideoCapture(0)
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.resize(frame, (320, 240))

    #Convert frame to Grayscale and HSV color spaces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    k_gauss = cv2.getTrackbarPos('Gauss K', 'UC3M CV Dashboard') * 2 + 1        #Get k-value regarding trackback for Gaussian Filter. With conversion to odd K
    blur_gray = cv2.GaussianBlur(gray, (k_gauss, k_gauss), 0)       #Apply Gaussian filtering to the image before edge detection

    #Tracking trackbars position
    k_edge = cv2.getTrackbarPos('Sobel/Canny K', 'UC3M CV Dashboard') * 2 + 1   #Conversion to odd K
    low_th = cv2.getTrackbarPos('Canny Low', 'UC3M CV Dashboard')
    high_th = cv2.getTrackbarPos('Canny High', 'UC3M CV Dashboard')

    #Manual Sobel implementation & Defining 3x3 kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    #Applying manual kernels using filter2D
    grad_x = cv2.filter2D(blur_gray, cv2.CV_32F, sobel_x)
    grad_y = cv2.filter2D(blur_gray, cv2.CV_32F, sobel_y)

    #Compute gradient magnitude for Sobel edge map
    edge_sobel = cv2.magnitude(grad_x, grad_y)
    edge_sobel = cv2.convertScaleAbs(edge_sobel)

    edge_canny = cv2.Canny(blur_gray, low_th, high_th)      #Compute Canny edge detection

    #Compute Laplacian of Gaussian (LoG)
    edge_log = cv2.Laplacian(blur_gray, cv2.CV_64F, ksize=k_edge)
    edge_log = cv2.convertScaleAbs(edge_log)

    #Generate live histograms that will be updating during the stream
    hist_bgr = get_histogram_img(frame, 3)
    hist_gray = get_histogram_img(gray, 1)
    hist_hsv = get_histogram_img(hsv, 3)

    #Add labels to frames before concatenation
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    sobel_bgr = cv2.cvtColor(edge_sobel, cv2.COLOR_GRAY2BGR)
    canny_bgr = cv2.cvtColor(edge_canny, cv2.COLOR_GRAY2BGR)
    log_bgr = cv2.cvtColor(edge_log, cv2.COLOR_GRAY2BGR)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'Original', (10, 220), font, 0.7, (0, 255, 255), 2)
    cv2.putText(gray_bgr, 'Grayscale', (10, 220), font, 0.7, (0, 255, 255), 2)
    cv2.putText(hsv, 'HSV', (10, 220), font, 0.7, (0, 255, 255), 2)
    cv2.putText(sobel_bgr, 'Manual Sobel', (10, 220), font, 0.7, (0, 255, 0), 2)
    cv2.putText(canny_bgr, 'Canny', (10, 220), font, 0.7, (0, 255, 0), 2)
    cv2.putText(log_bgr, 'LoG', (10, 220), font, 0.7, (0, 255, 0), 2)

    #Concatenate all filter and views
    row1 = cv2.hconcat([frame, gray_bgr, hsv])
    
    row2 = cv2.hconcat([hist_bgr, hist_gray, hist_hsv])
    
    row3 = cv2.hconcat([sobel_bgr, canny_bgr, log_bgr])

    dashboard = cv2.vconcat([row1, row2, row3])

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0     #Computing FPS to be shown
    prev_time = curr_time
    cv2.putText(dashboard, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)     #Showing FPS on screen

    cv2.imshow('UC3M CV Dashboard', dashboard)
    if cv2.waitKey(1) & 0xFF == ord('q'): break     #Closing window when pressing q

cap.release()
cv2.destroyAllWindows()