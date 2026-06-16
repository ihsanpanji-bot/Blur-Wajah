import cv2
import numpy as np


img_ref = cv2.imread('foto3.png', cv2.IMREAD_GRAYSCALE)
if img_ref is None:
    print("Error: foto.jpg ga ketemu!")
    exit()

#Buka video input
cap = cv2.VideoCapture('video3.mp4')
if not cap.isOpened():
    print("Error: Video video.mp4 ga bisa dibuka!")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

#VideoWriter nyimpen hasil
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_hybrid_blur.mp4', fourcc, fps, (width, height))


#SIFT & FLANN Matcher
sift = cv2.SIFT_create()
kp_ref, des_ref = sift.detectAndCompute(img_ref, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

#Parameter Shi-Tomasi
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

#Parameter Lucas-Kanade
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


mode = "DETECTION"
MIN_MATCH_COUNT = 10
MIN_TRACK_POINTS = 5

old_gray = None
p0 = None

print("Proses dimulai! Tungguin ya, kalau videonya panjang mungkin butuh waktu...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break # Video abis
        
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if mode == "DETECTION":
        #NYARI WAJAH PAKE SIFT
        kp_frame, des_frame = sift.detectAndCompute(curr_gray, None)
        
        if des_frame is not None and len(des_frame) > 2:
            matches = flann.knnMatch(des_ref, des_frame, k=2)
            
            # Lowe's Ratio Test
            good_matches = []
            for m_match in matches:
                if len(m_match) == 2:
                    m, n = m_match
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) > MIN_MATCH_COUNT:
                #Wajah Ketemu! Cari koordinatnya pake Homography
                src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                M, mask_homography = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if M is not None:
                    h, w = img_ref.shape
                    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)
                    
                    #Bikin mask buat nyari titik Shi-Tomasi di area wajah aja
                    mask = np.zeros_like(curr_gray)
                    cv2.fillPoly(mask, [np.int32(dst)], 255)
                    
                    p0 = cv2.goodFeaturesToTrack(curr_gray, mask=mask, **feature_params)
                    
                    if p0 is not None:
                        mode = "TRACKING"
                        old_gray = curr_gray.copy()
            else:
                #Wajah ga ketemu, biarin aja frame-nya original
                pass
                
    elif mode == "TRACKING":
        #NGIKUTIN WAJAH PAKE LUCAS-KANADE
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, curr_gray, p0, None, **lk_params)
        
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            
            if len(good_new) < MIN_TRACK_POINTS:
                mode = "DETECTION"
            else:
                x_coords = [int(p[0]) for p in good_new]
                y_coords = [int(p[1]) for p in good_new]
                
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                pad = 30
                x_min = max(0, x_min - pad)
                y_min = max(0, y_min - pad)
                x_max = min(width, x_max + pad)
                y_max = min(height, y_max + pad)
                
                # Kasih Gaussian Blur
                face_roi = frame[y_min:y_max, x_min:x_max]
                if face_roi.size > 0:
                    blurred_face = cv2.GaussianBlur(face_roi, (51, 51), 0)
                    frame[y_min:y_max, x_min:x_max] = blurred_face
                
                old_gray = curr_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)
        else:
            mode = "DETECTION"

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print("Selesai! Coba cek file output_hybrid_blur.mp4")