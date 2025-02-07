import cv2
import numpy as np
import matplotlib.pyplot as plt


# Region of Interest (ROI) coordinates
ROI_TOP = 210
ROI_BOTTOM = 510
ROI_LEFT = 490
ROI_RIGHT = 790


# Smoothing settings
smoothed_midpoints = None
SMOOTHING_ALPHA = 0.05
NUM_POINTS = 60  # Fixed number of points for smoothing consistency




kalman_filters = None # Kalman filters for each midpoint




def compute_midline(curve1, curve2): # Align curve lengths and calculate midpoints
   min_len = min(len(curve1), len(curve2))
   curve1, curve2 = curve1[:min_len], curve2[:min_len]
   midpoints = (curve1 + curve2) // 2
   return midpoints




def resample_midpoints(midpoints, num_points=NUM_POINTS):
   midpoints = midpoints.astype(np.float32) # Resample midpoints to maintain a fixed number of points
   if len(midpoints) == num_points:
       return midpoints
   indices = np.linspace(0, len(midpoints) - 1, num_points)
   resampled = np.empty((num_points, 2), dtype=np.float32)
   for i, idx in enumerate(indices):
       low = int(np.floor(idx))
       high = min(int(np.ceil(idx)), len(midpoints) - 1)
       weight = idx - low
       resampled[i] = (1 - weight) * midpoints[low] + weight * midpoints[high]
   return resampled




def create_kalman_filter():
   kalman = cv2.KalmanFilter(4, 2) # Setup a Kalman filter for tracking 2D points
   kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                        [0, 1, 0, 0]], np.float32)
   kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                       [0, 1, 0, 1],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]], np.float32)
   kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-3
   kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-2
   kalman.errorCovPost = np.eye(4, dtype=np.float32)
   return kalman




def kalman_smooth_midpoints(midpoints):
   global kalman_filters # Initialize or update Kalman filters
   if kalman_filters is None or len(kalman_filters) != len(midpoints):
       kalman_filters = [create_kalman_filter() for _ in range(len(midpoints))]
       for i, point in enumerate(midpoints):
           kalman_filters[i].statePre = np.array([[point[0]], [point[1]], [0], [0]], np.float32)
           kalman_filters[i].statePost = np.array([[point[0]], [point[1]], [0], [0]], np.float32)
   smoothed = []
   for i, point in enumerate(midpoints):
       measurement = np.array([[np.float32(point[0])], [np.float32(point[1])]])
       kalman_filters[i].predict()
       estimated = kalman_filters[i].correct(measurement)
       smoothed.append([estimated[0, 0], estimated[1, 0]])
   return np.array(smoothed, dtype=np.float32)




def detect_curves(frame):
   # Process the frame to extract ROI and detect edges
   roi = frame[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT]
   gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
   blurred = cv2.GaussianBlur(gray, (9, 9), 0)
   edges = cv2.Canny(blurred, 50, 150)


   # Apply morphological transformation
   kernel = np.ones((3, 3), np.uint8)
   edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


   # Find contours
   contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   cv2.drawContours(roi, contours, -1, (0, 0, 255), 2)


   # Select the two largest contours
   valid_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
   if len(valid_contours) < 2:
       return None, None, None, edges


   # Approximate contours and compute midline
   curve1 = cv2.approxPolyDP(valid_contours[0], epsilon=5, closed=False).reshape(-1, 2)
   curve2 = cv2.approxPolyDP(valid_contours[1], epsilon=5, closed=False).reshape(-1, 2)
   midpoints = compute_midline(curve1, curve2)


   curve1 += [ROI_LEFT, ROI_TOP]
   curve2 += [ROI_LEFT, ROI_TOP]
   midpoints += [ROI_LEFT, ROI_TOP]


   return curve1, curve2, midpoints, edges




def draw_centerline_on_frame(frame, curve1, curve2, midpoints, edges):
   # Visualize curves and centerline on the frame
   if curve1 is not None:
       for point in curve1:
           cv2.circle(frame, tuple(map(int, point)), 1, (0, 0, 255), -1)
   if curve2 is not None:
       for point in curve2:
           cv2.circle(frame, tuple(map(int, point)), 1, (0, 255, 0), -1)
   if midpoints is not None:
       for i in range(len(midpoints) - 1):
           pt1 = tuple(map(int, midpoints[i]))
           pt2 = tuple(map(int, midpoints[i + 1]))
           cv2.line(frame, pt1, pt2, (255, 0, 0), 2)


   if edges is not None:
       edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
       frame[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT] = cv2.addWeighted(
           frame[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT], 0.7, edges_colored, 0.3, 0
       )
   return frame




def main():
   # Capture and process video stream
   global smoothed_midpoints
   cap = cv2.VideoCapture(0)


   while True:
       ret, frame = cap.read()
       if not ret:
           break


       curve1, curve2, midpoints, edges = detect_curves(frame)


       if midpoints is not None:
           current_midpoints = resample_midpoints(midpoints.astype(np.float32))


           if smoothed_midpoints is None or len(smoothed_midpoints) != len(current_midpoints):
               smoothed_midpoints = current_midpoints
           else:
               smoothed_midpoints = (SMOOTHING_ALPHA * current_midpoints +
                                     (1 - SMOOTHING_ALPHA) * smoothed_midpoints)


           smoothed_midpoints = kalman_smooth_midpoints(smoothed_midpoints)


           frame = draw_centerline_on_frame(frame, curve1, curve2, smoothed_midpoints.astype(np.int32), edges)


       cv2.rectangle(frame, (ROI_LEFT, ROI_TOP), (ROI_RIGHT, ROI_BOTTOM), (0, 255, 255), 2)


       cv2.imshow("Live Centerline Detection", frame)


       if cv2.waitKey(1) & 0xFF == ord('q'):
           break


   cap.release()
   cv2.destroyAllWindows()




if __name__ == "__main__":
   main()

