import cv2
import pygame
import time

# Initialize Pygame
pygame.init()

# Path to the video file
video_path = 'news_video.mp4'

# Create a VideoCapture object
cap = cv2.VideoCapture(video_path)

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_delay = int(1000 / fps)

# Initialize Pygame mixer
pygame.mixer.init()
pygame.mixer.music.load(video_path)
pygame.mixer.music.play()

# Read and display the video frames
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Video', frame)

    # Press 'q' to exit the video
    if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()
pygame.quit()