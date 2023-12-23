import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
import imageio
from IPython.display import HTML, display

# Load the MoveNet model from TensorFlow Hub
model_name = "movenet_lightning"  # Choose "movenet_lightning" or "movenet_thunder"
if "movenet_lightning" in model_name:
    module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    input_size = 192
elif "movenet_thunder" in model_name:
    module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
    input_size = 256
else:
    raise ValueError("Unsupported model name: %s" % model_name)


def movenet(input_image):
    model = module.signatures['serving_default']
    input_image = tf.cast(input_image, dtype=tf.int32)
    outputs = model(input_image)
    keypoints_with_scores = outputs['output_0'].numpy()
    return keypoints_with_scores


def draw_keypoints(image, keypoints, confidence_threshold):
    height, width, _ = image.shape
    for ki in range(keypoints.shape[0]):
        y, x, c = keypoints[ki]
        x_scaled, y_scaled = int(x * width), int(y * height)



        if c > confidence_threshold:
            print(f"Drawing keypoint {ki} at ({x}, {y}) with confidence {c}")
            # Draw the keypoint
            cv2.circle(image, (x_scaled, y_scaled), 4, (0, 0, 255), -1)  # Red color for keypoints

            # Display the keypoint index and coordinates
            text = f"{ki}: ({x_scaled}, {y_scaled})"
            cv2.putText(image, text, (x_scaled + 10, y_scaled - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)




def draw_skeleton(image, keypoints, edges, confidence_threshold):
    height, width, _ = image.shape
    keypoints = keypoints * [height, width, 1]  # Scale keypoints to match image dimensions

    for edge in edges:
        p1, p2 = edge
        y1, x1, c1 = keypoints[p1]
        y2, x2, c2 = keypoints[p2]
        if c1 > confidence_threshold and c2 > confidence_threshold:
            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green color, thicker line

    return image

def draw_prediction_on_image(image, keypoints_with_scores, confidence_threshold=0.4):
    # Define a list of edges to connect the keypoints
    edges = [(0, 1), (0, 2), (1, 3), (2, 4),
             (0, 5), (0, 6), (5, 7), (7, 9),
             (6, 8), (8, 10), (5, 6), (5, 11),
             (6, 12), (11, 12), (11, 13), (13, 15),
             (12, 14), (14, 16)]

    # Extract keypoints
    keypoints = keypoints_with_scores[0, 0, :, :3]

    # Draw keypoints and skeleton
    draw_keypoints(image, keypoints, confidence_threshold)
    draw_skeleton(image, keypoints, edges, confidence_threshold)

    return image



# Webcam capture
cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the image from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize and pad the image to fit the model input
        input_image = tf.expand_dims(frame_rgb, axis=0)
        input_image = tf.image.resize_with_pad(input_image, input_size, input_size)

        # Run model inference
        keypoints_with_scores = movenet(input_image)

        # for idx, (y, x, c) in enumerate(keypoints_with_scores[0, 0, :, :3]):
        #     print(f"Keypoint {idx}: (x: {x:.2f}, y: {y:.2f}), Confidence: {c:.2f}")

        # Draw predictions on the image
        output_overlay = draw_prediction_on_image(frame_rgb, keypoints_with_scores)

        # Convert the image back to BGR for displaying
        output_overlay = cv2.cvtColor(output_overlay, cv2.COLOR_RGB2BGR)

        # Display the results
        cv2.imshow('MoveNet Pose Estimation', output_overlay)

        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
