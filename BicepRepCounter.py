from MotionDetection import load_model, movenet, draw_prediction_on_image
import cv2
import tensorflow as tf

class BicepRepCounter:
    def __init__(self):
        self.is_wrist_above_elbow = False
        self.rep_count = 0

    def count_bicep_reps(self, keypoints_with_scores, threshold=0.4):
        wrist = keypoints_with_scores[0, 0, 10, :3]  # Keypoint 10 (right wrist)
        elbow = keypoints_with_scores[0, 0, 8, :3]  # Keypoint 8 (right elbow)

        # Check if keypoints are detected with enough confidence
        if wrist[2] > threshold and elbow[2] > threshold:
            # Check if wrist is above elbow
            if wrist[1] < elbow[1]:
                self.is_wrist_above_elbow = True
            # When the wrist goes down past the elbow and was above before, count the rep
            elif self.is_wrist_above_elbow:
                self.rep_count += 1
                self.is_wrist_above_elbow = False

        return self.rep_count

def main():
    module, input_size = load_model()
    cap = cv2.VideoCapture(0)

    # Initialize the rep counter
    rep_counter = BicepRepCounter()

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
            keypoints_with_scores = movenet(input_image, module)

            # Count bicep reps
            rep_count = rep_counter.count_bicep_reps(keypoints_with_scores)

            # Draw predictions on the image
            output_overlay = draw_prediction_on_image(frame_rgb, keypoints_with_scores)

            # Convert the image back to BGR for displaying
            output_overlay = cv2.cvtColor(output_overlay, cv2.COLOR_RGB2BGR)

            # Display the rep count on the frame
            cv2.putText(output_overlay, f'Reps: {rep_count}', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the results
            cv2.imshow('Bicep Rep Counter', output_overlay)

            # Break loop with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
