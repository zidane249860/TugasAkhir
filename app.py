import cv2 # You might need this if you want to add cv2.destroyAllWindows() later
from ultralytics import YOLO

# It's good practice to wrap the main execution in if __name__ == '__main__':
# especially when dealing with multiprocessing (which YOLOv8 uses for data loading)
# on Windows.
if __name__ == '__main__':
    # 1. Load your trained model
    model_path = 'runs/detect/yolov8n_custom/weights/best.pt'
    model = YOLO(model_path)

    # 2. Define the source video
    video_source = 'test_video/test2.mp4' # <--- Your video path

    # 3. Run inference on the video with real-time display
    # Set show=True to display the results in a window as the video plays.
    # Set stream=True for efficient frame-by-frame processing.
    results_generator = model.predict(
        source=video_source,
        save=True,        # Still save the annotated video to a file
        conf=0.25,        # Confidence threshold
        iou=0.7,          # IoU threshold for NMS
        show=True,        # <--- CHANGE THIS TO TRUE FOR REAL-TIME DISPLAY
        stream=True,      # Process video frame by frame as a generator
        name='realtime_video_prediction_display', # A descriptive name for your output folder
        # To save detection details to text files, uncomment the next line:
        # save_txt=True,
        # save_conf=True # If you also want confidence scores in the text files
    )

    print("Starting real-time video prediction. A window should appear. Press 'q' or close the window to stop.")

    # 4. Iterate through the results for each frame to drive the processing
    # The `show=True` argument handles the actual window display internally.
    for i, result in enumerate(results_generator):
        # The code to print detected classes per frame is still here.
        # This loop also ensures that the video processing continues frame by frame.
        boxes = result.boxes # Boxes object

        detected_classes_in_frame = set() # Use a set to store unique class names per frame

        if boxes is not None:
            # Access the class names mapping from the model
            class_names = model.names # This is already initialized from `model` object

            # Iterate over each detected object's class ID
            for cls_id_tensor in boxes.cls:
                class_id = int(cls_id_tensor.item()) # Convert tensor to integer

                # Look up the class name using the model's names dictionary
                class_name = class_names[class_id]
                detected_classes_in_frame.add(class_name) # Add to set to get unique names

        print(f"Frame {i+1}: Detected classes: {list(detected_classes_in_frame)}")

    print(f"\nReal-time display stopped. Annotated video also saved to: {model.predictor.save_dir}")

    # Optional: Ensure all OpenCV windows are closed, though Ultralytics often handles this.
    cv2.destroyAllWindows()