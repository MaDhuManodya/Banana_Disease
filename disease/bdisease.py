from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

# Load your trained YOLOv8 model
model = YOLO('last.pt')  # Use forward slashes for paths if needed

# Run inference on the image
results = model.predict('agronomy-12-02215-g008.png')

# Flag to check if any detections are found
detections_found = False

# Get predictions for the detected objects
for result in results:
    # Access the detected classes (class IDs)
    class_ids = result.boxes.cls.numpy()

    # Access the names and confidence scores of the detected classes
    class_names = [model.names[class_id] for class_id in class_ids]
    confidences = result.boxes.conf.numpy()

    # Print all detections with their confidence scores
    for class_name, conf in zip(class_names, confidences):
        detections_found = True
        print(f"Detected class: {class_name}, Confidence: {conf:.2f}")

    # Plot the image with detection results if any detections exist
    if detections_found:
        annotated_image = result.plot()  # Annotate the image with bounding boxes and labels

# Display the annotated image or inform if no detections were made
if detections_found:
    # Convert BGR image (OpenCV format) to RGB (for displaying with matplotlib)
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # Display the image using matplotlib
    plt.imshow(annotated_image_rgb)
    plt.axis('off')  # Hide axis
    plt.show()
else:
    print("No objects detected in the image.")
