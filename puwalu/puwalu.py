from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

# Load your trained YOLOv8 model
model = YOLO('last.pt')

# Run inference on the image
results = model.predict('IMG_2254.jpg')

# Get the predictions for the detected objects
for result in results:
    # Access the detected classes (class IDs)
    class_ids = result.boxes.cls.numpy()

    # Access the names of the detected classes
    class_names = [model.names[class_id] for class_id in class_ids]

    # Print the detected class names and their confidence scores
    for class_name, conf in zip(class_names, result.boxes.conf.numpy()):
        print(f"Detected class: {class_name}, Confidence: {conf:.2f}")

    # Plot the image with the detection results
    annotated_image = result.plot()  # Annotate the image with bounding boxes and labels

# Convert BGR image (OpenCV format) to RGB (for displaying with matplotlib)
annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

# Display the image using matplotlib
plt.imshow(annotated_image_rgb)
plt.axis('off')  # Hide axis
plt.show()
