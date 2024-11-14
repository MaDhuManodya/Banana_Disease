from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

# Load your YOLOv8 model
model = YOLO('puwaluModel/puwalu.pt')  # Path to your primary model

# Path to the image for both stages
image_path = 'IMG_2473.jpg'

# Run inference on the image for initial detection
results = model.predict(image_path)

# Flag to track if "puwalu" is detected
puwalu_detected = False

# Process initial predictions
for result in results:
    class_ids = result.boxes.cls.numpy()
    class_names = [model.names[int(class_id)] for class_id in class_ids]

    # Check if "puwalu" is detected
    for class_name in class_names:
        if class_name.lower() == "puwalu-banana":
            puwalu_detected = True
            print(f"Detected class: {class_name} ❤️")
            break

# If "puwalu" was detected, perform additional detection using the same image
if puwalu_detected:
    print("Running additional detection on the same image...")

    # Load secondary model for further analysis
    secondary_model = YOLO('diseaseModel\disease.pt')  # Path to your secondary model

    # Run inference on the same image again
    secondary_results = secondary_model.predict(image_path)

    # Flag for any detections in secondary analysis
    detections_found = False

    # Process secondary predictions
    for result in secondary_results:
        class_ids = result.boxes.cls.numpy()
        class_names = [secondary_model.names[int(class_id)] for class_id in class_ids]
        confidences = result.boxes.conf.numpy()

        # Display detections
        for class_name, conf in zip(class_names, confidences):
            detections_found = True
            print(f"Secondary Detection - Class: {class_name}, Confidence: {conf:.2f}")

        # Plot annotated image if detections found
        if detections_found:
            annotated_image = result.plot()

    # Display the annotated image
    if detections_found:
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        plt.imshow(annotated_image_rgb)
        plt.axis('off')
        plt.show()
    else:
        print("No detections found in the secondary analysis.")
else:
    print("No 'puwalu' detected in the initial analysis.")
