import cv2
from ultralytics import YOLO
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Function to open a file dialog to select an image
def select_image():
    Tk().withdraw()  # Hide the root Tkinter window
    file_path = askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    return file_path

# Load the YOLOv8 model
model = YOLO('runs/detect/roof/weights/best.pt')  # Replace with your model path

# Select an image file
image_path = select_image()

if not image_path:
    print("No file selected. Exiting...")
else:
    # Read the image
    image = cv2.imread(image_path)

    # Run YOLOv8 inference on the image
    results = model(image, conf=0.4)

    # Annotate the image
    annotated_image = image.copy()

    for result in results:
        if result.boxes:
            for box in result.boxes:
                # Extract class ID and name
                class_id = int(box.cls)
                object_name = model.names[class_id]

                # Extract bounding box coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Calculate bounding box area
                width = x2 - x1
                height = y2 - y1
                area = width * height

                # Draw the bounding box and label
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{object_name}: {area} px²"
                cv2.putText(
                    annotated_image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )

                print(f"Detected: {object_name}, Area: {area} px²")

    # Display the annotated image
    cv2.imshow("YOLOv8 Prediction", annotated_image)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()
