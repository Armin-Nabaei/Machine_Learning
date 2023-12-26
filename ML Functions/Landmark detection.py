import cv2
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN

def draw_detection_box(image, box, landmarks):
    """
    Draws the detection box and landmarks on the image.

    Parameters:
    - image: The image on which to draw.
    - box: The bounding box for a face (x, y, width, height).
    - landmarks: Facial landmarks (eyes, nose, mouth).
    """
    x, y, w, h = box
    colors = {'box': (0, 255, 0), 'landmarks': (0, 0, 255)}

    # Draw rectangle around the face
    cv2.rectangle(image, (x, y), (x + w, y + h), colors['box'], 2)

    # Draw circles for each landmark
    for landmark in landmarks:
        cv2.circle(image, landmark, radius=2, color=colors['landmarks'], thickness=2)

def detect_faces(image, model):
    """
    Detects faces in an image using the specified MTCNN model.

    Parameters:
    - image: The image to process.
    - model: The MTCNN model for face detection.
    """
    boxes, _, landmarks = model.detect(image, landmarks=True)

    for box, landmark in zip(boxes, landmarks):
        # Adjust box and landmark coordinates
        box = [max(int(coord), 1) for coord in box]
        box[2] -= box[0]
        box[3] -= box[1]

        landmark_coords = [(int(point[0]), int(point[1])) for point in landmark]

        draw_detection_box(image, box, landmark_coords)

    # Display the image with detected faces
    cv2.imshow("Detected Faces", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # Initialize MTCNN model
    mtcnn_model = MTCNN(min_face_size=10, thresholds=[0.7, 0.7, 0.9], factor=0.9)

    # Load and preprocess the image
    test_image_path = '/content/test_2.jpeg'
    test_image = Image.open(test_image_path)
    test_image = asarray(test_image)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

    # Detect faces in the image
    detect_faces(test_image, mtcnn_model)

if __name__ == "__main__":
    main()

