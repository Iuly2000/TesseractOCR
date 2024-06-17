import cv2
import numpy as np
from keras.models import load_model

# Load the trained character recognition model
model = load_model('emnist_model.h5')

# Character mapping for EMNIST Letters dataset
character_map = {
    i: chr(j) for i, j in enumerate(list(range(65, 91)), 0)
}


def segment_image(img_path, threshold=100):
    """
    Segments the image into characters using contour detection.

    Parameters:
        img_path (str): Path to the input image.
        threshold (int): Threshold value for binarization (default: 100).

    Returns:
        List of character images.
    """
    # Load image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    img = cv2.GaussianBlur(img, (1, 1), 0)

    # Adaptive thresholding
    binary_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

    # Find contours of characters
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours from left to right
    contours = sorted(contours, key=lambda contour: cv2.boundingRect(contour)[0])

    character_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        char_img = binary_img[y:y + h, x:x + w]
        character_images.append(char_img)

    return character_images


def preprocess_character_image(char_img):
    """
    Preprocesses the character image with data augmentation.
    """
    # Resize the image to 28x28 pixels
    char_img_resized = cv2.resize(char_img, (28, 28))

    # Normalize the pixel values to the range [0, 1]
    char_img_normalized = char_img_resized / 255.0

    # Expand the dimensions of the image to match the expected input shape (None, 28, 28, 1)
    char_img_normalized = np.expand_dims(char_img_normalized, axis=(0, -1))

    return char_img_normalized


def recognize_book_title(model, img_path):
    """
    Recognizes the book title by predicting each character.
    """
    # Segment the image into characters
    character_images = segment_image(img_path)
    # Predict each character and combine them to form the title
    predicted_title = ''
    for char_img in character_images:
        # Preprocess the character image
        char_img_normalized = preprocess_character_image(char_img)

        # Predict the character using the model
        char_prediction = model.predict(char_img_normalized)
        char_index = np.argmax(char_prediction)

        # Map the index to the character using the character map
        predicted_char = character_map[char_index]

        # Append the predicted character to the title
        predicted_title += predicted_char

    return predicted_title


# Example usage
img_path = 'img_2.png'
predicted_title = recognize_book_title(model, img_path)
print(f'Recognized book title: {predicted_title}')