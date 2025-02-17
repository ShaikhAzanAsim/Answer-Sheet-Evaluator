import cv2
import pytesseract
from pytesseract import Output


# Function to Preprocess the Image for Handwriting
def preprocess_image_for_handwriting(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Denoise the image (helps remove background noise)
    denoised_image = cv2.fastNlMeansDenoising(gray_image, None, 30, 7, 21)

    # Adaptive thresholding for better handwriting detection
    processed_image = cv2.adaptiveThreshold(
        denoised_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    return processed_image


# Function to Extract Text from Handwriting
def extract_handwritten_text(image_path):
    # Preprocess the image
    processed_image = preprocess_image_for_handwriting(image_path)

    # Perform OCR using Tesseract
    text = pytesseract.image_to_string(
        processed_image, lang="eng"
    )  # Ensure 'eng' is installed
    return text


# Function to Analyze Extracted Text
def analyze_text(text):
    # Split text into lines and clean it up
    lines = text.strip().split("\n")
    cleaned_lines = [line.strip() for line in lines if line.strip()]

    # Organize into key points
    results = {"key_points": cleaned_lines}
    return results


# Main Function
if __name__ == "__main__":
    image_path = "./mid2/Testq1.jpeg"  # Replace with your file path

    # Step 1: Extract text from the image
    extracted_text = extract_handwritten_text(image_path)
    print("Extracted Text:\n", extracted_text)

    # Step 2: Analyze the text
    analysis = analyze_text(extracted_text)
    print("\nAnalysis Results:\n", analysis)
