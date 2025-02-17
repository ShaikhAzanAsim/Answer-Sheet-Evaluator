import base64
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

import cv2
import numpy as np

def preprocess_image(image_path):
    """
    Preprocesses the image using OpenCV to enhance OCR accuracy.
    """
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Apply morphological operations to remove noise and fill gaps
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    # Save the preprocessed image to a temporary file
    temp_path = "temp_preprocessed_image.png"
    cv2.imwrite(temp_path, cleaned)
    
    return temp_path
class ImageOCRAnalyzer:
    def __init__(self, model_name="llama-3.2-90b-vision-preview"):
        self.model_name = model_name
        self.client = Groq()

    def encode_image(self, image_path):
        """
        Encodes the preprocessed image from the given file path into base64 format.
        """
        try:
            # preprocessed_image_path = preprocess_image(image_path)
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except FileNotFoundError:
            raise Exception(f"File not found: {image_path}")

    def perform_ocr(
        self, image_base64=None, prompt="", temperature=1, max_tokens=1024, top_p=1
    ):
        """
        Perform OCR and process the image using the provided prompt.
        """
        # Construct the message payload
        messages = [{"role": "user", "content": prompt}]
        if image_base64:
            image_url = f"data:image/jpeg;base64,{image_base64}"
            messages[0]["content"] = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]

        # Generate the OCR results
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=False,
                stop=None,
            )
            return completion.choices[0].message
        except Exception as e:
            raise Exception(f"Failed to perform OCR: {e}")
class AssessmentTool:
    def __init__(self, ocr_analyzer):
        self.ocr_analyzer = ocr_analyzer

    def extract_student_response(self, student_image_path):
        """
        Extract the student's response from their image.
        """
        encoded_image = self.ocr_analyzer.encode_image(student_image_path)
        prompt = (
            "You are an OCR assistant analyzing a student response image. Extract all text accurately, including subparts. "
            "Group the text for each main question and subparts as follows:\n\n"
            "Question 1:\n"
            "    a) Subpart (i): [Extracted question text] - Student's Answer: [Extracted student response]\n"
            "    a) Subpart (ii): [Extracted question text] - Student's Answer: [Extracted student response]\n"
            "    b) Subpart (i): [Extracted question text] - Student's Answer: [Extracted student response]\n"
            "    b) Subpart (ii): [Extracted question text] - Student's Answer: [Extracted student response]\n\n"
            "If any text is unclear or illegible, indicate [unclear] for that part. "
            "Make sure to capture the student's answer along with the corresponding question text. "
            "Ensure the text is grouped properly for each question and subpart, and the format follows the example."
        )
        result = self.ocr_analyzer.perform_ocr(image_base64=encoded_image, prompt=prompt)
        return result.content

    def extract_marking_scheme(self, marking_scheme_image_path):
        """
        Extract the marking scheme from the provided image.
        """
        encoded_image = self.ocr_analyzer.encode_image(marking_scheme_image_path)
        prompt = (
            "You are an OCR assistant analyzing a marking scheme image. Extract all text and format it as follows:\n\n"
            "Marking Scheme:\n"
            "Question X:\n"
            "    a) Subpart (i): Correct Answer: [Answer text] - Marks: [X]\n"
            "    a) Subpart (ii): Correct Answer: [Answer text] - Marks: [X]\n"
            "    b) Subpart (i): Correct Answer: [Answer text] - Marks: [X]\n"
            "    b) Subpart (ii): Correct Answer: [Answer text] - Marks: [X]\n\n"
            "Group subparts under their main questions. Indicate [unclear] for illegible parts. "
            "Format it as:\n"
            "Question: ...\nAnswer: ...\nMarks: ..."
        )
        result = self.ocr_analyzer.perform_ocr(image_base64=encoded_image, prompt=prompt)
        return result.content

    def assess_student_response(self, student_response, marking_scheme):
        """
        Assess the student's response using the marking scheme.
        """
        prompt = (
            "You are an evaluator comparing a student's response to a marking scheme. Evaluate each main question and its subparts together. "
            "Award marks only for correct answers, and format your assessment as follows:\n\n"
            "Assessment:\n"
            "Question X:\n"
            "    a) Subpart (i): [Correct/Incorrect] - Awarded Marks: [X]\n"
            "    a) Subpart (ii): [Correct/Incorrect] - Awarded Marks: [X]\n"
            "    b) Subpart (i): [Correct/Incorrect] - Awarded Marks: [X]\n"
            "    b) Subpart (ii): [Correct/Incorrect] - Awarded Marks: [X]\n\n"
            "...\nTotal Marks: Y\n\n"
            "Marking Scheme:\n{marking_scheme}\n\nStudent Response:\n{student_response}"
        ).format(marking_scheme=marking_scheme, student_response=student_response)

        result = self.ocr_analyzer.perform_ocr(prompt=prompt)
        return result.content

if __name__ == "__main__":
    # Initialize the OCR analyzer
    ocr_analyzer = ImageOCRAnalyzer()

    # Initialize the assessment tool
    assessment_tool = AssessmentTool(ocr_analyzer)

    # File paths for student response and marking scheme images
    student_image_path = "./test_files/stud2.jpeg"
    marking_scheme_image_path = "./test_files/sol3.png"

    try:
        # Step 1: Extract the student's response
        print("Extracting student response...")
        student_response = assessment_tool.extract_student_response(student_image_path)
        print("Student Response:\n", student_response)

        # Step 2: Extract the marking scheme
        print("\nExtracting marking scheme...")
        marking_scheme = assessment_tool.extract_marking_scheme(marking_scheme_image_path)
        print("Marking Scheme:\n", marking_scheme)

        # Step 3: Assess the student's response
        print("\nAssessing student response...")
        assessment_result = assessment_tool.assess_student_response(student_response, marking_scheme)
        print("Assessment Result:\n", assessment_result)

    except Exception as e:
        print(f"Error: {e}")