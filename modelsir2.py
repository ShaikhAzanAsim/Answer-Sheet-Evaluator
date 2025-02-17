import base64
from dotenv import load_dotenv
from groq import Groq
import cv2
import numpy as np
from matplotlib import pyplot as plt
import docx  # To read marking scheme from .docx files

load_dotenv()


class ImageOCRAnalyzer:
    def __init__(self, model_name="llama-3.2-90b-vision-preview"):
        self.model_name = model_name
        self.client = Groq()

    def preprocess_image(self, image_path, output_path):
        """
        Preprocess the image to enhance quality for OCR.
        - Convert to grayscale
        - Apply contrast enhancement
        - Apply adaptive thresholding
        - Denoise the image
        """
        try:
            # Read the image
            image = cv2.imread(image_path)

            # Step 1: Convert to Grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            

            # Step 4: Denoising using a Median Filter
            denoised_image = cv2.medianBlur(gray_image, 3)

            # Save and return the preprocessed image
            cv2.imwrite(output_path, denoised_image)
            return output_path

        except Exception as e:
            raise Exception(f"Error during image preprocessing: {e}")

    def encode_image(self, image_path):
        """
        Encodes the image from the given file path into base64 format.
        """
        try:
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
        messages = [{"role": "user", "content": prompt}]
        if image_base64:
            image_url = f"data:image/jpeg;base64,{image_base64}"
            messages[0]["content"] = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]

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

    def extract_student_response(self, student_image_paths):
        """
        Extracts the student's response from multiple images.
        Combines responses into a single string.
        """
        all_responses = []

        for image_path in student_image_paths:
            # Preprocess the image to enhance OCR accuracy
            preprocessed_path = self.ocr_analyzer.preprocess_image(
                image_path, f"{image_path.split('.')[0]}_preprocessed.jpeg"
            )

            encoded_image = self.ocr_analyzer.encode_image(preprocessed_path)
            prompt = (
                "You are a highly accurate OCR assistant tasked with analyzing a student's handwritten response. "
                "Extract all visible text, including handwritten parts, and structure it as follows:\n"
                "Question Number: (e.g., Q1a)\n"
                "Answer: (Extracted answer content)\n\n"
                "If some parts are unclear, mark them as [illegible]. Provide the extracted text verbatim.\n\n"
                "Here are examples to guide you:\n\n"
                "Example 1:\n"
                "Input Image Content:\n"
                "DevOps refers to a mindset of software teams to deliver a product through integration, collaboration, "
                "and communication between development and operations teams.\n"
                "Benefits:\n"
                "1. Fast Time to Market: Due to rapid and frequent merging of code and features, the product is delivered at a faster rate through 'continuous/incremental improvement'.\n"
                "2. Automated Testing Improves Reliability: Automated testing at each step ensures the quality of code is reliable and increases the chance of 'early bug detection'.\n\n"
                "Extracted Response:\n"
                "Question Number: Q1a\n"
                "Answer: DevOps refers to a mindset of software teams to deliver a product through integration, collaboration, and communication between development and operations teams.\n"
                "Benefits:\n"
                "1. Fast Time to Market: Due to rapid and frequent merging of code and features, the product is delivered at a faster rate through 'continuous/incremental improvement'.\n"
                "2. Automated Testing Improves Reliability: Automated testing at each step ensures the quality of code is reliable and increases the chance of 'early bug detection'.\n\n"
                "Example 2:\n"
                "Input Image Content:\n"
                "3. End-to-End Product Responsibility/Ownership: Due to the DevOps cycle, both the development and operations teams are tightly integrated and have a higher level of ownership towards the product.\n"
                "4. Eliminating Manual Tasks: Manual tasks of building and deployment are automated, increasing team efficiency by eliminating 'repetitive tasks'.\n"
                "5. Prevent Large Scale Issues at Production: Continuous testing and deployment of features help identify and resolve potential problems early.\n"
                "6. Feedback Loops: Monitoring team and user responses improve the UX through continuous feedback.\n\n"
                "Extracted Response:\n"
                "Question Number: Q1a (continued)\n"
                "Answer:\n"
                "3. End-to-End Product Responsibility/Ownership: Due to the DevOps cycle, both the development and operations teams are tightly integrated and have a higher level of ownership towards the product.\n"
                "4. Eliminating Manual Tasks: Manual tasks of building and deployment are automated, increasing team efficiency by eliminating 'repetitive tasks'.\n"
                "5. Prevent Large Scale Issues at Production: Continuous testing and deployment of features help identify and resolve potential problems early.\n"
                "6. Feedback Loops: Monitoring team and user responses improve the UX through continuous feedback.\n\n"
                "Now, analyze the provided image and ensure maximum accuracy."
            )

            result = self.ocr_analyzer.perform_ocr(
                image_base64=encoded_image, prompt=prompt
            )
            all_responses.append(result.content)

        return "\n".join(all_responses)

    def extract_marking_scheme_from_docx(self, docx_path):
        """
        Extract the marking scheme from a .docx file.
        """
        try:
            doc = docx.Document(docx_path)
            marking_scheme = []

            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    marking_scheme.append(text)

            return "\n".join(marking_scheme)

        except Exception as e:
            raise Exception(f"Failed to extract marking scheme from DOCX: {e}")

    def assess_student_response(self, student_response, marking_scheme):
        """
        Assess the student's response using the marking scheme.
        """
        prompt = (
            "You are an evaluator tasked with assessing a student's answers using a provided marking scheme. "
            "Evaluate each question, comparing the student's response to the correct answer in the marking scheme. "
            "Award marks for correct answers and provide a total score. Structure your response as:\n"
            "Be lenient with grading"
            "Question 1: Correct/Incorrect - Awarded Marks: X\n"
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

    # File paths for student response images and marking scheme docx
    student_image_paths = [
        "./mid2/Testq1.jpeg"
    ]
    marking_scheme_docx_path = (
        "./mid2/solution.docx"  # Replace with your marking scheme file path
    )

    try:
        # Step 1: Extract the student's response
        print("Extracting student responses from images...")
        student_response = assessment_tool.extract_student_response(student_image_paths)
        print("Student Response:\n", student_response)

        # Step 2: Extract the marking scheme from the DOCX file
        print("\nExtracting marking scheme from DOCX...")
        marking_scheme = assessment_tool.extract_marking_scheme_from_docx(
            marking_scheme_docx_path
        )
        print("Marking Scheme:\n", marking_scheme)

        # Step 3: Assess the student's response
        print("\nAssessing student response...")
        assessment_result = assessment_tool.assess_student_response(
            student_response, marking_scheme
        )
        print("Assessment Result:\n", assessment_result)

    except Exception as e:
        print(f"Error: {e}")
