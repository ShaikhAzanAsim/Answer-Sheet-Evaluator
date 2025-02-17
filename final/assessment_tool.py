import os
import base64
from dotenv import load_dotenv
from groq import Groq
import cv2
import numpy as np
import docx  # For extracting marking scheme from .docx f
import fitz  # PyMuPDF

GROQ_API_KEY = "xyz"

class ImageOCRAnalyzer:
    def __init__(self, model_name="llama-3.2-90b-vision-preview"):
        self.model_name = model_name
        self.client = Groq(
            api_key=GROQ_API_KEY
        )  # Replace with your actual API client initialization

    def preprocess_image(self, image_path, output_path):
        """
        Preprocess an image to enhance quality for OCR.
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            # Load the image in color
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image for preprocessing: {image_path}")

            # Save the preprocessed image (no grayscale applied)
            cv2.imwrite(output_path, image)
            return output_path
        except Exception as e:
            raise Exception(f"Error during image preprocessing: {e}")

    def process_pdf(self, pdf_path, output_dir="./uploads"):
        """
        Convert a PDF into images and return the image paths.
        """
        # Ensure absolute path
        output_dir = os.path.abspath(output_dir)

        try:
            # Create the output directory if it does not exist
            os.makedirs(output_dir, exist_ok=True)
        except Exception as dir_error:
            raise Exception(
                f"Failed to create output directory '{output_dir}': {dir_error}"
            )

        try:
            # Open the PDF file
            doc = fitz.open(pdf_path)
            image_paths = []

            for page_number in range(len(doc)):
                # Load a single page and render it to an image
                page = doc.load_page(page_number)
                pix = page.get_pixmap()
                # Construct the output image path
                output_path = os.path.join(output_dir, f"page_{page_number + 1}.png")
                # Save the rendered image
                pix.save(output_path)
                image_paths.append(output_path)

            # Close the PDF document
            doc.close()

            return image_paths

        except Exception as e:
            raise Exception(f"Error processing PDF '{pdf_path}': {e}")

    def encode_image(self, image_path):
        """
        Encode an image to base64 format.
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except FileNotFoundError:
            raise Exception(f"File not found: {image_path}")

    def perform_ocr(self, image_base64=None, prompt="", **kwargs):
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
                model=self.model_name, messages=messages, **kwargs
            )
            return completion.choices[0].message
        except Exception as e:
            raise Exception(f"Failed to perform OCR: {e}")

class AssessmentTool:
    def __init__(self, ocr_analyzer):
        self.ocr_analyzer = ocr_analyzer

    def extract_student_response(self, student_image_paths):
        """
        Extract the student's response from multiple images.
        """
        all_responses = []
        for image_path in student_image_paths:
            try:
                # Skip preprocessing for images extracted from PDF
                # Directly encode and process the image
                encoded_image = self.ocr_analyzer.encode_image(image_path)

                # Prompt for OCR
                prompt = (
                    "You are a highly accurate OCR assistant specialized in analyzing text from both handwritten and printed content in images. "
                    "Your task is to extract all visible text from the provided image, including detailed formatting, and organize it into a structured format as follows:"
                    "1. Clearly identify the **Question Number** (if present) in the image (e.g., Q1a)."
                    "2. Extract and structure the **Answer** text verbatim, maintaining its logical order. If the image includes lists, bullet points, or numbered items, ensure they are preserved in the response."
                    "3. If the content is unclear or illegible, mark it as [unclear] in the relevant section, and do not attempt to guess the text."
                    "If some parts are unclear, mark them as [unclear]. Provide the extracted text verbatim.\n\n"
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
                    "Example 2:\n"
                    "Input Image Content:\n"
                    "1. FAST Delivery"
                    "Since development & ops team work continuously all collaboratively, it allows for faster development of features through continuous integration and deployment."
                    "2. Quality Product"
                    "Test is carried out in each stage of DevOps resulting in a quality product free of any bugs."
                    "3. Customer Trust"
                    "Customers are involved throughout the lifecycle with giving continuous feedback of the software & viewing changes which creates a sense of satisfaction."
                    "4. Mean Value Product"
                    "The product produced is of value and to the point of what was required."
                    "5. Collaboration"
                    "The developers and operations team work together throughout, communicating back and forth, breaking the practices of silos. Work of each team is visible to others."
                    "6. Seamless Integration"
                    "Since there is clear communication & each work of each team is visible to each other, the product is delivered without any conflicts."
                    "Extracted Response:"
                    "Question Number: Q1a"
                    "Answer:"
                    "1. FAST Delivery"
                    "Since development & ops team work continuously all collaboratively, it allows for faster development of features through continuous integration and deployment."
                    "2. Quality Product"
                    "Test is carried out in each stage of DevOps resulting in a quality product free of any bugs."
                    "3. Customer Trust"
                    "Customers are involved throughout the lifecycle with giving continuous feedback of the software & viewing changes which creates a sense of satisfaction."
                    "4. Mean Value Product"
                    "The product produced is of value and to the point of what was required."
                    "5. Collaboration"
                    "The developers and operations team work together throughout, communicating back and forth, breaking the practices of silos. Work of each team is visible to others."
                    "6. Seamless Integration"
                    "Since there is clear communication & each work of each team is visible to each other, the product is delivered without any conflicts."
                    ### Instructions for New Input:
                    "Now, analyze the provided image and:  "
                    "- Identify the **Question Number** if available.  "
                    "- Extract the **Answer** text verbatim, preserving structure and formatting.  "
                    "- Use [unclear] where content is illegible or ambiguous. "
                    "- Ensure the output is clear, logical, and follows the examples above."
                )

                # Perform OCR
                result = self.ocr_analyzer.perform_ocr(
                    image_base64=encoded_image, prompt=prompt
                )
                all_responses.append(result.content)

            except Exception as e:
                all_responses.append(f"[Error processing {image_path}: {e}]")
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
            "You are an evaluator tasked with assessing a student's answers using a provided marking scheme. Evaluate each question by comparing the student's response to the correct answer in the marking scheme. Follow these guidelines for grading:"
            "Evaluate each question, comparing the student's response to the correct answer in the marking scheme. Be lenient with evaluation"
            "Award marks for correct answers and provide a total score. Structure your response as:\n"
            "Question 1: Correct/Incorrect - Awarded Marks: X\n"
            "...\nTotal Marks: Y\n\n"
            "Marking Scheme:\n{marking_scheme}\n\nStudent Response:\n{student_response}"
        ).format(marking_scheme=marking_scheme, student_response=student_response)

        try:
            result = self.ocr_analyzer.perform_ocr(prompt=prompt)
            return result.content
        except Exception as e:
            raise Exception(f"Failed to assess student response: {e}")
