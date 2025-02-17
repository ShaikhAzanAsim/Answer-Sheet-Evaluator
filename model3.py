import base64
from dotenv import load_dotenv
from groq import Groq
import docx  # To read marking scheme from .docx files

load_dotenv()


class ImageOCRAnalyzer:
    def __init__(self, model_name="llama-3.2-90b-vision-preview"):
        self.model_name = model_name
        self.client = Groq()

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

    def extract_student_response(self, student_image_paths):
        """
        Extracts the student's response from multiple images.
        Combines responses into a single string.
        """
        all_responses = []

        for image_path in student_image_paths:
            encoded_image = self.ocr_analyzer.encode_image(image_path)
            prompt = (
                "You are an OCR assistant tasked with analyzing the provided student response image. "
                "Extract all text from the image as accurately as possible, including any handwritten parts. "
                "Format each response as:\n"
                "Question Number: ...\nAnswer: ...\n\n"
                "Indicate [unclear] if any text is illegible."
            )
            result = self.ocr_analyzer.perform_ocr(
                image_base64=encoded_image, prompt=prompt
            )
            all_responses.append(result.content)

        # Concatenate responses into a single response string
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

            # Join all extracted text into a single string
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
        "./mid2/Q1a.jpeg",
        "./mid2/Q1b.jpeg",
    ]
    marking_scheme_docx_path = "./mid2/solution.docx"

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
