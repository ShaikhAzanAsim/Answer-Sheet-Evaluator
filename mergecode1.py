import os
import base64
from dotenv import load_dotenv
from groq import Groq
import fitz
import numpy as np
import cv2
import docx

class PDFAssessmentTool:
    def __init__(self, model_name="llama-3.2-90b-vision-preview"):
        self.model_name = model_name
        self.client = Groq(api_key="hehehaha")

    def encode_image(self, image_path):
        """Encode image to base64"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except FileNotFoundError:
            raise Exception(f"File not found: {image_path}")

    def preprocess_image(self, image_path):
        """Enhanced image preprocessing for OCR"""
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

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

    def process_pdf(self, pdf_path, output_dir='pdf_output'):
        """
        Comprehensive PDF processing with OCR
        Args:
            pdf_path (str): Path to PDF file
            output_dir (str): Directory to save processed files
        Returns:
            dict: Processed PDF data with OCR results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # PDF processing variables
        total_pages = 0
        total_text = ""
        page_results = []

        # Open PDF
        pdf_document = fitz.open(pdf_path)
        total_pages = len(pdf_document)

        # Process each page
        for page_num in range(total_pages):
            page = pdf_document[page_num]
            
            # Render page to image
            pix = page.get_pixmap()
            image_path = os.path.join(output_dir, f'page_{page_num+1}.jpg')
            pix.save(image_path)

            # Preprocess image
            preprocessed_image = self.preprocess_image(image_path)
            cv2.imwrite(image_path, preprocessed_image)

            # Encode image
            encoded_image = self.encode_image(image_path)

            # OCR Prompt
            ocr_prompt = (
                "Perform precise OCR on this page. Extract all text, "
                "maintaining original formatting. If text is unclear, "
                "mark as [unreadable]. Preserve question numbers and "
                "answer structures."
            )

            # Perform OCR
            try:
                ocr_result = self.perform_ocr(
                    image_base64=encoded_image, 
                    prompt=ocr_prompt
                )
                
                page_text = ocr_result.content
                total_text += page_text + "\n\n"
                page_results.append({
                    'page_number': page_num + 1,
                    'text': page_text
                })

            except Exception as e:
                page_results.append({
                    'page_number': page_num + 1,
                    'error': str(e)
                })

        # Close PDF
        pdf_document.close()

        return {
            'total_pages': total_pages,
            'total_text': total_text,
            'page_results': page_results
        }

    def perform_ocr(
        self, 
        image_base64=None, 
        prompt="", 
        temperature=0.3, 
        max_tokens=2048, 
        top_p=0.9
    ):
        """Advanced OCR with customizable parameters"""
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
            )
            return completion.choices[0].message
        except Exception as e:
            raise Exception(f"OCR Failed: {e}")

    def grade_response(self, student_response, marking_scheme=None):
        """
        Grade student response out of 100 marks
        Uses AI to evaluate response based on predefined or custom marking scheme
        """
        # Use default grading prompt if no marking scheme provided
        grading_prompt = (
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
                   f"You are an expert examiner. Grade the student's response using the following marking scheme:\n\n"
                   f"Marking Scheme:\n{marking_scheme}\n\n"
                   f"Student Response:\n{student_response}\n\n"
                    "Provide a detailed grading breakdown, allocating marks for each key point from the marking scheme. "
                    "Explain why marks were awarded or deducted. Total marks should sum to 100."
        )

        try:
            grading_result = self.perform_ocr(prompt=grading_prompt)
            return grading_result.content
        except Exception as e:
            return f"Grading Error: {e}"

def main():
    # Example usage
    pdf_tool = PDFAssessmentTool()
    
    # Paths for PDF and marking scheme
    pdf_path = 'devops_grade/Doc4.pdf'
    marking_scheme_path = '/devops_grade/marking_scheme.docx'
    
    # Extract marking scheme
    marking_scheme = pdf_tool.extract_marking_scheme_from_docx(marking_scheme_path)
    
    # Process PDF
    pdf_data = pdf_tool.process_pdf(pdf_path)
    
    # Print total pages and aggregated text
    print(f"Total Pages: {pdf_data['total_pages']}")
    
    # Grade entire response
    grade_result = pdf_tool.grade_response(
        student_response=pdf_data['total_text'], 
        marking_scheme=marking_scheme
    )
    print("Grading Result:\n", grade_result)

if __name__ == "__main__":
    main()
