import base64
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class ImageOCRAnalyzer:
    def __init__(self, model_name="llama-3.2-90b-vision-preview"):
        self.model_name = model_name
        self.client = Groq()

    def encode_image(self, image_path):
        """
        Encodes the image from the given file path into base64 format.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def perform_ocr(self, image_base64, temperature=1, max_tokens=1024, top_p=1):
        """
        Perform OCR on the image and extract text in a structured format.
        """
        image_url = f"data:image/jpeg;base64,{image_base64}"

        # Define the prompt with image data and clear instructions
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are an OCR assistant tasked with analyzing the provided image. Your primary objectives are:\n"
                            "1. Recognize all visible text in the image as accurately as possible.\n"
                            "2. If any words or phrases are unclear, indicate this with [unclear] in your transcription.\n"
                            "Provide only the transcription without any additional comments.\n"
                            "Format your response as:\n"
                            "- Question 1(typed text): ...\n then Answer 1 (handwritten): ... and so on"
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]

        # Generate the OCR results
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


if __name__ == "__main__":
    # Initialize the ImageOCRAnalyzer with the specified model
    ocr_analyzer = ImageOCRAnalyzer()

    # Encode the image (assumed to be located at the specified path)
    encoded_image = ocr_analyzer.encode_image("./test_files/sample4.jpeg")

    # Perform OCR on the encoded image
    ocr_result = ocr_analyzer.perform_ocr(encoded_image)

    # Access the text content of the result
    ocr_text = ocr_result.content  # Use .content to access the message text

    # Print the OCR results, skipping a line for each '\n'
    for line in ocr_text.split("\n"):
        print(line)
        print()  # Add a blank line after each line to skip