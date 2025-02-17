import streamlit as st
from assessment_tool import ImageOCRAnalyzer, AssessmentTool
import os
import time

# Initialize the OCR analyzer and assessment tool
ocr_analyzer = ImageOCRAnalyzer()
assessment_tool = AssessmentTool(ocr_analyzer)


# Streamlit App
def main():
    st.set_page_config(
        page_title="Automated Grading System",
        page_icon=":books:",
        layout="wide",
    )

    # Sidebar for uploads
    with st.sidebar:
        st.title("Upload Files")
        st.info("Upload the necessary files to begin the grading process.")

        # File upload for student answer sheet
        uploaded_student_file = st.file_uploader(
            "Upload Student Answer Sheet (PDF/Images)",
            type=["pdf", "jpeg", "jpg", "png"],
            accept_multiple_files=True,
        )

        # File upload for marking scheme (DOCX)
        uploaded_marking_scheme = st.file_uploader(
            "Upload Marking Scheme (DOCX)", type=["docx"]
        )

    # Main Section
    st.title(":books: Automated Grading System")
    st.subheader("A simple tool for grading student responses using OCR.")
    st.markdown(
        """
        1. Upload the student answer sheet (PDF or images) in the **sidebar**.
        2. Upload the marking scheme in the **sidebar**.
        3. Click the button below to grade the responses.
        """
    )

    # Grading button
    if st.button("Grade Student Responses"):
        if not uploaded_student_file or not uploaded_marking_scheme:
            st.error(
                "Please upload both the student answer sheet and a marking scheme."
            )
            return

        try:
            student_image_paths = []
            if isinstance(uploaded_student_file, list):  # Multiple files uploaded
                
                for uploaded_file in uploaded_student_file:
                    file_name = uploaded_file.name
                    image_path = f"./uploads/{file_name}"
                    
            # Save the uploaded file
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getvalue())
        
            # Check if the file is a PDF
            if file_name.lower().endswith(".pdf"):
                
                try:
                    # Process PDF and append resulting image paths
                    pdf_image_paths = ocr_analyzer.process_pdf(image_path)
                    student_image_paths.extend(pdf_image_paths)
                    
                except Exception as e:
                    st.error(f"Error processing PDF {file_name}: {e}")
            else:
                # Assume it's an image and append its path
                student_image_paths.append(image_path)
                

            # Save marking scheme to disk
            marking_scheme_path = f"./uploads/{uploaded_marking_scheme.name}"
            with open(marking_scheme_path, "wb") as f:
                f.write(uploaded_marking_scheme.getvalue())

            # Step 1: Extract the student's response
            st.info("Extracting student responses from images...")
            with st.spinner("Processing student answer sheet..."):
                student_response = assessment_tool.extract_student_response(
                    student_image_paths
                )
            st.success("Student responses extracted successfully.")
            st.text_area("Extracted Student Response", student_response, height=200)

            # Step 2: Extract the marking scheme
            st.info("Extracting marking scheme from uploaded file...")
            with st.spinner("Processing marking scheme..."):
                marking_scheme = assessment_tool.extract_marking_scheme_from_docx(
                    marking_scheme_path
                )
            st.success("Marking scheme extracted successfully.")
            st.text_area("Marking Scheme", marking_scheme, height=200)

            # Step 3: Assess the student's response
            st.info("Assessing student responses...")
            with st.spinner("Grading the responses..."):
                assessment_result = assessment_tool.assess_student_response(
                    student_response, marking_scheme
                )
            st.success("Assessment completed.")
            st.text_area("Assessment Result", assessment_result, height=200)

            # Extract only marks obtained and calculate percentage
            try:
                raw_marks = assessment_result.split(":")[-1].strip()  # Extract the marks part
                if "/" in raw_marks:
                    numerator, denominator = map(int, raw_marks.split("/"))
                    percentage = (numerator / denominator) * 100
                else:
                    numerator = int(raw_marks)
                    denominator = 20  # Assume max marks are 20 if not provided
                    percentage = (numerator / denominator) * 100

                # Circular Progress Bar with Marks
                progress_html = f"""
                <div style="display: flex; justify-content: center; align-items: center; height: 200px;">
                    <div style="
                        position: relative;
                        width: 150px;
                        height: 150px;
                        border-radius: 50%;
                        background: conic-gradient(
                            #4CAF50 {percentage * 3.6}deg,
                            #ddd {percentage * 3.6}deg
                        );
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        font-size: 24px;
                        font-weight: bold;
                        color: #333;
                    ">
                        <div style="
                            position: absolute;
                            width: 130px;
                            height: 130px;
                            background: white;
                            border-radius: 50%;
                            display: flex;
                            justify-content: center;
                            align-items: center;
                        ">
                            {numerator}/{denominator}
                        </div>
                    </div>
                </div>
                """
                st.markdown(progress_html, unsafe_allow_html=True)
            except ValueError:
                st.error("Failed to extract the marks or calculate percentage.")

        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
