import streamlit as st
# from ocr_model import process_answer_sheet
# from grading_model import assess_answers


# Page title
st.title("📄 AI-Powered Answer Sheet Grader")

# Sidebar for file upload
st.sidebar.title("Upload Section")
uploaded_file = st.sidebar.file_uploader(
    "Upload Answer Sheet (PDF/Image)", type=["pdf", "png", "jpg"]
)

# Main content
# if uploaded_file:
#     with st.spinner("Processing answer sheet..."):
#         # Save uploaded file
#         file_path = f"./test_files/{uploaded_file.name}"
#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.read())

#         # Process file with OCR
#         ocr_results = process_answer_sheet(file_path)
#         if "error" in ocr_results:
#             st.error(ocr_results["error"])
#         else:
#             st.success("Answer sheet processed successfully!")
#             st.subheader("Extracted Answers:")
#             st.json(ocr_results)

#             # Mock Answer Key (can be uploaded or loaded from a database)
#             answer_key = {"1": "Answer A", "2": "Answer B", "3": "Answer C"}

#             # Send for grading
#             with st.spinner("Grading the answers..."):
#                 grading_results = assess_answers(ocr_results, answer_key)
#                 if "error" in grading_results:
#                     st.error(grading_results["error"])
#                 else:
#                     st.success("Grading completed!")
#                     st.subheader("Grades:")
#                     st.json(grading_results)

# else:
#     st.warning("Please upload an answer sheet to begin.")
