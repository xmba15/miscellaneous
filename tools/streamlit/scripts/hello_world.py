import cv2
import numpy as np
import streamlit as st
from PIL import Image


def main():
    """main"""
    st.title("hello world")
    with st.form("upload-image", clear_on_submit=True):
        image_files = st.file_uploader(
            "Upload Your Image", type=["jpg", "png", "jpeg"], accept_multiple_files=True
        )
        submitted = st.form_submit_button("UPLOAD!")

        if submitted and image_files is not None:
            st.write("Uploaded")
            if len(image_files) != 0:
                for image_file in image_files:
                    image = np.array(Image.open(image_file))
                    st.image(image, caption=image_file.name, use_column_width=True)


if __name__ == "__main__":
    main()
