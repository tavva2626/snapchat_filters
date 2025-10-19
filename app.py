import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile

# ----------------- Page Config -----------------
st.set_page_config(
    page_title="Snapchat Filters Pro",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- CSS Styling -----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
.stApp {
    background: linear-gradient(to right, #fceabb, #f8b500);
    font-family: 'Roboto', sans-serif;
}
.title {
    color: #FF4B4B;
    font-size: 56px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 0;
}
.subtitle {
    color: #333333;
    font-size: 22px;
    text-align: center;
    margin-top: 0;
    margin-bottom: 30px;
}
.filter-img {
    border-radius: 20px;
    box-shadow: 0px 8px 15px rgba(0,0,0,0.2);
    transition: transform 0.2s;
}
.filter-img:hover {
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">📸 Snapchat Filters Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Apply multiple filters and download all at once!</div>', unsafe_allow_html=True)

# ----------------- Filter Function -----------------
def apply_filter(img, filter_name, params={}):
    img = np.array(img)
    brightness = params.get("brightness", 0)
    contrast = params.get("contrast", 1.0)
    blur_val = params.get("blur", 0)
    sepia_intensity = params.get("sepia", 1.0)
    canny_min = params.get("canny_min", 100)
    canny_max = params.get("canny_max", 200)

    if filter_name == "Original":
        return img
    elif filter_name == "Gray":
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    elif filter_name == "Gaussian Blur":
        k = blur_val*2+1
        return cv2.GaussianBlur(img, (k, k), 0)
    elif filter_name == "Canny Edge":
        edges = cv2.Canny(img, canny_min, canny_max)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    elif filter_name == "Sepia":
        sepia_matrix = np.array([[0.393, 0.769, 0.189],
                                 [0.349, 0.686, 0.168],
                                 [0.272, 0.534, 0.131]])
        sepia_img = cv2.transform(img, sepia_matrix)
        sepia_img = np.clip(sepia_img * sepia_intensity + img*(1-sepia_intensity), 0, 255)
        return sepia_img.astype(np.uint8)
    elif filter_name == "Negative":
        return 255 - img
    elif filter_name == "Pencil Sketch":
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        inv = 255 - gray_img
        blur_inv = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray_img, 255 - blur_inv, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
    elif filter_name == "Cartoon":
        gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 5)
        edges = cv2.adaptiveThreshold(gray, 255,
                                      cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(img, 9, 250, 250)
        return cv2.bitwise_and(color, color, mask=edges)
    elif filter_name == "Emboss":
        kernel = np.array([[-2, -1, 0],
                           [-1,  1, 1],
                           [ 0,  1, 2]])
        return cv2.filter2D(img, -1, kernel) + 128
    elif filter_name == "Warm Tone":
        warm = img.copy()
        warm[:,:,0] = cv2.add(warm[:,:,0], -30)
        warm[:,:,2] = cv2.add(warm[:,:,2], 40)
        return warm
    elif filter_name == "Cool Tone":
        cool = img.copy()
        cool[:,:,0] = cv2.add(cool[:,:,0], 40)
        cool[:,:,2] = cv2.add(cool[:,:,2], -30)
        return cool
    elif filter_name == "Bright Light":
        return cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
    elif filter_name == "Dark Mood":
        return cv2.convertScaleAbs(img, alpha=contrast*0.6, beta=brightness-30)
    elif filter_name == "Vivid (High Contrast)":
        return cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    elif filter_name == "Vintage":
        return cv2.addWeighted(img, 0.7, np.full(img.shape, 120, dtype=np.uint8), 0.3, 0)
    else:
        return img

# ----------------- Sidebar -----------------
st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Upload an image:", type=["png","jpg","jpeg"])

filter_names = [
    "Original", "Gray", "Gaussian Blur", "Canny Edge", "Sepia", "Negative",
    "Pencil Sketch", "Cartoon", "Emboss", "Warm Tone", "Cool Tone",
    "Bright Light", "Dark Mood", "Vivid (High Contrast)", "Vintage"
]

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Select Filters to Apply")
    selected_filters = st.multiselect("Choose filters (order matters!)", filter_names, default=["Original"])

    # ----------------- Filter Parameters -----------------
    filter_params = {}
    for fname in selected_filters:
        if fname == "Gaussian Blur":
            filter_params[fname] = {"blur": st.slider(f"{fname} Blur", 0, 10, 3, key=f"{fname}_blur")}
        elif fname == "Canny Edge":
            filter_params[fname] = {
                "canny_min": st.slider(f"{fname} Min", 50, 200, 100, key=f"{fname}_min"),
                "canny_max": st.slider(f"{fname} Max", 100, 300, 200, key=f"{fname}_max")
            }
        elif fname == "Sepia":
            filter_params[fname] = {"sepia": st.slider(f"{fname} Intensity", 0.0, 1.0, 1.0, 0.1, key=f"{fname}_sepia")}
        elif fname in ["Bright Light", "Dark Mood"]:
            filter_params[fname] = {
                "brightness": st.slider(f"{fname} Brightness", -100, 100, 40, key=f"{fname}_bright"),
                "contrast": st.slider(f"{fname} Contrast", 0.5, 2.0, 1.2, 0.1, key=f"{fname}_contrast")
            }
        else:
            filter_params[fname] = {}

    # ----------------- Apply Filters Sequentially -----------------
    stacked_image = np.array(image)
    for fname in selected_filters:
        stacked_image = apply_filter(stacked_image, fname, filter_params[fname])

    st.subheader("Result after Stacking Filters")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original", use_container_width=True)
    with col2:
        st.image(stacked_image, caption="Filtered", use_container_width=True)

    # ----------------- Download Single Filter -----------------
    if st.button("💾 Save Final Filtered Image"):
        Image.fromarray(stacked_image).save("stacked_filtered_image.png")
        st.success("Saved as stacked_filtered_image.png!")

    # ----------------- Download All Filters as ZIP -----------------
    if st.button("💾 Save All Filters as ZIP"):
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for fname in filter_names:
                filtered_img = apply_filter(np.array(image), fname)
                img_pil = Image.fromarray(filtered_img)
                img_byte_arr = io.BytesIO()
                img_pil.save(img_byte_arr, format='PNG')
                zip_file.writestr(f"{fname.replace(' ','_')}.png", img_byte_arr.getvalue())
        st.download_button(label="Download ZIP", data=zip_buffer.getvalue(), file_name="all_filters.zip", mime="application/zip")

else:
    st.info("Upload an image to apply filters and download results.")
