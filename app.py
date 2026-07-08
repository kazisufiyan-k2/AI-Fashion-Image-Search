
import os
import streamlit as st
from PIL import Image

from search import load_model, load_index, load_image_paths, get_results

st.set_page_config(page_title="AI Fashion Image Search", layout="wide")

EMBEDDINGS_DIR = "embeddings"
INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "faiss.index")
PATHS_PKL = os.path.join(EMBEDDINGS_DIR, "image_paths.pkl")


@st.cache_resource(show_spinner="Loading CLIP model (first run only)...")
def get_model():
    return load_model()


@st.cache_resource(show_spinner="Loading FAISS index...")
def get_index_and_paths():
    index = load_index(INDEX_PATH)
    image_paths = load_image_paths(PATHS_PKL)
    return index, image_paths


def show_image(image_or_path):
    """
    Display an image filling its container, regardless of Streamlit version.
    Different Streamlit releases have used use_container_width=True,
    width="stretch", or neither — try each so this never crashes again
    after a Streamlit upgrade/downgrade.
    """
    try:
        st.image(image_or_path, width="stretch")
    except TypeError:
        try:
            st.image(image_or_path, use_container_width=True)
        except TypeError:
            st.image(image_or_path)


def sidebar():
    with st.sidebar:
        st.header("Project Information")
        st.markdown("**Model:** CLIP (ViT-B/32)")
        st.markdown("**Similarity Search:** FAISS")
        st.markdown("**Dataset:** Fashion Images")
        st.markdown("**Results:** Top-5 Similar Images")


def main():
    sidebar()
    st.title("AI Fashion Image Search")
    st.write("Upload a fashion image and find the **Top-5 most similar products** using **CLIP + FAISS**.")
    st.divider()

    if not os.path.exists(INDEX_PATH) or not os.path.exists(PATHS_PKL):
        st.error(
            "No precomputed index found. Run `python embed.py` first to generate "
            "embeddings/faiss.index and embeddings/image_paths.pkl, then redeploy."
        )
        return

    model, preprocess = get_model()
    index, image_paths = get_index_and_paths()

    st.subheader("Upload an Image")
    uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        st.info("Please upload an image to begin searching.")
        return

    query_image = Image.open(uploaded)

    col_query, col_status = st.columns([1, 3])
    with col_query:
        st.subheader("Query Image")
        show_image(query_image)

    with col_status:
        st.subheader("Searching...")
        with st.spinner("Searching..."):
            results = get_results(model, preprocess, index, image_paths, query_image, k=5)
        st.success("Search Completed!")

    st.divider()
    st.subheader("Top 5 Similar Images")
    cols = st.columns(5)
    for col, r in zip(cols, results):
        with col:
            if os.path.exists(r["path"]):
                show_image(r["path"])
            else:
                st.warning("Image file missing")
            st.markdown(f"**Similarity:** {r['similarity'] * 100:.2f}%")


if __name__ == "__main__":
    main()