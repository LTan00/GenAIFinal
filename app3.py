import streamlit as st
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
import pandas as pd
import faiss
import requests
import re
from huggingface_hub import InferenceClient

# --- Load model and processor ---
@st.cache_resource(show_spinner=True)
def load_clip_model():
    model = CLIPModel.from_pretrained("ltan1/clip-finetuned")
    processor = CLIPProcessor.from_pretrained("ltan1/clip-finetuned")
    return model, processor

model, processor = load_clip_model()

# --- Load dataset and prepare metadata ---
@st.cache_data(show_spinner=True)
def load_data():
    df = pd.read_csv("amazon_com_ecommerce.csv")
    df = df.dropna(axis=1, how="all")
    df = df.drop(columns=["Upc Ean Code"], errors='ignore')
    fill_values = {
        "Category": "Not available",
        "Selling Price": "Not available",
        "Model Number": "Not available",
        "About Product": "Not available",
        "Product Specification": "Not available",
        "Technical Details": "Not available",
        "Shipping Weight": "Not available",
        "Product Dimensions": "Not available",
        "Variants": "Not available"
    }
    df.fillna(value=fill_values, inplace=True)

    def build_text_for_embedding(row):
        parts = [f"Product Name: {row['Product Name']}"]
        if pd.notnull(row['Category']):
            parts.append(f"Category: {row['Category']}")
        return " | ".join(parts)

    df['combined_text'] = df.apply(build_text_for_embedding, axis=1)

    metadata_list = []
    for _, row in df.iterrows():
        metadata_list.append({
            "image_url": row.get("Image", ""),
            "product_url": row.get("Product Url", ""),
            "Variants products link": row.get("Variants", ""),
            "Shipping Weight": row.get("Shipping Weight", ""),
            "Product Dimensions": row.get("Product Dimensions", ""),
            "Product Specification": row.get("Product Specification", ""),
            "Technical Details": row.get("Technical Details", ""),
            "Is Amazon Seller": row.get("Is Amazon Seller", ""),
            "Selling Price": row.get("Selling Price", ""),
            "Model Number": row.get("Model Number", ""),
            "About Product": row.get("About Product", ""),
            "combined_text": row.get("combined_text", "")
        })
    return df, metadata_list

df, metadata_list = load_data()

# --- Load FAISS index and embeddings ---
@st.cache_resource(show_spinner=True)
def load_faiss_index():
    final_embs = np.load("final_clip_embeddings.npy")
    d = final_embs.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(final_embs.astype("float32"))
    return index

index = load_faiss_index()

# --- Embedding functions with device fix ---
def get_text_embedding(text):
    inputs = processor(text=[text], return_tensors="pt", truncation=True)
    inputs = {k: v.to("cpu") for k, v in inputs.items()}
    with torch.no_grad():
        emb = model.get_text_features(**inputs).squeeze().cpu().numpy()
    return emb

def get_image_embedding(image_file):
    try:
        image = Image.open(image_file).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        with torch.no_grad():
            emb = model.get_image_features(**inputs).squeeze().cpu().numpy()
        return emb
    except Exception as e:
        st.error(f"Failed to process uploaded image: {e}")
        return None

# --- Build prompt with top metadata ---
def build_prompt(query, top_metadata):
    prompt_parts = []
    for i, item in enumerate(top_metadata):
        section = f"{i+1}."
        for key, value in item.items():
            if pd.notnull(value) and str(value).strip() != "":
                section += f"\n  - {key}: {value}"
        prompt_parts.append(section)
    context = "\n\n".join(prompt_parts)

    prompt = f"""You are a helpful AI assistant for an e-commerce website. Your job is to answer customer questions based on available product details.

User question: {query}

Here are some product descriptions that may be relevant:
{context}

Provide an informative and accurate response using the product information above. If multiple items are relevant, mention them.
If you don't know the answer, say "I'm not sure based on the available product data."
If query requests to show pictures related to the product, please return the corresponding image URLs from the context."""
    
    return prompt.strip()

# --- Query embedding & retrieval ---
def get_query_embedding(text_query=None, image_file=None, k=3):
    text_emb = get_text_embedding(text_query) if text_query else None
    image_emb = get_image_embedding(image_file) if image_file else None

    if text_emb is not None and image_emb is not None:
        query_emb = (0.3 * text_emb + 0.7 * image_emb).reshape(1, -1)
    elif text_emb is not None:
        query_emb = text_emb.reshape(1, -1)
    elif image_emb is not None:
        query_emb = image_emb.reshape(1, -1)
    else:
        raise ValueError("Must provide at least a text or image query.")

    D, I = index.search(query_emb.astype("float32"), k)
    retrieved_items = [metadata_list[i] for i in I[0]]

    prompt = build_prompt(text_query if text_query else "", retrieved_items)
    return prompt, retrieved_items

# --- Setup Huggingface LLM client ---
@st.cache_resource(show_spinner=True)
def load_llm_client():
    return InferenceClient(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        token=st.secrets.get("hf_token", "")  # Store your token in .streamlit/secrets.toml
    )

client = load_llm_client()

# --- Streamlit UI ---
st.title("🦜 Multimodal E-commerce Assistant")

st.markdown(
    """
    Ask a question about products or upload an image to find relevant product info.
    You can enter text queries, upload an image, or both!
    """
)

with st.form(key="query_form"):
    text_query = st.text_input("Enter your question or description:")
    uploaded_image = st.file_uploader("Upload an image of the product (optional):", type=["png", "jpg", "jpeg"])
    submit_button = st.form_submit_button(label="Ask")

if submit_button:
    if not text_query and not uploaded_image:
        st.warning("Please enter a text query or upload an image to ask.")
    else:
        try:
            with st.spinner("Generating response..."):
                prompt, top_items = get_query_embedding(text_query, image_file=uploaded_image, k=2)
                
                # Use chat_completion instead of text_generation
                messages = [{"role": "user", "content": prompt}]
                response = client.chat_completion(
                    messages=messages,
                    max_tokens=500,
                    temperature=0.05
                )
                
                # Extract the response text
                response_text = response.choices[0].message.content
            
            st.subheader("Response:")
            st.write(response_text)

            # Extract image URLs from the LLM response and display them
            img_urls = re.findall(r"https:\/\/[^\s\"']+?\.jpg", response_text, flags=re.IGNORECASE)
            if img_urls:
                st.subheader("Images:")
                for url in img_urls:
                    st.image(url, use_column_width=True)
            else:
                # If no images found in response, optionally show images from top results
                st.info("No images found in the response.")

        except Exception as e:
            st.error(f"Something went wrong: {e}")
