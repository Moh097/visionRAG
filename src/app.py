import streamlit as st
import requests

# Set API base URL (ensure FastAPI is running on this port)
API_BASE_URL = "http://127.0.0.1:5000/api/v1"

# Streamlit UI title
st.set_page_config(page_title="Uni's RAG - AI Chat", layout="wide")
st.title("🧠 Uni's RAG - AI Chat")

# === PROJECT ID INPUT ===
project_id = st.text_input("📌 Enter Project ID", "1")

# === FILE UPLOAD SECTION ===
st.header("📂 Upload a File to Knowledge Base")
uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf"])

if uploaded_file and st.button("Upload"):
    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
    upload_url = f"{API_BASE_URL}/data/upload/{project_id}"

    with st.spinner("Uploading..."):
        response = requests.post(upload_url, files=files)

    if response.status_code == 200:
        st.success("✅ File uploaded successfully!")
        st.json(response.json())
    else:
        st.error(f"❌ Upload failed: {response.text}")

# === NLP QUESTION & ANSWERING SECTION ===
st.header("💬 Ask a Question (UniData AI)")

user_input = st.text_area("Enter your question here:")

# Slider to set how many documents to retrieve (top_k) if not ignoring top_k
initial_limit = st.slider("Initial number of documents (top_k)", 5, 200, 20)

# Checkboxes for ignoring top_k and/or threshold
ignore_top_k = st.checkbox("Ignore top_k? (retrieve all docs)")
ignore_threshold = st.checkbox("Ignore threshold? (no similarity filtering)")

# A slider for similarity threshold (0.0 to 1.0)
similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.75, 0.05)

# For demonstration, show the mode
mode_msg = "Mode: "
if ignore_top_k and not ignore_threshold:
    mode_msg += "All docs (no top_k), filter by threshold"
elif ignore_threshold and not ignore_top_k:
    mode_msg += f"Top {initial_limit}, no threshold filtering"
elif ignore_threshold and ignore_top_k:
    mode_msg += "All docs, no threshold (everything!)"
else:
    mode_msg += f"Top {initial_limit} with threshold ≥ {similarity_threshold}"
st.caption(mode_msg)

# Checkbox to request advanced re-ranking if your backend supports it
use_rerank = st.checkbox("Use advanced re-ranking (server side)?", value=True)

# Add a checkbox to optionally do a second pass
second_pass = st.checkbox("Perform a second pass if needed (refine query)?", value=True)

if st.button("Get Answer"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a question.")
        st.stop()
    
    # Validate Project ID
    try:
        project_id_int = int(project_id)  # Ensure project_id is an integer
    except ValueError:
        st.error("❌ Project ID must be a number.")
        st.stop()

    # Decide on retrieval limit:
    # If ignoring top_k, set doc_limit to a big number
    if ignore_top_k:
        doc_limit = 999999
    else:
        doc_limit = initial_limit

    # Decide on threshold:
    # If ignoring threshold, pass None
    if ignore_threshold:
        threshold_to_use = None
    else:
        threshold_to_use = similarity_threshold

    # --- PASS 1: INITIAL RETRIEVAL & ANSWER ---
    st.info(
        f"Retrieving up to {doc_limit} docs. "
        f"{'No threshold' if threshold_to_use is None else f'Threshold ≥ {threshold_to_use}'}."
    )
    
    # This payload matches your FastAPI SearchRequest model
    first_pass_payload = {
        "text": user_input,
        "limit": doc_limit,
        "similarity_threshold": threshold_to_use,  # <-- sending threshold
        "use_rerank": use_rerank
    }
    
    query_url = f"{API_BASE_URL}/nlp/index/answer/{project_id_int}"

    with st.spinner("Fetching answer (Pass 1)..."):
        response_1 = requests.post(query_url, json=first_pass_payload)

    # If the response is OK, parse JSON, otherwise show raw text
    if response_1.status_code == 200:
        # Parse JSON response safely
        result_1 = response_1.json()
        
        # Debug: Show full response from the first pass
        with st.expander("First Pass Raw JSON", expanded=False):
            st.json(result_1)

        # Extract the first pass answer
        first_pass_answer = result_1.get("answer", "")
        st.success("✅ First Pass Answer:")
        st.text_area("Generated Answer (Pass 1)", value=first_pass_answer, height=150)

        # (Optional) Show which docs were used (with score and text)
        used_docs_1 = result_1.get("used_documents", [])
        if used_docs_1:
            with st.expander("Documents Used in First Pass", expanded=False):
                for idx, doc in enumerate(used_docs_1, 1):
                    score = doc.get("score", "?")
                    text = doc.get("text", "")
                    st.markdown(f"**Doc {idx}** | **Score**: {score}\n\n{text}")

        # --- PASS 2: OPTIONAL SECOND PASS ---
        if second_pass:
            st.info("Performing second pass with refined query...")

            # Refine the query using the first pass answer as context
            refined_query = (
                user_input 
                + " | Additional context from first pass: " 
                + first_pass_answer[:250]  # limit context if necessary
            )

            # For the second pass, let's retrieve 10 more docs
            second_limit = doc_limit + 10

            second_pass_payload = {
                "text": refined_query,
                "limit": second_limit,
                "similarity_threshold": threshold_to_use,
                "use_rerank": use_rerank
            }

            st.info(
                f"Second pass retrieval: up to {second_limit} docs, "
                f"{'no threshold' if threshold_to_use is None else f'threshold ≥ {threshold_to_use}'}"
            )

            with st.spinner("Fetching answer (Pass 2)..."):
                response_2 = requests.post(query_url, json=second_pass_payload)

            if response_2.status_code == 200:
                result_2 = response_2.json()

                with st.expander("Second Pass Raw JSON", expanded=False):
                    st.json(result_2)

                # Extract the second pass answer
                final_answer = result_2.get("answer", "")
                st.success("✅ Second Pass Answer:")
                st.text_area("Generated Answer (Pass 2)", value=final_answer, height=150)

                # Show the used documents in pass 2 (including score, text)
                used_docs_2 = result_2.get("used_documents", [])
                if used_docs_2:
                    with st.expander("Documents Used in Second Pass", expanded=False):
                        for idx, doc in enumerate(used_docs_2, 1):
                            score = doc.get("score", "?")
                            text = doc.get("text", "")
                            st.markdown(f"**Doc {idx}** | **Score**: {score}\n\n{text}")
            else:
                st.error(f"❌ Second pass query failed: {response_2.status_code}")
                # Show raw text if not JSON
                st.write("Raw response text:", response_2.text)
        else:
            st.write("Second pass retrieval is disabled.")
    else:
        st.error(f"❌ First pass query failed: {response_1.status_code}")
        # Show raw text if not JSON
        st.write("Raw response text:", response_1.text)


# === VECTOR DATABASE STATUS ===
st.header("🗃 Vector Database Status")

if st.button("Check DB Connection"):
    db_status_url = f"{API_BASE_URL}/data/db_status"
    response_db = requests.get(db_status_url)

    if response_db.status_code == 200:
        st.success("✅ VectorDB Connected")
        st.json(response_db.json())
    else:
        st.error("❌ VectorDB Connection Failed")
        st.write("Raw response text:", response_db.text)
