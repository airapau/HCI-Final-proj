import streamlit as st
import requests
import whisper
from tempfile import NamedTemporaryFile
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import json

# IMPORTANT: Import numpy explicitly to catch issues early
try:
    import numpy as np
except ImportError:
    st.error("❌ NumPy is not installed. Please run: pip install numpy==1.24.3")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="SoundScape",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS (No changes here)
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    
    /* Header styling */
    h1 {
        color: #ffffff;
        font-size: 3.5rem !important;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    /* Brand name styling */
    .brand-name {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .brand-sound {
        color: #ffffff;
    }
    
    .brand-scape {
        color: #00d66f;
    }
    
    /* Subtitle */
    .subtitle {
        color: #ffffff;
        font-size: 1.2rem;
        margin-bottom: 0rem;
        padding-bottom: 0rem;
    }
    
    /* Remove all default spacing and containers */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
        max-width: 100%;
    }
    
    /* Target and hide empty divs */
    div[data-testid="stVerticalBlock"] > div:has(> .element-container:empty) {
        display: none !important;
    }
    
    div[data-testid="stVerticalBlock"] > div:first-child:empty {
        display: none !important;
    }
    
    .element-container:empty {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Remove gaps between elements */
    .stMarkdown {
        margin-bottom: 0rem;
    }
    
    /* Input container */
    .input-container {
        background-color: #2a2a2a;
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #00d66f;
        color: #000000;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #00ff80;
        transform: translateY(-2px);
    }
    
    /* Form submit button styling */
    .stForm button[type="submit"] {
        background-color: #00d66f;
        color: #000000;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        margin-top: 0.5rem;
    }
    
    .stForm button[type="submit"]:hover {
        background-color: #00ff80;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #2a2a2a;
        color: #ffffff;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #00d66f;
        color: #000000;
    }
    
    /* Success message */
    .success-message {
        background-color: #00d66f;
        color: #000000;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 2rem 0;
    }
    
    /* Content boxes */
    .content-box {
        background-color: #2a2a2a;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Sentiment badges */
    .sentiment-positive {
        background-color: #00d66f;
        color: #000000;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .sentiment-neutral {
        background-color: #666666;
        color: #ffffff;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .sentiment-negative {
        background-color: #dc143c;
        color: #ffffff;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    /* Loading states */
    .stProgress > div > div > div > div {
        background-color: #00d66f;
    }
    
    /* Input field */
    .stTextInput > div > div > input {
        background-color: #3a3a3a;
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Chat messages */
    .chat-message {
        background-color: #2a2a2a;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Chat input container */
    .stForm {
        background-color: transparent;
        border: none;
        margin-top: 1rem;
    }
    
    .stForm .stTextInput > div > div > input {
        background-color: #3a3a3a;
        color: #ffffff;
        border: 2px solid #555555;
        border-radius: 8px;
        padding: 1rem;
        font-size: 1rem;
    }
    
    .stForm .stTextInput > div > div > input:focus {
        border-color: #00d66f;
        outline: none;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'transcribed_text' not in st.session_state:
    st.session_state.transcribed_text = ""
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'sentiment_data' not in st.session_state:
    st.session_state.sentiment_data = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'conversion_complete' not in st.session_state:
    st.session_state.conversion_complete = False
if 'current_episode' not in st.session_state:
    st.session_state.current_episode = ""
# --- NEW: Session state for generated themes ---
if 'generated_themes' not in st.session_state:
    st.session_state.generated_themes = []

# Spotify API credentials (hardcoded as per requirement)
CLIENT_ID = 'b019bfe32e0e49a88aa64f330164fd2b'
CLIENT_SECRET = 'fc26fa6008434b499a57a0a8348575fd'

@st.cache_resource
def get_spotify_token():
    """Get Spotify access token"""
    auth_response = requests.post(
        'https://accounts.spotify.com/api/token',
        data={'grant_type': 'client_credentials'},
        auth=(CLIENT_ID, CLIENT_SECRET)
    )
    return auth_response.json().get('access_token')

@st.cache_resource
def load_whisper_model():
    """Load Whisper model (cached for performance)"""
    return whisper.load_model("base")

@st.cache_resource
def load_summarizer():
    """Load summarization model (cached for performance)"""
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource
def load_sentiment_analyzer():
    """Load sentiment analysis model"""
    return pipeline("sentiment-analysis")

@st.cache_resource
def load_embedding_model():
    """Load embedding model for chatbot"""
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_episode_preview_url(episode_url, access_token):
    """Extract episode ID and fetch preview URL from Spotify API"""
    try:
        episode_id = episode_url.split("/")[-1].split("?")[0]
        endpoint = f"https://api.spotify.com/v1/episodes/{episode_id}"
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.get(endpoint, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data.get("audio_preview_url"), data.get("name", "Unknown Episode")
    except Exception as e:
        st.error(f"Error getting preview URL: {e}")
        return None, None

def download_audio(url):
    """Download audio from URL to temporary file"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        tmpfile = NamedTemporaryFile(delete=False, suffix=".mp3")
        with open(tmpfile.name, 'wb') as f:
            f.write(response.content)
        return tmpfile.name
    except Exception as e:
        st.error(f"Error downloading audio: {e}")
        return None

def transcribe_audio(file_path):
    """Transcribe audio using Whisper"""
    try:
        model = load_whisper_model()
        transcription_result = model.transcribe(file_path)
        return transcription_result['text']
    except ModuleNotFoundError as e:
        if "numpy" in str(e).lower():
            st.error("❌ NumPy is not installed. Run: pip install numpy")
        else:
            st.error(f"❌ Missing module: {e}")
        return None
    except FileNotFoundError as e:
        if "ffmpeg" in str(e).lower():
            st.error("❌ FFmpeg is not installed. Please install FFmpeg on your system.")
            st.info("Windows: Download from https://www.gyan.dev/ffmpeg/builds/ and add to PATH")
            st.info("Mac: brew install ffmpeg")
            st.info("Linux: sudo apt install ffmpeg")
        else:
            st.error(f"❌ File not found: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Error during transcription: {e}")
        return None

def summarize_text(text, min_length=30, max_length=130):
    """Summarize text using BART"""
    try:
        summarizer = load_summarizer()
        summary = summarizer(text, min_length=min_length, max_length=max_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        st.error(f"Error during summarization: {e}")
        return "Summary unavailable"

def analyze_sentiment(text):
    """Analyze sentiment of text"""
    try:
        analyzer = load_sentiment_analyzer()
        # Split into sentences for better analysis
        sentences = text.split(". ")
        if not sentences or (len(sentences) == 1 and not sentences[0]):
             return {'positive': 0, 'negative': 0, 'neutral': 100}
        
        results = analyzer(sentences[:10])  # Analyze first 10 sentences
        
        # Count sentiments
        positive = sum(1 for r in results if r['label'] == 'POSITIVE')
        negative = sum(1 for r in results if r['label'] == 'NEGATIVE')
        neutral = len(results) - positive - negative
        
        total = len(results)
        if total == 0:
            return {'positive': 0, 'negative': 0, 'neutral': 100}
            
        return {
            'positive': int((positive / total) * 100),
            'negative': int((negative / total) * 100),
            'neutral': int((neutral / total) * 100)
        }
    except Exception as e:
        st.error(f"Error during sentiment analysis: {e}")
        return {'positive': 0, 'negative': 0, 'neutral': 100}

def ollama_generate(prompt, model_name="deepseek-r1:1.5b-qwen-distill-q8_0"):
    """Generate response using Ollama"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "options": {"temperature": 0.8, "top_p": 0.95, "max_tokens": 200}
            },
            stream=True,
            timeout=30
        )
        output = ""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode("utf-8"))
                    if "response" in data:
                        output += data["response"]
                except:
                    continue
        return output.strip()
    except Exception as e:
        return f"Error: {str(e)}. Make sure Ollama is running locally."

# --- NEW: Function to extract themes using Ollama ---
def extract_themes(text):
    """Extract key themes from text using Ollama"""
    # Prompt engineering is key here. We explicitly ask for a comma-separated list.
    prompt = f"""You are an assistant. Analyze the following transcript and identify the top 5-7 main themes. 
                 Return ONLY a comma-separated list of these themes. Do not add any other text, intro, or explanation.
                 Example: Personal Growth, Career Advice, Tech Trends
                 
                 Transcript: "{text}"
                 
                 Themes: """
    try:
        # Reuse the existing Ollama function
        theme_string = ollama_generate(prompt)
        
        # Clean up the output string and split it into a list
        themes_list = [theme.strip() for theme in theme_string.split(",") if theme.strip()]
        
        if not themes_list:
            return ["No themes found"]
            
        return themes_list
    except Exception as e:
        st.error(f"Error extracting themes: {e}")
        return ["Themes unavailable"] # Fallback

def chatbot_response(query, transcript):
    """Generate chatbot response using semantic search"""
    if not transcript:
        return "Please convert a podcast first before asking questions."
    
    try:
        embedding_model = load_embedding_model()
        
        # Chunk the transcript
        transcript_chunks = transcript.split(". ")
        chunk_embeddings = embedding_model.encode(transcript_chunks, convert_to_tensor=True)
        
        # Find relevant chunks
        query_embedding = embedding_model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, chunk_embeddings, top_k=3)
        relevant_chunks = " ".join([transcript_chunks[hit['corpus_id']] for hit in hits[0]])
        
        # Build prompt
        prompt = f"Context: {relevant_chunks}\n\nQuestion: {query}\nAnswer:"
        return ollama_generate(prompt)
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Main app layout
st.markdown('<div class="brand-name"><span class="brand-sound">Sound</span><span class="brand-scape">Scape</span></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Transform podcast into readable text</div>', unsafe_allow_html=True)

# Input section (starts directly without extra space)
st.markdown('<div class="input-container">', unsafe_allow_html=True)
st.markdown("### Enter Spotify Podcast URL")

col1, col2 = st.columns([4, 1])
with col1:
    spotify_url = st.text_input(
        "",
        placeholder="https://open.spotify.com/episode/...",
        label_visibility="collapsed"
    )
with col2:
    convert_button = st.button("Convert", use_container_width=True)

st.markdown("**Supported:** 30s to 1 minute spotify podcast text output")
st.markdown('</div>', unsafe_allow_html=True)

# Conversion process
if convert_button and spotify_url:
    with st.spinner("🔄 Fetching Episode..."):
        access_token = get_spotify_token()
        preview_url, episode_name = get_episode_preview_url(spotify_url, access_token)
        
        if preview_url:
            st.success(f"✓ Found: {episode_name}")
            
            # Store current episode name
            st.session_state.current_episode = episode_name
            
            with st.spinner("⬇️ Downloading audio..."):
                audio_file = download_audio(preview_url)
            
            if audio_file:
                with st.spinner("🎙️ Transcribing audio (this may take a moment)..."):
                    transcribed_text = transcribe_audio(audio_file)
                
                if transcribed_text:
                    st.session_state.transcribed_text = transcribed_text
                    
                    # Clear chat history for new podcast
                    st.session_state.chat_history = []
                    
                    with st.spinner("📝 Generating summary..."):
                        st.session_state.summary = summarize_text(transcribed_text)
                    
                    with st.spinner("💭 Analyzing sentiment..."):
                        st.session_state.sentiment_data = analyze_sentiment(transcribed_text)
                    
                    # --- NEW: Call the theme extraction function ---
                    with st.spinner("🌱 Extracting themes..."):
                        st.session_state.generated_themes = extract_themes(transcribed_text)
                    
                    st.session_state.conversion_complete = True
                    st.rerun()
        else:
            st.error("❌ No preview audio available or invalid URL")

# Show results if conversion is complete
if st.session_state.conversion_complete:
    st.markdown('<div class="success-message">✓ Conversion Complete<br>Your podcast has successfully been converted to text</div>', unsafe_allow_html=True)
    
    # Display current episode name if available
    if st.session_state.current_episode:
        st.caption(f"📻 Currently viewing: {st.session_state.current_episode}")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📝 TRANSCRIPT", "📊 SUMMARY", "😊 SENTIMENT", "💬 CHATBOT"])
    
    with tab1:
        st.markdown("### Full Transcript")
        if st.session_state.transcribed_text:
            st.markdown('<div class="content-box">', unsafe_allow_html=True)
            st.write(st.session_state.transcribed_text)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No transcript available yet.")
    
    with tab2:
        st.markdown("### Episode Summary")
        if st.session_state.summary:
            st.markdown('<div class="content-box">', unsafe_allow_html=True)
            st.write(st.session_state.summary)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No summary available yet.")
    
    with tab3:
        st.markdown("### OVERALL SENTIMENT")
        
        col1, col2, col3 = st.columns(3)
        
        # MODIFIED: Added a check for sentiment_data
        if st.session_state.sentiment_data:
            with col1:
                st.markdown(f'<div class="sentiment-positive">{st.session_state.sentiment_data["positive"]}% Positive</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f'<div class="sentiment-neutral">{st.session_state.sentiment_data["neutral"]}% Neutral</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown(f'<div class="sentiment-negative">{st.session_state.sentiment_data["negative"]}% Negative</div>', unsafe_allow_html=True)
        else:
            st.info("Sentiment data is not available.")

        st.markdown("### COMMON THEMES")
        
        # --- MODIFIED: Use the dynamically generated themes ---
        if st.session_state.generated_themes:
            # We'll stick to 5 columns for layout consistency
            cols = st.columns(5)
            for i, theme in enumerate(st.session_state.generated_themes):
                with cols[i % 5]:
                    # Using 'sentiment-neutral' for consistent styling
                    st.markdown(f'<div class="sentiment-neutral">{theme}</div>', unsafe_allow_html=True)
        else:
            st.info("No common themes were extracted from this episode.")
        # --- END OF MODIFICATION ---
            
    with tab4:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown("### Ask About This Episode")
        with col2:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        st.markdown("Hi! Ask me anything about this podcast episode.")
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f'<div class="chat-message">**You:** {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message">**Chatbot:** {message["content"]}</div>', unsafe_allow_html=True)
        
        # Chat input - using a form to prevent auto-rerun
        with st.form(key="chat_form", clear_on_submit=True):
            user_question = st.text_input("Ask a question...", key="chat_input", label_visibility="collapsed")
            submit_button = st.form_submit_button("Send")
            
            if submit_button and user_question:
                # Add user message
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                
                # Generate response
                with st.spinner("Thinking..."):
                    response = chatbot_response(user_question, st.session_state.transcribed_text)
                
                # Add bot response
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()