import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from youtube_transcript_api import YouTubeTranscriptApi
from googletrans import Translator, LANGUAGES

# Set page title and favicon
st.set_page_config(
    page_title="YouTube Video Summarizer and Translator",
    page_icon=":clapper:",
    layout="wide"
)

# Title of the app with styling
st.title("YouTube Video Summarizer and Translator")

# Sidebar with header and description
st.sidebar.title("Instructions")
st.sidebar.markdown(
    """
    1. Enter the YouTube video URL.
    2. Watch the video and read the summarized text.
    3. Translate the summarized text to your preferred language.
    """
)

# Input URL
youtube_video = st.text_input("Enter the YouTube video URL:")

if youtube_video:
    # Extract video ID
    video_id = youtube_video.split("=")[-1]

    # Display the video
    st.video(youtube_video)

    # Fetch transcript
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        result = " ".join([i['text'] for i in transcript])

        # Initialize tokenizer and model
        model_name = "sshleifer/distilbart-cnn-12-6"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Summarize transcript
        summarizer = pipeline('summarization', model=model, tokenizer=tokenizer)

        # Batch size for summarization
        batch_size = 1000  # Number of characters per batch
        summarized_text = []

        # Process transcript in batches
        for i in range(0, len(result), batch_size):
            chunk = result[i:i + batch_size]
            if chunk:
                out = summarizer(chunk)
                out = out[0]['summary_text']
                summarized_text.append(out)

        summarized_text = " ".join(summarized_text)

        # Display summarized text with styled subheader
        st.subheader("Summarized Text")
        st.markdown(f"> {summarized_text}")

        # Translate summarized text
        translator = Translator()
        langcode = st.selectbox("Select the language you want to translate to:",
                                list(LANGUAGES.values()), index=21)

        if st.button("Translate"):
            translated_text = translator.translate(summarized_text, dest=langcode)
            st.subheader("Translated Text")
            st.markdown(f"> {translated_text.text}")

    except Exception as e:
        st.error(f"Error fetching transcript: {e}")


