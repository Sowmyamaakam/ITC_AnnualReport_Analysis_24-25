# -*- coding: utf-8 -*-
import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from datetime import datetime
# Try to import plotly, fall back to matplotlib if not available
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("⚠️ Plotly not installed. Using matplotlib for visualizations. Install plotly with: pip install plotly")
# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass
STOP_WORDS = set(stopwords.words('english'))
# =============================
# PAGE CONFIGURATION
# =============================
st.set_page_config(
    page_title="Annual Report Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
# =============================
# CUSTOM CSS STYLING
# =============================
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --background-dark: #0f172a;
        --card-background: #1e293b;
    }
    
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    /* Custom card styling */
    .custom-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    /* Metric card styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Section headers */
    .section-header {
        color: #667eea;
        font-size: 1.8rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Info boxes */
    .insight-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(240, 147, 251, 0.3);
    }
    
    .positive-box {
        background: linear-gradient(135deg, #0ba360 0%, #3cba92 100%);
    }
    
    .negative-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
</style>
""", unsafe_allow_html=True)

# =============================
# HELPER FUNCTIONS
# =============================
def read_pdf_to_dataframe(pdf_file):
    """Extract text from PDF pages"""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    pages = [page.get_text("text") for page in doc]
    return pd.DataFrame({'Page_Number': range(1, len(pages)+1), 'Text': pages})
def clean_text(text):
    """Clean and preprocess text - keeps digits for financial analysis"""
    if not isinstance(text, str):
        return ''
    text = text.lower()
    # Remove special characters but keep alphanumeric (including digits)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return ' '.join(words)
def extractive_summary(text, num_sentences=5):
    sentences = [s for s in sent_tokenize(text) if 40 < len(s) < 300]  # avoid too short/long
    vectorizer = TfidfVectorizer(max_df=0.8, min_df=2, stop_words='english')
    X = vectorizer.fit_transform(sentences)
    scores = X.sum(axis=1).A1
    top = scores.argsort()[-num_sentences:][::-1]
    return [sentences[i] for i in sorted(top)] 
    try:
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(sentences)
        sentence_scores = X.sum(axis=1).A1
        top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
        return [sentences[i] for i in sorted(top_indices)]
    except:
        return sentences[:num_sentences]

def build_lda_model_gensim(cleaned_docs, num_topics=10):
    """Builds an LDA model using gensim"""
    try:
        from gensim.corpora.dictionary import Dictionary
        import gensim
        
        tokenized_docs = [doc.split() for doc in cleaned_docs if doc.strip()]
        if not tokenized_docs:
            return None, None, None
        
        dictionary = Dictionary(tokenized_docs)
        corpus = [dictionary.doc2bow(text) for text in tokenized_docs]
        
        if len(dictionary) == 0 or len(corpus) == 0:
            return None, None, None
        
        lda_model = gensim.models.LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=100,
            chunksize=100,
            passes=10,
            per_word_topics=True,
            workers=1
        )
        return lda_model, dictionary, corpus
    except:
        return None, None, None

def extract_key_metrics(text):
    """Extract percentages and currency with context"""
    # Capture small context around percentages
    percentage_contexts = re.findall(r'(\b\w{0,5}\s*\w{0,5}\s*\d+\.?\d*%)', text)
    currency_contexts = re.findall(r'(\$?\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:million|billion|trillion|INR|₹)?)', text, re.IGNORECASE)
    return percentage_contexts[:5], currency_contexts[:5]


# =============================
# MAIN APP
# =============================

# Header
st.markdown("""
<div class="main-header">
    <h1>📊 Annual Report Intelligence Platform</h1>
    <p>AI-Powered Financial Document Analysis & Insights</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Analysis Settings")
    
    analysis_options = st.multiselect(
        "Select Analysis Types",
        ["Sentiment Analysis", "Topic Modeling", "Word Cloud", "Summary Generation", 
         "Bigram Analysis", "Key Metrics Extraction", "Page-wise Analysis", "Frequent Words"],
        default=["Sentiment Analysis", "Summary Generation", "Topic Modeling", "Frequent Words"]
    )
    
    st.markdown("---")
    
    num_topics = st.slider("Number of Topics for LDA", 2, 15, 10)
    num_summary_sentences = st.slider("Summary Sentences", 3, 10, 5)
    
    st.markdown("---")
    st.markdown("### 📖 About")
    st.info("This tool uses NLP and ML techniques to analyze annual reports, extract insights, and identify key themes.")

# File uploader
uploaded_file = st.file_uploader("📁 Upload Annual Report (PDF)", type=["pdf"])

if uploaded_file:
    # Extract text
    with st.spinner("🔍 Extracting text from PDF..."):
        try:
            df = read_pdf_to_dataframe(uploaded_file)
            st.success(f"✅ Successfully extracted {len(df)} pages")
        except Exception as e:
            st.error(f"❌ Failed to read PDF: {e}")
            st.stop()
    
    # Clean text
    df['Cleaned_Text'] = df['Text'].astype(str).apply(clean_text)
    full_text = ' '.join(df['Text'].astype(str).tolist())
    
    # Quick Metrics Dashboard
    st.markdown('<div class="section-header">📈 Quick Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Pages</div>
            <div class="metric-value">{len(df)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        word_count = len(full_text.split())
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Words</div>
            <div class="metric-value">{word_count:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        sentences = sent_tokenize(full_text)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Sentences</div>
            <div class="metric-value">{len(sentences):,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_sentiment = TextBlob(full_text[:5000]).sentiment.polarity
        sentiment_emoji = "😊" if avg_sentiment > 0 else "😐" if avg_sentiment == 0 else "😟"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Sentiment</div>
            <div class="metric-value">{sentiment_emoji} {avg_sentiment:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    
    # Summary Generation
    if "Summary Generation" in analysis_options:
        st.markdown('<div class="section-header">📝 Executive Summary</div>', unsafe_allow_html=True)
        
        with st.spinner("Generating summary..."):
            summary_sentences = extractive_summary(full_text, num_summary_sentences)
            
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown("### 🎯 Key Highlights")
            for i, sent in enumerate(summary_sentences, 1):
                st.markdown(f"**{i}.** {sent}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Key Metrics Extraction
    if "Key Metrics Extraction" in analysis_options:
        st.markdown('<div class="section-header">💰 Key Financial Metrics</div>', unsafe_allow_html=True)
        
        percentages, currency = extract_key_metrics(full_text)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown("#### 📊 Percentages Found")
            if percentages:
                for pct in percentages:
                    st.markdown(f"- {pct}")
            else:
                st.info("No percentages found")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown("#### 💵 Currency Amounts")
            if currency:
                for curr in currency:
                    st.markdown(f"- {curr}")
            else:
                st.info("No currency amounts found")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Sentiment Analysis
    if "Sentiment Analysis" in analysis_options:
        st.markdown('<div class="section-header">😊 Sentiment Analysis</div>', unsafe_allow_html=True)
        
        with st.spinner("Analyzing sentiment..."):
            sentiment_data = []
            for s in sentences[:500]:  # Limit for performance
                polarity = TextBlob(s).sentiment.polarity
                if polarity > 0.05:
                    sentiment = 'Positive'
                elif polarity < -0.05:
                    sentiment = 'Negative'
                else:
                    sentiment = 'Neutral'
                sentiment_data.append((s, polarity, sentiment))
            
            if sentiment_data:
                sentiment_df = pd.DataFrame(sentiment_data, columns=['Sentence', 'Polarity', 'Sentiment'])
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                    summary = sentiment_df['Sentiment'].value_counts()
                    
                    if PLOTLY_AVAILABLE:
                        fig = go.Figure(data=[go.Pie(
                            labels=summary.index,
                            values=summary.values,
                            hole=.4,
                            marker_colors=['#0ba360', '#f5576c', '#667eea']
                        )])
                        fig.update_layout(
                            title="Sentiment Distribution",
                            height=400,
                            showlegend=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Fallback to matplotlib
                        fig, ax = plt.subplots(figsize=(8, 6))
                        colors = ['#0ba360', '#f5576c', '#667eea']
                        ax.pie(summary.values, labels=summary.index, autopct='%1.1f%%', 
                               colors=colors, startangle=90)
                        ax.set_title("Sentiment Distribution")
                        st.pyplot(fig)
                        plt.close(fig)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                    st.markdown("### 🌟 Most Positive Insights")
                    for sent in sentiment_df.sort_values(by='Polarity', ascending=False)['Sentence'].head(3):
                        st.markdown(f'<div class="positive-box">✓ {sent[:200]}...</div>', unsafe_allow_html=True)
                    
                    st.markdown("### ⚠️ Areas of Concern")
                    for sent in sentiment_df.sort_values(by='Polarity')['Sentence'].head(3):
                        st.markdown(f'<div class="negative-box">⚠ {sent[:200]}...</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
    
    # Word Cloud
    if "Word Cloud" in analysis_options:
        st.markdown('<div class="section-header">☁️ Word Cloud Visualization</div>', unsafe_allow_html=True)
        
        all_words = ' '.join(df['Cleaned_Text'].astype(str))
        if all_words.strip():
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            wc = WordCloud(width=1200, height=500, background_color='white', 
                          colormap='viridis', max_words=100).generate(all_words)
            fig, ax = plt.subplots(figsize=(15, 7))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)
            st.markdown('</div>', unsafe_allow_html=True)
    # Frequent Words
    if "Frequent Words" in analysis_options:
        st.markdown('<div class="section-header">📝 Frequent Words</div>', unsafe_allow_html=True)
        all_words = ' '.join(df['Cleaned_Text'].astype(str)).split()
        word_counts = Counter([w for w in all_words if w not in STOP_WORDS])
        most_common_words = word_counts.most_common(15)
        if most_common_words:
            freq_df = pd.DataFrame(most_common_words, columns=['Word', 'Frequency'])
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            if PLOTLY_AVAILABLE:
                fig = px.bar(freq_df, x='Word', y='Frequency', 
                            title='Top 15 Frequent Words', color='Frequency', color_continuous_scale='Viridis')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(freq_df['Word'], freq_df['Frequency'], color='#667eea')
                ax.set_title('Top 15 Frequent Words', fontsize=14, fontweight='bold')
                ax.set_ylabel('Frequency')
                ax.set_xticklabels(freq_df['Word'], rotation=45, ha='right')
                st.pyplot(fig)
                plt.close(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No frequent words found")

   
# Topic Modeling (Gensim LDA with Gibbs Sampling)
    if "Topic Modeling" in analysis_options:
        st.markdown('<div class="section-header">🧠 Topic Modeling (LDA with Gibbs Sampling)</div>', unsafe_allow_html=True)

        with st.spinner("Identifying topics using Gensim LDA..."):
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)

            try:
                # Build LDA model using Gensim
                lda_model, dictionary, corpus = build_lda_model_gensim(df['Cleaned_Text'], num_topics=num_topics)

                if lda_model is not None:
                    topics_data = []
                    for idx, topic in lda_model.print_topics(num_topics=num_topics, num_words=10):
                        topics_data.append({
                            "Topic": f"Topic {idx + 1}",
                            "Keywords": re.sub(r'[\*\+\d\."]+', '', topic)
                        })

                    topics_df = pd.DataFrame(topics_data)
                    st.dataframe(topics_df, use_container_width=True, hide_index=True)

                    # Optional: visualize topic weights
                    if PLOTLY_AVAILABLE:
                        fig = px.bar(topics_df, x="Topic", y=topics_df.index,
                                    orientation="h", title="LDA Topic Distribution",
                                    color_discrete_sequence=["#667eea"])
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ Not enough data for topic modeling.")
            except Exception as e:
                st.error(f"Topic modeling failed: {e}")

            st.markdown('</div>', unsafe_allow_html=True)

    
    # Bigram Analysis
    if "Bigram Analysis" in analysis_options:
        st.markdown('<div class="section-header">🔗 Key Phrases (Bigrams)</div>', unsafe_allow_html=True)
        
        try:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            bigram_vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words='english', max_features=20)
            X_bigrams = bigram_vectorizer.fit_transform(df['Cleaned_Text'])
            bigram_counts = X_bigrams.toarray().sum(axis=0)
            bigram_df = pd.DataFrame({
                'Phrase': bigram_vectorizer.get_feature_names_out(),
                'Frequency': bigram_counts
            }).sort_values('Frequency', ascending=False)
            
            if not bigram_df.empty:
                if PLOTLY_AVAILABLE:
                    fig = px.bar(bigram_df.head(15), x='Frequency', y='Phrase', 
                                orientation='h', title='Top 15 Key Phrases',
                                color='Frequency', color_continuous_scale='Viridis')
                    fig.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Fallback to matplotlib
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.barh(bigram_df['Phrase'].head(15), bigram_df['Frequency'].head(15), color='#667eea')
                    ax.invert_yaxis()
                    ax.set_title('Top 15 Key Phrases', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Frequency')
                    st.pyplot(fig)
                    plt.close(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Bigram analysis failed: {e}")
    
    # Page-wise Analysis
    if "Page-wise Analysis" in analysis_options:
        st.markdown('<div class="section-header">📄 Page-by-Page Analysis</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        page_sentiments = []
        for _, row in df.iterrows():
            polarity = TextBlob(str(row['Text'])).sentiment.polarity
            page_sentiments.append({
                'Page': row['Page_Number'],
                'Sentiment': polarity,
                'Word Count': len(str(row['Text']).split())
            })
        
        page_df = pd.DataFrame(page_sentiments)
        
        if PLOTLY_AVAILABLE:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=page_df['Page'], y=page_df['Sentiment'],
                                    mode='lines+markers', name='Sentiment',
                                    line=dict(color='#667eea', width=3)))
            fig.update_layout(title='Sentiment Flow Across Pages',
                             xaxis_title='Page Number',
                             yaxis_title='Sentiment Score',
                             height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback to matplotlib
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(page_df['Page'], page_df['Sentiment'], marker='o', 
                   color='#667eea', linewidth=2, markersize=6)
            ax.set_title('Sentiment Flow Across Pages', fontsize=14, fontweight='bold')
            ax.set_xlabel('Page Number')
            ax.set_ylabel('Sentiment Score')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    
    
    st.success("✅ Analysis Complete! Scroll up to view all insights.")

else:
    st.markdown("""
    <div class="custom-card" style="text-align: center; padding: 3rem;">
        <h2>👆 Get Started</h2>
        <p style="font-size: 1.2rem; color: #666;">
            Upload a company's annual report (PDF) to unlock powerful AI-driven insights including:
        </p>
        <ul style="text-align: left; max-width: 600px; margin: 2rem auto; font-size: 1.1rem;">
            <li>📊 Sentiment analysis across the entire document</li>
            <li>🎯 AI-generated executive summaries</li>
            <li>🧠 Topic modeling and theme identification</li>
            <li>💰 Automatic extraction of key financial metrics</li>
            <li>📈 Visual analytics and word clouds</li>
            <li>🔗 Key phrase and bigram analysis</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)