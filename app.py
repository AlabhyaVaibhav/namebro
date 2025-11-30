"""
Namebro - Streamlit UI
Neo-Brutalist Design

AI-powered startup name generation and evaluation tool.
"""

import streamlit as st
from startup_name_agent import StartupNameAgent, NameStyle, NameEvaluation
import os

# Neo-Brutalist CSS with New Color Palette
st.markdown("""
<style>
    /* Color Palette Variables */
    :root {
        --red: #E63946;
        --cream: #F1FAEE;
        --yellow: #F1C40F;
        --light-grey: #D8E2DC;
        --light-blue: #A8DADC;
        --blue: #457B9D;
        --dark-blue: #1D3557;
        --yellow-orange: #F9C74F;
        --orange: #F3722C;
        --teal: #2A9D8F;
        --black: #000000;
        --text-dark: #1D3557;
        --text-light: #F1FAEE;
    }
    
    /* Neo-Brutalist Styling */
    .main {
        background-color: var(--cream);
    }
    
    .stApp {
        background-color: var(--cream);
        font-family: 'Courier New', monospace;
        color: var(--text-dark);
    }
    
    /* Headers with bold borders - High contrast for readability */
    h1, h2, h3 {
        font-family: 'Courier New', monospace;
        font-weight: 900;
        text-transform: uppercase;
        border: 4px solid var(--black);
        padding: 10px;
        background-color: var(--red);
        color: var(--text-light);
        box-shadow: 8px 8px 0px 0px var(--black);
        margin: 20px 0;
        display: block;
        width: fit-content;
        min-width: 200px;
    }
    
    h2 {
        background-color: var(--teal);
        font-size: 1.5em;
        color: var(--text-light);
    }
    
    h3 {
        background-color: var(--orange);
        font-size: 1.2em;
        color: var(--text-light);
    }
    
    /* Buttons - High contrast */
    .stButton > button {
        background-color: var(--red);
        color: var(--text-light);
        border: 4px solid var(--black);
        border-radius: 0;
        font-weight: 900;
        font-family: 'Courier New', monospace;
        text-transform: uppercase;
        padding: 10px 20px;
        box-shadow: 6px 6px 0px 0px var(--black);
        transition: all 0.1s;
    }
    
    .stButton > button:hover {
        box-shadow: 3px 3px 0px 0px var(--black);
        transform: translate(3px, 3px);
        background-color: var(--orange);
    }
    
    /* Text areas and inputs - Light background with dark text for readability */
    .stTextArea > div > div > textarea,
    .stTextInput > div > div > input {
        border: 4px solid var(--black);
        border-radius: 0;
        background-color: var(--text-light);
        color: var(--text-dark) !important;
        font-family: 'Courier New', monospace;
        box-shadow: 4px 4px 0px 0px var(--black);
    }
    
    /* Specific text area styling - ensure dark text on light background */
    textarea[id*="text_area"],
    textarea[data-testid*="text_area"],
    textarea#text_area_2,
    textarea[aria-label*="ENTER NAMES"] {
        color: var(--text-dark) !important;
        background-color: var(--text-light) !important;
    }
    
    /* All textarea elements should have dark text */
    textarea {
        color: var(--text-dark) !important;
    }
    
    /* Labels for text areas - dark text for readability */
    .stTextArea label,
    .stTextInput label {
        color: var(--text-dark) !important;
        font-weight: 900;
    }
    
    /* All text in light background elements should be dark */
    .stTextArea,
    .stTextInput,
    .stSelectbox,
    .stNumberInput {
        color: var(--text-dark) !important;
    }
    
    /* Text in light background containers */
    div[style*="background-color: #FFF"],
    div[style*="background-color: white"],
    .score-box {
        color: var(--text-dark) !important;
    }
    
    /* Ensure all text in light backgrounds is visible */
    .stMarkdown p,
    .stMarkdown div {
        color: var(--text-dark);
    }
    
    /* Light background sections */
    section[data-testid="stSidebar"] .stMarkdown,
    .element-container .stMarkdown {
        color: var(--text-dark);
    }
    
    /* Select boxes - light background, dark text */
    .stSelectbox > div > div {
        border: 4px solid var(--black);
        border-radius: 0;
        background-color: var(--text-light);
        color: var(--text-dark);
        font-family: 'Courier New', monospace;
    }
    
    /* Sidebar - cream background */
    .css-1d391kg {
        background-color: var(--cream);
        border-right: 4px solid var(--black);
    }
    
    /* Expanders - teal background with light text */
    .streamlit-expanderHeader {
        background-color: var(--teal);
        border: 4px solid var(--black);
        color: var(--text-light);
        font-family: 'Courier New', monospace;
        font-weight: 900;
        box-shadow: 4px 4px 0px 0px var(--black);
    }
    
    /* Info boxes - light background, dark text */
    .stAlert {
        border: 4px solid var(--black);
        border-radius: 0;
        box-shadow: 4px 4px 0px 0px var(--black);
        font-family: 'Courier New', monospace;
        background-color: var(--light-blue);
        color: var(--text-dark);
    }
    
    .stInfo {
        background-color: var(--light-blue);
        color: var(--text-dark);
    }
    
    .stError {
        background-color: var(--red);
        color: var(--text-light);
    }
    
    .stSuccess {
        background-color: var(--teal);
        color: var(--text-light);
    }
    
    /* Metrics - yellow background, dark text */
    .stMetric {
        background-color: var(--yellow);
        border: 4px solid var(--black);
        padding: 10px;
        box-shadow: 4px 4px 0px 0px var(--black);
        font-family: 'Courier New', monospace;
        color: var(--text-dark);
    }
    
    /* Score boxes - light backgrounds with dark text for readability */
    .score-box {
        background-color: var(--text-light);
        border: 4px solid var(--black);
        padding: 15px;
        margin: 10px 0;
        box-shadow: 6px 6px 0px 0px var(--black);
        font-family: 'Courier New', monospace;
        color: var(--text-dark) !important;
    }
    
    /* Score box color variations with high contrast text */
    .score-high {
        background-color: var(--light-blue);
        color: var(--text-dark) !important;
    }
    
    .score-medium {
        background-color: var(--yellow);
        color: var(--text-dark) !important;
    }
    
    .score-low {
        background-color: var(--red);
        color: var(--text-light) !important;
    }
    
    /* Ensure header text is visible and inside box */
    h1 *, h2 *, h3 * {
        color: inherit !important;
    }
    
    /* Streamlit specific overrides for text colors - prioritize readability */
    .stTextArea > div > label,
    .stTextInput > div > label,
    .stNumberInput > div > label {
        color: var(--text-dark) !important;
        font-weight: 900;
    }
    
    /* Ensure all input text is dark */
    input[type="text"],
    input[type="number"],
    textarea {
        color: white !important;
    }
    
    /* Placeholder text should be medium grey for readability */
    ::placeholder {
        color: var(--blue) !important;
        opacity: 0.7;
    }
    
    /* All text in Streamlit containers with light backgrounds */
    [data-testid="stTextArea"] label,
    [data-testid="stTextInput"] label,
    [data-testid="stNumberInput"] label {
        color: var(--text-dark) !important;
    }
    
    /* Ensure markdown text in light areas is dark */
    .element-container .stMarkdown,
    .element-container .stMarkdown p,
    .element-container .stMarkdown strong {
        color: var(--text-dark);
    }
    
    /* File uploader text */
    .stFileUploader label {
        color: var(--text-dark) !important;
    }
    
    /* Tabs - high contrast */
    .stTabs [data-baseweb="tab-list"] {
        background-color: var(--light-grey);
        border-bottom: 4px solid var(--black);
    }
    
    .stTabs [data-baseweb="tab"] {
        color: var(--text-dark);
        font-weight: 900;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--blue);
        color: var(--text-light);
    }
    
    /* General text color override for readability */
    body, .stApp {
        color: var(--text-dark);
    }
    
    /* Strong/bold text should be darker for emphasis */
    strong, b {
        color: var(--dark-blue);
    }
</style>
""", unsafe_allow_html=True)

# Page configuration
st.set_page_config(
    page_title="Namebro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "context" not in st.session_state:
    st.session_state.context = ""
if "styles" not in st.session_state:
    st.session_state.styles = []
if "generated_names" not in st.session_state:
    st.session_state.generated_names = []


def initialize_agent(llm_provider: str, model_name: str = None):
    """Initialize the agent with selected LLM provider"""
    try:
        return StartupNameAgent(
            llm_provider=llm_provider,
            model_name=model_name,
            temperature=0.9
        )
    except ValueError as e:
        st.error(f"ERROR: {e}")
        return None


def get_score_color(score: int) -> str:
    """Get color class based on score"""
    if score >= 8:
        return "score-high"
    elif score >= 5:
        return "score-medium"
    else:
        return "score-low"


def main():
    st.markdown("<h1>NAMEBRO</h1><br>AI-POWERED NAME GENERATION & EVALUATION<br><br>", unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown("<h2>⚙️ CONFIGURATION</h2>", unsafe_allow_html=True)
        
        # LLM Provider selection
        llm_provider = st.selectbox(
            "LLM PROVIDER",
            ["openai", "anthropic"],
            help="Choose between OpenAI (ChatGPT) or Anthropic (Claude)"
        )
        
        # Model selection based on provider
        if llm_provider == "openai":
            model_name = st.selectbox(
                "MODEL",
                ["gpt-4-turbo-preview", "gpt-4", "gpt-3.5-turbo"],
                help="Select the OpenAI model to use"
            )
        else:
            model_name = st.selectbox(
                "MODEL",
                ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
                help="Select the Anthropic model to use"
            )
        
        # Initialize agent button
        if st.button("INITIALIZE AGENT", type="primary", use_container_width=True):
            with st.spinner("Initializing..."):
                st.session_state.agent = initialize_agent(llm_provider, model_name)
                if st.session_state.agent:
                    st.success("✅ AGENT READY!")
        
        st.markdown("---")
        
        # Style selection
        st.markdown("<h3>🎨 STYLE PREFERENCES</h3>", unsafe_allow_html=True)
        available_styles = [style.value for style in NameStyle]
        selected_styles = st.multiselect(
            "SELECT STYLES",
            available_styles,
            default=["professional"],
            help="Choose one or more style preferences"
        )
        st.session_state.styles = selected_styles
        
        st.markdown("---")
        
        # API Key info
        st.info("💡 Set API keys in .env file:\nOPENAI_API_KEY or\nANTHROPIC_API_KEY\n\n🌐 For domain checking:\nGODADDY_API_KEY\nGODADDY_API_SECRET\nGODADDY_USE_OTE=true\n\n🔍 For Google search:\nGOOGLE_API_KEY\nGOOGLE_SEARCH_ENGINE_ID")
    
    # Tabs for different sections
    tab1, tab2 = st.tabs(["🔮 GENERATE NAMES ", " 📊 EVALUATE NAMES "])
    
    with tab1:
        # Context input area
        st.markdown("<h2>📝 STARTUP CONTEXT</h2><br>", unsafe_allow_html=True)
        context_input = st.text_area(
            "DESCRIBE YOUR STARTUP:",
            value=st.session_state.context,
            height=150,
            help="Provide detailed context about your startup"
        )
        st.session_state.context = context_input
        
        # File upload option
        uploaded_file = st.file_uploader(
            "OR UPLOAD A FILE",
            type=["txt", "md"],
            help="Upload a text file with your startup description"
        )
        
        if uploaded_file is not None:
            file_content = uploaded_file.read().decode("utf-8")
            st.session_state.context = file_content
            st.text_area("FILE CONTENT:", value=file_content, height=100, disabled=True)
        
        # Generate names section
        col1, col2 = st.columns([1, 4])
        with col1:
            num_suggestions = st.number_input("NUMBER", min_value=1, max_value=20, value=10)
        
        if st.button("✨ GENERATE NAMES", type="primary", use_container_width=True):
            if not st.session_state.agent:
                st.error("❌ INITIALIZE AGENT FIRST!")
            elif not st.session_state.context.strip():
                st.error("❌ PROVIDE STARTUP CONTEXT!")
            elif not st.session_state.styles:
                st.error("❌ SELECT AT LEAST ONE STYLE!")
            else:
                with st.spinner("GENERATING CREATIVE NAMES..."):
                    try:
                        # Load context
                        st.session_state.agent.load_context_from_text(st.session_state.context)
                        st.session_state.agent.set_style_parameters(st.session_state.styles)
                        
                        # Generate names
                        suggestions = st.session_state.agent.generate_names(num_suggestions=num_suggestions)
                        st.session_state.generated_names = [s.name for s in suggestions]
                        
                        st.markdown("<h2>💡 NAME SUGGESTIONS</h2>", unsafe_allow_html=True)
                        
                        for i, suggestion in enumerate(suggestions, 1):
                            with st.expander(f"**{suggestion.name.upper()}** ({suggestion.style.upper()})", expanded=(i <= 3)):
                                st.markdown(f"**STYLE:** {suggestion.style.upper()}")
                                
                                if suggestion.variations:
                                    st.markdown("**VARIATIONS:**")
                                    variations_text = " | ".join([f"`{v}`" for v in suggestion.variations[:10]])
                                    st.markdown(variations_text)
                                
                                if suggestion.reasoning:
                                    st.markdown("**REASONING:**")
                                    st.info(suggestion.reasoning)
                        
                    except Exception as e:
                        st.error(f"ERROR: {str(e)}")
    
    with tab2:
        st.markdown("<h2>📊 EVALUATE STARTUP NAMES</h2><br>", unsafe_allow_html=True)
        st.markdown("**ENTER NAMES TO EVALUATE ACROSS 12 PARAMETERS**")
        
        # Name input
        names_input = st.text_area(
            "ENTER NAMES (ONE PER LINE OR COMMA-SEPARATED):",
            height=100,
            help="Enter the names you want to evaluate"
        )
        
        # Use generated names if available
        if st.session_state.generated_names:
            if st.button("USE GENERATED NAMES", use_container_width=True):
                names_input = "\n".join(st.session_state.generated_names)
        
        if st.button("🔍 EVALUATE NAMES", type="primary", use_container_width=True):
            if not st.session_state.agent:
                st.error("❌ INITIALIZE AGENT FIRST!")
            elif not names_input.strip():
                st.error("❌ ENTER NAMES TO EVALUATE!")
            else:
                # Parse names
                names = []
                for line in names_input.split('\n'):
                    line = line.strip()
                    if ',' in line:
                        names.extend([n.strip() for n in line.split(',') if n.strip()])
                    elif line:
                        names.append(line)
                
                if not names:
                    st.error("❌ NO VALID NAMES FOUND!")
                else:
                    with st.spinner("EVALUATING NAMES..."):
                        try:
                            evaluations = st.session_state.agent.evaluate_names(
                                names=names,
                                startup_context=st.session_state.context if st.session_state.context else None
                            )
                            
                            # Sort by total score
                            evaluations.sort(key=lambda x: x.total_score, reverse=True)
                            
                            st.markdown("<h2>📈 EVALUATION RESULTS</h2>", unsafe_allow_html=True)
                            
                            # Summary metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("BEST SCORE", f"{evaluations[0].total_score}/120", evaluations[0].name.upper())
                            with col2:
                                avg_score = sum(e.total_score for e in evaluations) / len(evaluations)
                                st.metric("AVG SCORE", f"{int(avg_score)}/120", "AVERAGE")
                            with col3:
                                st.metric("NAMES EVALUATED", len(evaluations), "TOTAL")
                            
                            # Detailed evaluations
                            for i, eval_result in enumerate(evaluations, 1):
                                st.markdown(f"### {i}. {eval_result.name.upper()}")
                                
                                # Score grid
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.markdown(f'<div class="score-box {get_score_color(eval_result.memorability)}">'
                                              f'<strong>MEMORABILITY</strong><br>{eval_result.memorability}/10</div>',
                                              unsafe_allow_html=True)
                                    st.markdown(f'<div class="score-box {get_score_color(eval_result.pronounceability)}">'
                                              f'<strong>PRONOUNCEABILITY</strong><br>{eval_result.pronounceability}/10</div>',
                                              unsafe_allow_html=True)
                                
                                with col2:
                                    st.markdown(f'<div class="score-box {get_score_color(eval_result.spelling_simplicity)}">'
                                              f'<strong>SPELLING</strong><br>{eval_result.spelling_simplicity}/10</div>',
                                              unsafe_allow_html=True)
                                    st.markdown(f'<div class="score-box {get_score_color(eval_result.meaning_associations)}">'
                                              f'<strong>MEANING</strong><br>{eval_result.meaning_associations}/10</div>',
                                              unsafe_allow_html=True)
                                
                                with col3:
                                    st.markdown(f'<div class="score-box {get_score_color(eval_result.phonetic_appeal)}">'
                                              f'<strong>PHONETICS</strong><br>{eval_result.phonetic_appeal}/10</div>',
                                              unsafe_allow_html=True)
                                    st.markdown(f'<div class="score-box {get_score_color(eval_result.originality)}">'
                                              f'<strong>ORIGINALITY</strong><br>{eval_result.originality}/10</div>',
                                              unsafe_allow_html=True)
                                
                                with col4:
                                    st.markdown(f'<div class="score-box {get_score_color(eval_result.category_fit)}">'
                                              f'<strong>CATEGORY FIT</strong><br>{eval_result.category_fit}/10</div>',
                                              unsafe_allow_html=True)
                                    st.markdown(f'<div class="score-box {get_score_color(eval_result.domain_availability)}">'
                                              f'<strong>DOMAIN</strong><br>{eval_result.domain_availability}/10</div>',
                                              unsafe_allow_html=True)
                                
                                # Second row of scores
                                col5, col6, col7, col8 = st.columns(4)
                                
                                with col5:
                                    st.markdown(f'<div class="score-box {get_score_color(eval_result.global_safety)}">'
                                              f'<strong>GLOBAL SAFETY</strong><br>{eval_result.global_safety}/10</div>',
                                              unsafe_allow_html=True)
                                with col6:
                                    st.markdown(f'<div class="score-box {get_score_color(eval_result.visual_identity)}">'
                                              f'<strong>VISUAL</strong><br>{eval_result.visual_identity}/10</div>',
                                              unsafe_allow_html=True)
                                with col7:
                                    st.markdown(f'<div class="score-box {get_score_color(eval_result.longevity)}">'
                                              f'<strong>LONGEVITY</strong><br>{eval_result.longevity}/10</div>',
                                              unsafe_allow_html=True)
                                with col8:
                                    st.markdown(f'<div class="score-box {get_score_color(eval_result.emotional_resonance)}">'
                                              f'<strong>EMOTION</strong><br>{eval_result.emotional_resonance}/10</div>',
                                              unsafe_allow_html=True)
                                
                                # Total score
                                st.markdown(f'<div class="score-box" style="background-color: #E63946; color: #F1FAEE; text-align: center; font-size: 1.5em;">'
                                          f'<strong>TOTAL SCORE: {eval_result.total_score}/120</strong></div>',
                                          unsafe_allow_html=True)
                                
                                # Strengths and Weaknesses
                                col9, col10 = st.columns(2)
                                with col9:
                                    st.markdown("**✅ STRENGTHS:**")
                                    for strength in eval_result.strengths:
                                        st.markdown(f"- {strength}")
                                
                                with col10:
                                    st.markdown("**❌ WEAKNESSES:**")
                                    for weakness in eval_result.weaknesses:
                                        st.markdown(f"- {weakness}")
                                
                                # Domain Availability Section
                                if eval_result.available_domains:
                                    st.markdown("**🌐 AVAILABLE DOMAINS:**")
                                    domain_cols = st.columns(min(5, len(eval_result.available_domains)))
                                    for idx, domain in enumerate(eval_result.available_domains[:5]):
                                        with domain_cols[idx % len(domain_cols)]:
                                            st.markdown(f'<div style="background-color: #2A9D8F; color: #F1FAEE; padding: 8px; border: 2px solid #000; box-shadow: 3px 3px 0px 0px #000; font-family: Courier New, monospace; font-weight: bold; text-align: center;">{domain}</div>', unsafe_allow_html=True)
                                
                                if eval_result.domain_check_error:
                                    st.warning(f"⚠️ Domain Check: {eval_result.domain_check_error}")
                                
                                # Google Search Insights Section
                                if eval_result.search_insights:
                                    st.markdown("**🔍 GOOGLE SEARCH INSIGHTS:**")
                                    st.markdown(f'<div style="background-color: #A8DADC; color: #1D3557; padding: 12px; border: 3px solid #000; box-shadow: 4px 4px 0px 0px #000; font-family: Courier New, monospace; margin: 10px 0;">{eval_result.search_insights}</div>', unsafe_allow_html=True)
                                    
                                    if eval_result.competing_domains:
                                        st.markdown("**Competing Domains Found:**")
                                        domain_list = " | ".join([f"`{d}`" for d in eval_result.competing_domains[:8]])
                                        st.markdown(domain_list)
                                    
                                    if eval_result.search_categories:
                                        st.markdown("**Categories/Areas:**")
                                        category_list = " | ".join([f"`{c}`" for c in eval_result.search_categories[:8]])
                                        st.markdown(category_list)
                                
                                if eval_result.search_error:
                                    st.warning(f"⚠️ Search Error: {eval_result.search_error}")
                                
                                # Recommendation
                                st.markdown("**💡 RECOMMENDATION:**")
                                st.info(eval_result.recommendation)
                                
                                st.markdown("---")
                            
                        except Exception as e:
                            st.error(f"ERROR: {str(e)}")


if __name__ == "__main__":
    main()
