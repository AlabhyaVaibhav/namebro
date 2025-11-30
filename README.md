# Namebro

**Namebro** is an **LLM-powered** intelligent agent that generates creative startup name suggestions based on context and customizable style parameters. Built with LangChain, Pydantic, and featuring a beautiful neo-brutalist Streamlit interface.

## Features

- **🤖 LLM-Powered**: Uses ChatGPT (OpenAI) or Claude (Anthropic) for intelligent name generation
- **📝 Context Input**: Accept context from files or pasted text
- **🎨 Style Parameters**: Generate names in various styles (GenZ, abstract artistic, professional, etc.)
- **✨ Name Variations**: Automatically generates creative variations of suggested names
- **📊 Name Evaluation**: Comprehensive evaluation across 12 parameters (memorability, pronounceability, etc.)
- **🌐 Domain Availability**: Real-time domain checking via GoDaddy Domains API with creative variations (get<name>.app, <name>app.ai, etc.)
- **🔍 Google Search Insights**: Searches Google to find existing applications, competing domains, and market categories
- **💬 Neo-Brutalist UI**: Beautiful, bold Streamlit interface with high-contrast design
- **📚 Rulebook Integration**: Uses proven naming principles and evaluation frameworks
- **🔧 Extensible**: Ready for additional custom tools (trademark search, etc.)

## Installation

```bash
# Clone or navigate to the project directory
cd namebro

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
# Create a .env file with your API keys:
# OPENAI_API_KEY=your_key_here
# or
# ANTHROPIC_API_KEY=your_key_here
```

## Quick Start

### Using the Streamlit UI (Recommended)

```bash
python3 -m streamlit run app.py
```

This will launch a web interface where you can:
- Select your LLM provider (OpenAI or Anthropic)
- Choose model and style preferences
- Enter startup context
- Generate and view name suggestions interactively

### Using Python API

```python
from startup_name_agent import StartupNameAgent, NameStyle

# Initialize agent with OpenAI (ChatGPT)
agent = StartupNameAgent(llm_provider="openai", model_name="gpt-4-turbo-preview")

# Or use Anthropic (Claude)
# agent = StartupNameAgent(llm_provider="anthropic", model_name="claude-3-opus-20240229")

# Load context
agent.load_context_from_text("""
    We're building a platform for remote team collaboration.
    It's modern, intuitive, and designed for distributed teams.
""")

# Set style parameters
agent.set_style_parameters([NameStyle.GENZ, NameStyle.TECH_MODERN])

# Generate name suggestions
suggestions = agent.generate_names(num_suggestions=10)

# Display results
for suggestion in suggestions:
    print(f"Name: {suggestion.name}")
    print(f"Style: {suggestion.style}")
    print(f"Variations: {', '.join(suggestion.variations)}")
    print(f"Reasoning: {suggestion.reasoning}")
    print()
```

### Convenience Method

```python
agent = StartupNameAgent(llm_provider="openai")

# All-in-one method
suggestions = agent.brainstorm(
    context="Your startup description here...",
    styles=["genz", "tech_modern"],
    num_suggestions=15
)
```

## Style Parameters

Available style options:

- `genz` - GenZ-friendly names with trendy suffixes
- `abstract_artistic` - Poetic and abstract combinations
- `professional` - Clean, business-oriented names
- `playful` - Fun and creative names
- `tech_modern` - Modern tech startup names (io, labs, ai suffixes)
- `minimalist` - Short, clean, minimal names
- `bold_edgy` - Bold and edgy names
- `classic` - Traditional, timeless names

## Environment Setup

Create a `.env` file in the project root:

```env
# OpenAI API Key (for ChatGPT)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API Key (for Claude)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# GoDaddy API Credentials (for domain availability checking)
GODADDY_API_KEY=your_godaddy_api_key_here
GODADDY_API_SECRET=your_godaddy_api_secret_here
GODADDY_USE_OTE=true  # Set to false for production

# Google Custom Search API (for competitive research)
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id_here
```

**Note**: 
- You only need one LLM API key (OpenAI or Anthropic), depending on which provider you want to use.
- GoDaddy API credentials are optional but recommended for real-time domain availability checking.
- Google Custom Search API is optional but recommended for competitive insights. Falls back to alternative search if not configured.

## Example

```python
from startup_name_agent import StartupNameAgent, NameStyle

# Initialize with OpenAI
agent = StartupNameAgent(llm_provider="openai", model_name="gpt-4-turbo-preview")

context = """
We're building an AI-powered fitness app that personalizes workouts
based on user goals and provides real-time form feedback using computer vision.
Target audience: Gen Z and millennials who want tech-driven fitness solutions.
"""

agent.load_context_from_text(context)
agent.set_style_parameters([NameStyle.GENZ, NameStyle.TECH_MODERN])

suggestions = agent.generate_names(num_suggestions=5)

for i, suggestion in enumerate(suggestions, 1):
    print(f"{i}. {suggestion.name} ({suggestion.style})")
    print(f"   Variations: {', '.join(suggestion.variations[:3])}")
    print(f"   Reasoning: {suggestion.reasoning}")
```

## Architecture

The agent is built with modern Python tools:

- **LangChain**: LLM integration and prompt management
- **Pydantic**: Data validation and structured outputs
- **Streamlit**: Interactive web UI
- **Structured Outputs**: Uses Pydantic models for reliable parsing

### Key Components

- `StartupNameAgent`: Main agent class with LLM integration
- `NameSuggestion`: Pydantic model for name suggestions
- `NameEvaluation`: Pydantic model for name evaluations
- `NameStyle`: Enum for style parameters
- `app.py`: Streamlit neo-brutalist UI interface
- `rulebook.txt`: Naming principles and guidelines
- `evaluation_rulebook.txt`: Comprehensive evaluation framework

## Supported LLM Providers

### OpenAI (ChatGPT)
- Models: `gpt-4-turbo-preview`, `gpt-4`, `gpt-3.5-turbo`
- Requires: `OPENAI_API_KEY` environment variable

### Anthropic (Claude)
- Models: `claude-3-opus-20240229`, `claude-3-sonnet-20240229`, `claude-3-haiku-20240307`
- Requires: `ANTHROPIC_API_KEY` environment variable

## Features in Detail

### Name Generation
- Context-aware name suggestions based on your startup description
- Multiple style options (GenZ, abstract artistic, professional, etc.)
- Automatic name variations for each suggestion
- Integration with proven naming rulebooks

### Name Evaluation
- Comprehensive 12-parameter evaluation framework
- Scores for memorability, pronounceability, spelling simplicity, and more
- Strengths and weaknesses analysis
- Comparative analysis across multiple names
- Total score out of 120 points

## Domain Availability Checking

Namebro now includes **real-time domain availability checking** via the [GoDaddy Domains API](https://developer.godaddy.com/doc/endpoint/domains). When evaluating names, the system:

1. **Generates Creative Domain Variations**: Automatically creates variations like:
   - `name.com`, `name.io`, `name.app`, `name.ai`
   - `getname.com`, `tryname.com`, `usename.com`
   - `nameapp.com`, `nameai.com`, `namehub.com`
   - `getnameapp.com`, `trynameai.com`, and more

2. **Checks Availability**: Uses GoDaddy API to check which domains are actually available

3. **Scores Domain Potential**: Calculates a domain availability score (1-10) based on:
   - Number of available domains
   - Quality of TLDs (prioritizes .com, .io, .app, .ai)
   - Domain length (shorter is better)

4. **Displays Results**: Shows top available domains in the evaluation results

### Setting Up GoDaddy API

1. Sign in to your GoDaddy account and go to [API Key Management](https://developer.godaddy.com/keys)
2. Create a new API key (choose OTE for testing or Production for live use)
3. Copy your API Key and Secret
4. Add them to your `.env` file:
   ```env
   GODADDY_API_KEY=your_key_here
   GODADDY_API_SECRET=your_secret_here
   GODADDY_USE_OTE=true  # Use false for production
   ```

The domain checker works automatically when API credentials are configured. If not configured, the system will still evaluate names but use LLM-estimated domain availability scores.

## Google Search Insights

Namebro includes **Google search integration** to provide competitive insights when evaluating startup names. The system:

1. **Searches for Existing Applications**: Queries Google for the startup name to find:
   - Existing companies/applications with similar names
   - Competing domains and websites
   - Market categories and areas where the name is used

2. **Analyzes Competition**: 
   - Counts number of competing results
   - Identifies domains using similar names
   - Categorizes the competitive landscape (Software/App, AI/ML, E-commerce, etc.)

3. **Updates Evaluation Scores**:
   - Adjusts originality score based on competition level
   - Adds competitive insights to strengths/weaknesses
   - Includes search findings in recommendations

4. **Displays Results**: Shows search insights, competing domains, and categories in evaluation results

### Setting Up Google Custom Search API

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the "Custom Search API"
4. Create credentials (API Key)
5. Set up a Custom Search Engine at [Google Custom Search](https://cse.google.com/cse/all)
   - Choose "Search the entire web"
   - Copy your Search Engine ID
6. Add to your `.env` file:
   ```env
   GOOGLE_API_KEY=your_api_key_here
   GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id_here
   ```

**Note**: Google Custom Search API provides 100 free queries per day. The system will use a fallback search method if the API is not configured, though results may be less comprehensive.

## Future Enhancements

- ✅ LLM integration for intelligent name generation
- ✅ Name evaluation framework
- ✅ Neo-brutalist UI design
- ✅ Rulebook integration
- ✅ Domain availability checking with GoDaddy API
- ✅ Google search insights for competitive analysis
- 🔄 Trademark search integration
- 🔄 Social media handle availability
- 🔄 Multi-language support
- 🔄 Batch processing for multiple startups

## License

MIT
