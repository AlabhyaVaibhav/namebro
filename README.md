# Namebro

**Namebro** is an **LLM-powered** intelligent agent that generates creative startup name suggestions based on context and customizable style parameters. Built with LangChain, Pydantic, and featuring a beautiful neo-brutalist Streamlit interface.

## Features

- **🤖 LLM-Powered**: Uses ChatGPT (OpenAI) or Claude (Anthropic) for intelligent name generation
- **📝 Context Input**: Accept context from files or pasted text
- **🎨 Style Parameters**: Generate names in various styles (GenZ, abstract artistic, professional, etc.)
- **✨ Name Variations**: Automatically generates creative variations of suggested names
- **📊 Name Evaluation**: Comprehensive evaluation across 12 parameters (memorability, pronounceability, etc.)
- **💬 Neo-Brutalist UI**: Beautiful, bold Streamlit interface with high-contrast design
- **📚 Rulebook Integration**: Uses proven naming principles and evaluation frameworks
- **🔧 Extensible**: Ready for custom tools (domain checking, trademark search, etc.)

## Installation

```bash
# Clone or navigate to the project directory
cd Namebro

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
```

**Note**: You only need one API key, depending on which LLM provider you want to use.

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

## Future Enhancements

- ✅ LLM integration for intelligent name generation
- ✅ Name evaluation framework
- ✅ Neo-brutalist UI design
- ✅ Rulebook integration
- 🔄 Domain availability checking (tool integration ready)
- 🔄 Trademark search integration (tool integration ready)
- 🔄 Social media handle availability
- 🔄 Multi-language support
- 🔄 Batch processing for multiple startups

## License

MIT
