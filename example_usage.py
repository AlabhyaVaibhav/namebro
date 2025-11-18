"""
Example usage of the Startup Name Brainstorming Agent
"""

from startup_name_agent import StartupNameAgent, NameStyle


def example_1_basic_usage():
    """Basic usage example"""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    agent = StartupNameAgent()
    
    context = """
    We're building a platform that helps freelancers manage their projects,
    track time, and invoice clients. It's modern, user-friendly, and designed
    for the gig economy. We want something that resonates with Gen Z and
    millennial freelancers.
    """
    
    agent.load_context_from_text(context)
    agent.set_style_parameters([NameStyle.GENZ, NameStyle.TECH_MODERN])
    
    suggestions = agent.generate_names(num_suggestions=5)
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"\n{i}. {suggestion.name}")
        print(f"   Style: {suggestion.style}")
        print(f"   Top Variations: {', '.join(suggestion.variations[:3])}")
        print(f"   Reasoning: {suggestion.reasoning}")


def example_2_abstract_artistic():
    """Abstract artistic style example"""
    print("\n" + "=" * 60)
    print("Example 2: Abstract Artistic Style")
    print("=" * 60)
    
    agent = StartupNameAgent()
    
    context = """
    An AI-powered creative platform for artists and designers.
    Enables collaboration, inspiration, and digital art creation.
    Focus on creativity, innovation, and artistic expression.
    """
    
    agent.load_context_from_text(context)
    agent.set_style_parameters([NameStyle.ABSTRACT_ARTISTIC])
    
    suggestions = agent.generate_names(num_suggestions=5)
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"\n{i}. {suggestion.name}")
        print(f"   Variations: {', '.join(suggestion.variations[:4])}")


def example_3_convenience_method():
    """Using the convenience brainstorm method"""
    print("\n" + "=" * 60)
    print("Example 3: Convenience Method")
    print("=" * 60)
    
    agent = StartupNameAgent()
    
    suggestions = agent.brainstorm(
        context="""
        A fitness app that uses AI to personalize workout routines.
        Tracks progress, provides nutrition advice, and connects users
        with personal trainers. Modern, tech-forward approach.
        """,
        styles=["tech_modern", "minimalist"],
        num_suggestions=6
    )
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion.name} ({suggestion.style})")


def example_4_multiple_styles():
    """Multiple styles example"""
    print("\n" + "=" * 60)
    print("Example 4: Multiple Styles")
    print("=" * 60)
    
    agent = StartupNameAgent()
    
    context = """
    Sustainable e-commerce platform for eco-friendly products.
    Connects conscious consumers with ethical brands.
    Mission-driven, modern, and accessible.
    """
    
    agent.load_context_from_text(context)
    agent.set_style_parameters([
        NameStyle.PROFESSIONAL,
        NameStyle.MINIMALIST,
        NameStyle.TECH_MODERN
    ])
    
    suggestions = agent.generate_names(num_suggestions=8)
    
    # Group by style
    by_style = {}
    for suggestion in suggestions:
        if suggestion.style not in by_style:
            by_style[suggestion.style] = []
        by_style[suggestion.style].append(suggestion)
    
    for style, style_suggestions in by_style.items():
        print(f"\n{style.upper()} Style:")
        for suggestion in style_suggestions:
            print(f"  • {suggestion.name}")


if __name__ == "__main__":
    example_1_basic_usage()
    example_2_abstract_artistic()
    example_3_convenience_method()
    example_4_multiple_styles()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)

