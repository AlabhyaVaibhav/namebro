"""
Startup Name Brainstorming Agent

An LLM-powered agent that generates startup name suggestions based on context and style parameters.
Uses LangChain for LLM integration and Pydantic for data validation.
"""

import os
import json
import re
from typing import List, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

# Import domain checker
try:
    from domain_checker import DomainChecker
    DOMAIN_CHECKER_AVAILABLE = True
except ImportError:
    DOMAIN_CHECKER_AVAILABLE = False
    DomainChecker = None

# Import Google search
try:
    from google_search import GoogleSearcher
    GOOGLE_SEARCH_AVAILABLE = True
except ImportError:
    GOOGLE_SEARCH_AVAILABLE = False
    GoogleSearcher = None

# Load environment variables
load_dotenv()


class NameStyle(str, Enum):
    """Style parameters for name generation"""
    GENZ = "genz"
    ABSTRACT_ARTISTIC = "abstract_artistic"
    PROFESSIONAL = "professional"
    PLAYFUL = "playful"
    TECH_MODERN = "tech_modern"
    MINIMALIST = "minimalist"
    BOLD_EDGY = "bold_edgy"
    CLASSIC = "classic"


class NameSuggestion(BaseModel):
    """Represents a name suggestion with metadata"""
    name: str = Field(description="The main startup name suggestion")
    style: str = Field(description="The style category of the name")
    variations: List[str] = Field(description="List of name variations")
    reasoning: Optional[str] = Field(default=None, description="Reasoning behind the name choice")


class NameSuggestionsResponse(BaseModel):
    """Response containing multiple name suggestions"""
    suggestions: List[NameSuggestion] = Field(description="List of name suggestions")


class NameEvaluation(BaseModel):
    """Evaluation scores for a single name"""
    name: str = Field(description="The name being evaluated")
    memorability: int = Field(description="Score 1-10 for memorability", ge=1, le=10)
    pronounceability: int = Field(description="Score 1-10 for pronounceability", ge=1, le=10)
    spelling_simplicity: int = Field(description="Score 1-10 for spelling simplicity", ge=1, le=10)
    meaning_associations: int = Field(description="Score 1-10 for meaning and associations", ge=1, le=10)
    phonetic_appeal: int = Field(description="Score 1-10 for phonetic appeal", ge=1, le=10)
    originality: int = Field(description="Score 1-10 for originality", ge=1, le=10)
    category_fit: int = Field(description="Score 1-10 for category fit", ge=1, le=10)
    domain_availability: int = Field(description="Score 1-10 for domain/trademark availability potential", ge=1, le=10)
    global_safety: int = Field(description="Score 1-10 for global safety", ge=1, le=10)
    visual_identity: int = Field(description="Score 1-10 for visual identity potential", ge=1, le=10)
    longevity: int = Field(description="Score 1-10 for longevity", ge=1, le=10)
    emotional_resonance: int = Field(description="Score 1-10 for emotional resonance", ge=1, le=10)
    total_score: int = Field(default=0, description="Total score out of 120")
    strengths: List[str] = Field(default_factory=list, description="List of key strengths")
    weaknesses: List[str] = Field(default_factory=list, description="List of key weaknesses")
    recommendation: str = Field(default="", description="Overall recommendation and reasoning")
    available_domains: List[str] = Field(default_factory=list, description="List of available domain variations")
    domain_check_error: Optional[str] = Field(default=None, description="Error message if domain check failed")
    search_insights: Optional[str] = Field(default=None, description="Google search insights summary")
    competing_domains: List[str] = Field(default_factory=list, description="List of competing domains found in search")
    search_categories: List[str] = Field(default_factory=list, description="Categories/areas found in search results")
    search_error: Optional[str] = Field(default=None, description="Error message if search failed")


class NameEvaluationsResponse(BaseModel):
    """Response containing multiple name evaluations"""
    evaluations: List[NameEvaluation] = Field(description="List of name evaluations")
    comparison: Optional[str] = Field(default="", description="Comparative analysis of all names")


class StartupNameAgent:
    """
    LLM-powered agent for brainstorming startup names based on context and style parameters.
    """
    
    def __init__(self, 
                 llm_provider: str = "openai",
                 model_name: Optional[str] = None,
                 temperature: float = 0.9,
                 enable_domain_check: bool = True):
        """
        Initialize the agent.
        
        Args:
            llm_provider: "openai" or "anthropic"
            model_name: Specific model name (e.g., "gpt-4", "claude-3-opus-20240229")
            temperature: Temperature for LLM generation (0.0-1.0)
            enable_domain_check: Enable domain availability checking via GoDaddy API
        """
        self.context: Optional[str] = None
        self.style_params: List[NameStyle] = []
        self.temperature = temperature
        self.enable_domain_check = enable_domain_check and DOMAIN_CHECKER_AVAILABLE
        self.enable_google_search = True  # Enable by default
        
        # Initialize domain checker if available
        self.domain_checker = None
        if self.enable_domain_check:
            try:
                use_ote = os.getenv("GODADDY_USE_OTE", "true").lower() == "true"
                self.domain_checker = DomainChecker(use_ote=use_ote)
            except Exception as e:
                print(f"Warning: Could not initialize domain checker: {e}")
                self.enable_domain_check = False
        
        # Initialize Google searcher if available
        self.google_searcher = None
        if self.enable_google_search and GOOGLE_SEARCH_AVAILABLE:
            try:
                self.google_searcher = GoogleSearcher()
            except Exception as e:
                print(f"Warning: Could not initialize Google searcher: {e}")
                self.enable_google_search = False
        
        # Initialize LLM
        if llm_provider.lower() == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            model_name = model_name or "gpt-4-turbo-preview"
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=api_key
            )
        elif llm_provider.lower() == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
            model_name = model_name or "claude-3-opus-20240229"
            self.llm = ChatAnthropic(
                model=model_name,
                temperature=temperature,
                api_key=api_key
            )
        else:
            raise ValueError(f"Unknown LLM provider: {llm_provider}. Use 'openai' or 'anthropic'")
        
        # Load rulebooks
        self._load_rulebooks()
        
        # Create prompt templates
        self._create_prompts()
    
    def _load_rulebooks(self):
        """Load rulebook files for guidance"""
        rulebook_path = os.path.join(os.path.dirname(__file__), "rulebook.txt")
        evaluation_rulebook_path = os.path.join(os.path.dirname(__file__), "evaluation_rulebook.txt")
        
        self.rulebook_content = ""
        self.evaluation_rulebook_content = ""
        
        if os.path.exists(rulebook_path):
            with open(rulebook_path, 'r', encoding='utf-8') as f:
                self.rulebook_content = f.read()
        
        if os.path.exists(evaluation_rulebook_path):
            with open(evaluation_rulebook_path, 'r', encoding='utf-8') as f:
                self.evaluation_rulebook_content = f.read()
        
    def _create_prompts(self):
        """Create prompt templates for name generation"""
        rulebook_guidance = f"""

NAMING RULEBOOK (Follow these principles):
{self.rulebook_content}

Apply these principles when generating names:
1. Infuse personal meaning or vision when possible
2. Keep names simple and memorable
3. Leverage phonetic appeal (names that roll off the tongue)
4. Use evocative or metaphorical names that create unique identity
"""
        
        self.system_prompt = f"""You are an expert startup naming consultant with deep knowledge of branding, 
marketing, and linguistic patterns. Your job is to generate creative, memorable, and brandable startup names 
based on the provided context and style preferences.

{rulebook_guidance}

Additional Guidelines:
- Generate names that are memorable, pronounceable, and brandable
- Consider domain availability potential (shorter is often better)
- Ensure names align with the requested style
- Provide creative variations of each name
- Explain your reasoning for each suggestion
- Avoid generic or overused names
- Consider the target audience and industry context"""

        self.name_generation_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_prompt),
            HumanMessagePromptTemplate.from_template("""
Context about the startup:
{context}

Style preferences: {styles}

Please generate {num_suggestions} startup name suggestions. For each name, provide:
1. The main name
2. At least 5 creative variations
3. Brief reasoning for why this name fits the context and style

Format your response as a structured list with clear sections for each suggestion.
""")
        ])
        
    def load_context_from_file(self, file_path: str) -> None:
        """
        Load context from a file.
        
        Args:
            file_path: Path to the file containing context
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            self.context = f.read()
    
    def load_context_from_text(self, text: str) -> None:
        """
        Load context from pasted text.
        
        Args:
            text: Text content to use as context
        """
        self.context = text
    
    def set_style_parameters(self, styles: Union[List[str], List[NameStyle]]) -> None:
        """
        Set style parameters for name generation.
        
        Args:
            styles: List of style names or NameStyle enums
        """
        self.style_params = []
        for style in styles:
            if isinstance(style, str):
                try:
                    self.style_params.append(NameStyle(style.lower()))
                except ValueError:
                    raise ValueError(f"Unknown style: {style}. Available styles: {[s.value for s in NameStyle]}")
            elif isinstance(style, NameStyle):
                self.style_params.append(style)
    
    def _parse_llm_response(self, response: str, styles: List[str]) -> List[NameSuggestion]:
        """
        Parse LLM response into NameSuggestion objects.
        Uses structured output if available, otherwise parses text.
        """
        suggestions = []
        
        # Try to use structured output with Pydantic
        try:
            structured_llm = self.llm.with_structured_output(NameSuggestionsResponse)
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=f"""
Context about the startup:
{self.context}

Style preferences: {', '.join(styles)}

Please generate startup name suggestions. For each name, provide:
- The main name
- At least 5 creative variations
- Brief reasoning for why this name fits the context and style

Return as a structured response with a list of suggestions.
""")
            ]
            result = structured_llm.invoke(messages)
            return result.suggestions
        except Exception as e:
            # Fallback to text parsing if structured output fails
            print(f"Structured output failed, using text parsing: {e}")
            return self._parse_text_response(response, styles)
    
    def _parse_text_response(self, response: str, styles: List[str]) -> List[NameSuggestion]:
        """Parse text response from LLM into NameSuggestion objects"""
        suggestions = []
        lines = response.split('\n')
        
        current_name = None
        current_variations = []
        current_reasoning = None
        current_style = styles[0] if styles else "professional"
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect name (usually numbered or marked)
            if any(marker in line.lower() for marker in ['1.', '2.', '3.', '4.', '5.', 'name:', 'suggestion:']):
                # Save previous suggestion
                if current_name:
                    suggestions.append(NameSuggestion(
                        name=current_name,
                        style=current_style,
                        variations=current_variations[:10],
                        reasoning=current_reasoning
                    ))
                
                # Extract new name
                parts = line.split(':', 1)
                if len(parts) > 1:
                    current_name = parts[1].strip().split()[0] if parts[1].strip() else None
                else:
                    # Try to extract from numbered list
                    import re
                    match = re.search(r'[A-Z][a-zA-Z]+', line)
                    if match:
                        current_name = match.group()
                current_variations = []
                current_reasoning = None
            
            # Detect variations
            elif any(marker in line.lower() for marker in ['variation', 'variant', '-']):
                import re
                # Extract potential names from the line
                words = re.findall(r'\b[A-Z][a-zA-Z]+\b', line)
                if words:
                    current_variations.extend(words)
            
            # Detect reasoning
            elif any(marker in line.lower() for marker in ['reason', 'why', 'because', 'fits']):
                current_reasoning = line
        
        # Add last suggestion
        if current_name:
            suggestions.append(NameSuggestion(
                name=current_name,
                style=current_style,
                variations=current_variations[:10] if current_variations else [],
                reasoning=current_reasoning
            ))
        
        return suggestions
    
    def generate_names(self, num_suggestions: int = 10) -> List[NameSuggestion]:
        """
        Generate startup name suggestions using LLM based on context and style parameters.
        
        Args:
            num_suggestions: Number of name suggestions to generate
            
        Returns:
            List of NameSuggestion objects
        """
        if not self.context:
            raise ValueError("No context provided. Use load_context_from_file() or load_context_from_text() first.")
        
        if not self.style_params:
            # Default to professional if no style specified
            self.style_params = [NameStyle.PROFESSIONAL]
        
        styles_str = ', '.join([style.value for style in self.style_params])
        
        # Generate names using LLM
        try:
            # Try structured output first
            structured_llm = self.llm.with_structured_output(NameSuggestionsResponse)
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=f"""
Context about the startup:
{self.context}

Style preferences: {styles_str}

Please generate {num_suggestions} startup name suggestions. For each name, provide:
- The main name (should be unique and brandable)
- At least 5 creative variations
- Brief reasoning (1-2 sentences) for why this name fits the context and style

Focus on creativity, memorability, and alignment with the style preferences.
""")
            ]
            result = structured_llm.invoke(messages)
            suggestions = result.suggestions[:num_suggestions]
            
            # Ensure each suggestion has the correct style
            for suggestion in suggestions:
                if not suggestion.style:
                    suggestion.style = styles_str
            
            return suggestions
            
        except Exception as e:
            # Fallback to regular LLM call with text parsing
            print(f"Structured output not available, using text parsing: {e}")
            prompt = self.name_generation_prompt.format_messages(
                context=self.context,
                styles=styles_str,
                num_suggestions=num_suggestions
            )
            response = self.llm.invoke(prompt)
            suggestions = self._parse_text_response(response.content, [s.value for s in self.style_params])
            return suggestions[:num_suggestions]
    
    def brainstorm(self, 
                   context: Optional[Union[str, List[str]]] = None,
                   styles: Optional[Union[List[str], List[NameStyle]]] = None,
                   num_suggestions: int = 10) -> List[NameSuggestion]:
        """
        Main method to brainstorm names. Convenience method that combines loading context and generating names.
        
        Args:
            context: File path(s) or text content
            styles: List of style parameters
            num_suggestions: Number of suggestions to generate
            
        Returns:
            List of NameSuggestion objects
        """
        # Load context
        if context:
            if isinstance(context, str):
                # Check if it's a file path
                if os.path.exists(context):
                    self.load_context_from_file(context)
                else:
                    self.load_context_from_text(context)
            elif isinstance(context, list):
                # Multiple files or text chunks
                combined_text = []
                for item in context:
                    if os.path.exists(item):
                        with open(item, 'r', encoding='utf-8') as f:
                            combined_text.append(f.read())
                    else:
                        combined_text.append(item)
                self.load_context_from_text("\n\n".join(combined_text))
        
        # Set styles
        if styles:
            self.set_style_parameters(styles)
        
        # Generate names
        return self.generate_names(num_suggestions)
    
    def _enhance_with_domain_data(self, evaluations: List[NameEvaluation]) -> List[NameEvaluation]:
        """
        Enhance evaluations with real domain availability data from GoDaddy API.
        
        Args:
            evaluations: List of NameEvaluation objects
            
        Returns:
            Updated evaluations with domain availability data
        """
        if not self.domain_checker:
            return evaluations
        
        for evaluation in evaluations:
            try:
                # Get domain availability score and available domains
                domain_score, available_domains = self.domain_checker.calculate_domain_availability_score(
                    evaluation.name
                )
                
                # Update domain availability score (use real data if available)
                if domain_score > 0:
                    evaluation.domain_availability = domain_score
                    evaluation.available_domains = available_domains
                    
                    # Update strengths/weaknesses based on domain availability
                    if available_domains:
                        domain_strength = f"Available domains: {', '.join(available_domains[:3])}"
                        if domain_strength not in evaluation.strengths:
                            evaluation.strengths.append(domain_strength)
                    else:
                        domain_weakness = "No available domain variations found"
                        if domain_weakness not in evaluation.weaknesses:
                            evaluation.weaknesses.append(domain_weakness)
                    
                    # Recalculate total score with updated domain_availability
                    evaluation.total_score = (
                        evaluation.memorability + evaluation.pronounceability +
                        evaluation.spelling_simplicity + evaluation.meaning_associations +
                        evaluation.phonetic_appeal + evaluation.originality +
                        evaluation.category_fit + evaluation.domain_availability +
                        evaluation.global_safety + evaluation.visual_identity +
                        evaluation.longevity + evaluation.emotional_resonance
                    )
                else:
                    evaluation.domain_check_error = "Domain check returned no results"
                    
            except Exception as e:
                evaluation.domain_check_error = f"Domain check failed: {str(e)}"
                # Keep the LLM-generated domain_availability score if domain check fails
        
        return evaluations
    
    def _enhance_with_search_data(self, evaluations: List[NameEvaluation]) -> List[NameEvaluation]:
        """
        Enhance evaluations with Google search insights.
        
        Args:
            evaluations: List of NameEvaluation objects
            
        Returns:
            Updated evaluations with search insights
        """
        if not self.google_searcher:
            return evaluations
        
        for evaluation in evaluations:
            try:
                # Search for the startup name
                search_insights = self.google_searcher.search_startup_name(evaluation.name)
                
                # Update evaluation with search data
                evaluation.search_insights = search_insights.summary
                evaluation.competing_domains = search_insights.domains_found[:10]
                evaluation.search_categories = search_insights.categories[:10]
                
                if search_insights.error:
                    evaluation.search_error = search_insights.error
                
                # Update originality score based on search results
                # If many competing results found, reduce originality score
                if search_insights.results:
                    num_competitors = len(search_insights.results)
                    if num_competitors > 10:
                        # Many competitors found - reduce originality
                        evaluation.originality = max(1, evaluation.originality - 2)
                        if "High competition" not in evaluation.weaknesses:
                            evaluation.weaknesses.append(f"High competition: {num_competitors} existing results found")
                    elif num_competitors > 5:
                        # Moderate competition
                        evaluation.originality = max(1, evaluation.originality - 1)
                        if "Some competition" not in evaluation.weaknesses:
                            evaluation.weaknesses.append(f"Some competition: {num_competitors} existing results found")
                    else:
                        # Low competition - good sign
                        if "Low competition" not in evaluation.strengths:
                            evaluation.strengths.append(f"Low competition: Only {num_competitors} existing results")
                    
                    # Add search insights to recommendation
                    if search_insights.domains_found:
                        domain_info = f"Competing domains: {', '.join(search_insights.domains_found[:3])}"
                        if domain_info not in evaluation.recommendation:
                            evaluation.recommendation += f" {domain_info}."
                else:
                    # No results found - good for originality
                    if "No existing competitors found" not in evaluation.strengths:
                        evaluation.strengths.append("No existing competitors found in search")
                
                # Recalculate total score with updated originality
                evaluation.total_score = (
                    evaluation.memorability + evaluation.pronounceability +
                    evaluation.spelling_simplicity + evaluation.meaning_associations +
                    evaluation.phonetic_appeal + evaluation.originality +
                    evaluation.category_fit + evaluation.domain_availability +
                    evaluation.global_safety + evaluation.visual_identity +
                    evaluation.longevity + evaluation.emotional_resonance
                )
                
            except Exception as e:
                evaluation.search_error = f"Search failed: {str(e)}"
                # Keep the LLM-generated scores if search fails
        
        return evaluations
    
    def evaluate_names(self, names: List[str], startup_context: Optional[str] = None) -> List[NameEvaluation]:
        """
        Evaluate a list of startup names using the evaluation framework.
        
        Args:
            names: List of names to evaluate
            startup_context: Optional context about the startup for better evaluation
            
        Returns:
            List of NameEvaluation objects with scores and recommendations
        """
        if not names:
            raise ValueError("No names provided for evaluation")
        
        evaluation_prompt = f"""You are an expert startup naming consultant evaluating names based on a comprehensive framework.

EVALUATION FRAMEWORK:
{self.evaluation_rulebook_content}

Evaluate each name on a scale of 1-10 for each parameter:
1. Memorability - How easy is it to recall after hearing once?
2. Pronounceability - Can people from different regions say it easily?
3. Spelling Simplicity - Can someone spell it correctly after hearing it?
4. Meaning & Associations - Does it evoke the right emotional/conceptual cues?
5. Phonetic Appeal - Does it feel good to say?
6. Originality & Distinctiveness - Does it stand out from competitors?
7. Category Fit - Does it feel appropriate for the industry?
8. Domain & Trademark Availability - Can you get a reasonable domain?
9. Global Safety - Does it cause issues in other cultures/languages?
10. Visual Identity Potential - Does it look appealing when written?
11. Longevity - Will it still make sense in 10 years?
12. Emotional Resonance - Does it feel right?

For each name, provide:
- Scores (1-10) for all 12 parameters
- Total score (sum of all parameters, max 120)
- Key strengths (2-3 items)
- Key weaknesses (2-3 items)
- Overall recommendation with reasoning

Startup Context (if provided):
{startup_context if startup_context else "No specific context provided"}

Names to evaluate: {', '.join(names)}
"""
        
        try:
            # Try to get structured output, but handle validation errors gracefully
            try:
                structured_llm = self.llm.with_structured_output(NameEvaluationsResponse)
                messages = [
                    SystemMessage(content=evaluation_prompt),
                    HumanMessage(content=f"""
Please evaluate these startup names: {', '.join(names)}

For EACH name, you MUST provide ALL of the following fields:
1. All 12 parameter scores (1-10): memorability, pronounceability, spelling_simplicity, meaning_associations, phonetic_appeal, originality, category_fit, domain_availability, global_safety, visual_identity, longevity, emotional_resonance
2. total_score: Integer sum of all 12 scores (must be provided, not calculated)
3. strengths: Array of 2-3 strings, e.g. ["Short and memorable", "Easy to pronounce"]
4. weaknesses: Array of 2-3 strings, e.g. ["May be too generic", "Domain might be taken"]
5. recommendation: String with brief recommendation

Return a JSON object with "evaluations" array and optional "comparison" string.
""")
                ]
                result = structured_llm.invoke(messages)
                evaluations = result.evaluations
                
            except Exception as validation_error:
                # If structured output fails, try parsing the raw response
                print(f"Structured output validation failed: {validation_error}")
                # Fall back to getting raw response and manually constructing
                raw_llm = self.llm
                messages = [
                    SystemMessage(content=evaluation_prompt),
                    HumanMessage(content=f"""
Please evaluate these startup names: {', '.join(names)}

For EACH name, provide scores for all 12 parameters and analysis.
Return as JSON with evaluations array.
""")
                ]
                raw_response = raw_llm.invoke(messages)
                
                # Try to parse JSON from response
                response_text = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)
                
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        data = json.loads(json_match.group())
                        evaluations_data = data.get('evaluations', [])
                        evaluations = []
                        for eval_data in evaluations_data:
                            # Calculate total if missing
                            if 'total_score' not in eval_data or eval_data['total_score'] == 0:
                                eval_data['total_score'] = sum([
                                    eval_data.get('memorability', 7),
                                    eval_data.get('pronounceability', 7),
                                    eval_data.get('spelling_simplicity', 7),
                                    eval_data.get('meaning_associations', 7),
                                    eval_data.get('phonetic_appeal', 7),
                                    eval_data.get('originality', 7),
                                    eval_data.get('category_fit', 7),
                                    eval_data.get('domain_availability', 7),
                                    eval_data.get('global_safety', 7),
                                    eval_data.get('visual_identity', 7),
                                    eval_data.get('longevity', 7),
                                    eval_data.get('emotional_resonance', 7),
                                ])
                            
                            # Ensure all fields exist
                            eval_data.setdefault('strengths', [])
                            eval_data.setdefault('weaknesses', [])
                            eval_data.setdefault('recommendation', f"Overall score: {eval_data['total_score']}/120")
                            
                            evaluations.append(NameEvaluation(**eval_data))
                    except json.JSONDecodeError:
                        # If JSON parsing fails, create default evaluations
                        evaluations = []
                        for name in names:
                            evaluations.append(NameEvaluation(
                                name=name,
                                memorability=7,
                                pronounceability=7,
                                spelling_simplicity=7,
                                meaning_associations=7,
                                phonetic_appeal=7,
                                originality=7,
                                category_fit=7,
                                domain_availability=7,
                                global_safety=7,
                                visual_identity=7,
                                longevity=7,
                                emotional_resonance=7,
                                total_score=84,
                                strengths=["Evaluation in progress"],
                                weaknesses=["Please retry"],
                                recommendation="Evaluation completed with default scores"
                            ))
                else:
                    # If no JSON found, create default evaluations
                    evaluations = []
                    for name in names:
                        evaluations.append(NameEvaluation(
                            name=name,
                            memorability=7,
                            pronounceability=7,
                            spelling_simplicity=7,
                            meaning_associations=7,
                            phonetic_appeal=7,
                            originality=7,
                            category_fit=7,
                            domain_availability=7,
                            global_safety=7,
                            visual_identity=7,
                            longevity=7,
                            emotional_resonance=7,
                            total_score=84,
                            strengths=["Evaluation in progress"],
                            weaknesses=["Please retry"],
                            recommendation="Evaluation completed with default scores"
                        ))
            
            # Ensure all required fields are populated (safety check)
            for evaluation in evaluations:
                # Calculate total score if not provided or is 0
                if not evaluation.total_score or evaluation.total_score == 0:
                    evaluation.total_score = (
                        evaluation.memorability + evaluation.pronounceability +
                        evaluation.spelling_simplicity + evaluation.meaning_associations +
                        evaluation.phonetic_appeal + evaluation.originality +
                        evaluation.category_fit + evaluation.domain_availability +
                        evaluation.global_safety + evaluation.visual_identity +
                        evaluation.longevity + evaluation.emotional_resonance
                    )
                
                # Ensure strengths list exists
                if not evaluation.strengths:
                    evaluation.strengths = ["Good overall score"]
                
                # Ensure weaknesses list exists
                if not evaluation.weaknesses:
                    evaluation.weaknesses = ["Consider domain availability"]
                
                # Ensure recommendation exists
                if not evaluation.recommendation:
                    evaluation.recommendation = f"Overall score: {evaluation.total_score}/120. Consider this name based on your specific startup context and requirements."
            
            # Enhance evaluations with real domain availability data
            if self.enable_domain_check and self.domain_checker:
                evaluations = self._enhance_with_domain_data(evaluations)
            
            # Enhance evaluations with Google search insights
            if self.enable_google_search and self.google_searcher:
                evaluations = self._enhance_with_search_data(evaluations)
            
            return evaluations
            
        except Exception as e:
            # Fallback to text parsing if structured output fails
            print(f"Structured output failed, using text parsing: {e}")
            prompt = self.name_generation_prompt.format_messages(
                context=evaluation_prompt,
                styles="evaluation",
                num_suggestions=len(names)
            )
            response = self.llm.invoke([
                SystemMessage(content=evaluation_prompt),
                HumanMessage(content=f"Evaluate these names: {', '.join(names)}")
            ])
            
            # Parse text response (simplified fallback)
            evaluations = []
            for name in names:
                evaluations.append(NameEvaluation(
                    name=name,
                    memorability=7,
                    pronounceability=7,
                    spelling_simplicity=7,
                    meaning_associations=7,
                    phonetic_appeal=7,
                    originality=7,
                    category_fit=7,
                    domain_availability=7,
                    global_safety=7,
                    visual_identity=7,
                    longevity=7,
                    emotional_resonance=7,
                    total_score=84,
                    strengths=["Needs LLM evaluation"],
                    weaknesses=["Structured output unavailable"],
                    recommendation="Please retry with structured output support"
                ))
            return evaluations


def main():
    """Example usage of the StartupNameAgent"""
    try:
        agent = StartupNameAgent(llm_provider="openai")
        
        # Example 1: Load from text
        example_context = """
        We're building a platform that helps freelancers manage their projects,
        track time, and invoice clients. It's modern, user-friendly, and designed
        for the gig economy. We want something that resonates with Gen Z and
        millennial freelancers.
        """
        
        agent.load_context_from_text(example_context)
        agent.set_style_parameters([NameStyle.GENZ, NameStyle.TECH_MODERN])
        
        print("Generating startup name suggestions...")
        suggestions = agent.generate_names(num_suggestions=5)
        
        print("\nStartup Name Suggestions:")
        print("=" * 50)
        for i, suggestion in enumerate(suggestions, 1):
            print(f"\n{i}. {suggestion.name}")
            print(f"   Style: {suggestion.style}")
            print(f"   Variations: {', '.join(suggestion.variations[:5])}")
            if suggestion.reasoning:
                print(f"   Reasoning: {suggestion.reasoning}")
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set OPENAI_API_KEY or ANTHROPIC_API_KEY in your environment variables or .env file")


if __name__ == "__main__":
    main()
