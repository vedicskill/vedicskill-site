"""
SMART Objectives Scoring System using Open-Source Transformer Models
Self-hosted solution with no API dependencies
"""

import json
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)

@dataclass
class SMARTScore:
    """Container for SMART scores"""
    specific: float
    measurable: float
    achievable: float
    relevant: float
    time_bound: float
    overall: float
    feedback: Dict[str, str]
    
    def to_dict(self):
        return {
            'specific': self.specific,
            'measurable': self.measurable,
            'achievable': self.achievable,
            'relevant': self.relevant,
            'time_bound': self.time_bound,
            'overall': self.overall,
            'feedback': self.feedback
        }


class OpenSourceSMARTScorer:
    """
    SMART Objective Scorer using open-source LLMs
    Supports: Llama 3, Mistral, Phi, Gemma, Qwen, and more
    """
    
    # Recommended models by use case
    RECOMMENDED_MODELS = {
        'high_accuracy': 'meta-llama/Meta-Llama-3.1-8B-Instruct',  # Best quality
        'balanced': 'mistralai/Mistral-7B-Instruct-v0.3',          # Good balance
        'fast': 'microsoft/Phi-3-mini-4k-instruct',                # Fastest, low memory
        'small': 'google/gemma-2-2b-it',                           # Very small, CPU-friendly
        'multilingual': 'Qwen/Qwen2.5-7B-Instruct',               # Multiple languages
    }
    
    def __init__(
        self, 
        model_name: str = 'mistralai/Mistral-7B-Instruct-v0.3',
        device: str = 'auto',
        quantization: bool = True,
        max_memory: Optional[Dict] = None
    ):
        """
        Initialize the scorer with an open-source model
        
        Args:
            model_name: HuggingFace model identifier
            device: 'cuda', 'cpu', or 'auto'
            quantization: Use 4-bit quantization to reduce memory (recommended)
            max_memory: Dict of memory limits per device
        """
        self.model_name = model_name
        self.device = device
        
        print(f"Loading model: {model_name}...")
        
        # Configure quantization for memory efficiency
        if quantization and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs = {
                "quantization_config": quantization_config,
                "device_map": "auto",
                "torch_dtype": torch.float16
            }
        else:
            model_kwargs = {
                "device_map": device,
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32
            }
        
        if max_memory:
            model_kwargs["max_memory"] = max_memory
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            **model_kwargs
        )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✓ Model loaded successfully on {self.model.device}")
        print(f"✓ Memory footprint: ~{self._get_model_size():.2f} GB")
    
    def _get_model_size(self) -> float:
        """Estimate model memory usage in GB"""
        param_size = sum(p.nelement() * p.element_size() for p in self.model.parameters())
        return param_size / (1024**3)
    
    def _format_prompt(self, objective: str, context: Optional[Dict] = None) -> str:
        """
        Format prompt based on model type (different models use different formats)
        """
        context_str = ""
        if context:
            context_str = f"""
Context Information:
- Employee Role: {context.get('role', 'Not specified')}
- Department: {context.get('department', 'Not specified')}
- Time Period: {context.get('period', 'Not specified')}
"""
        
        system_prompt = """You are an expert HR analyst specializing in evaluating employee objectives using the SMART framework."""
        
        user_prompt = f"""{context_str}

Evaluate this objective using SMART criteria:
"{objective}"

SCORING GUIDE (0-10 scale):

1. SPECIFIC (0-10):
   - 10: Crystal clear with all details (what, why, who, where)
   - 5: Somewhat clear but missing details
   - 0: Vague and ambiguous

2. MEASURABLE (0-10):
   - 10: Concrete metrics with numbers/percentages
   - 5: Some indicators but not fully quantifiable
   - 0: No way to measure success

3. ACHIEVABLE (0-10):
   - 10: Challenging but realistic with available resources
   - 5: Uncertain feasibility
   - 0: Impossible or too easy

4. RELEVANT (0-10):
   - 10: Directly aligned with business goals
   - 5: Tangentially related
   - 0: Irrelevant to role/company

5. TIME-BOUND (0-10):
   - 10: Specific deadline with clear timeline
   - 5: Vague timeframe (e.g., "soon", "eventually")
   - 0: No time reference

Respond ONLY with valid JSON in this exact format (no additional text):
{{
  "scores": {{
    "specific": 0-10,
    "measurable": 0-10,
    "achievable": 0-10,
    "relevant": 0-10,
    "time_bound": 0-10
  }},
  "feedback": {{
    "specific": "brief explanation",
    "measurable": "brief explanation",
    "achievable": "brief explanation",
    "relevant": "brief explanation",
    "time_bound": "brief explanation"
  }},
  "strengths": ["strength1", "strength2"],
  "improvements": ["improvement1", "improvement2"],
  "suggested_revision": "improved objective or null"
}}"""

        # Format for different model families
        if "llama-3" in self.model_name.lower() or "llama3" in self.model_name.lower():
            # Llama 3 format
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        elif "mistral" in self.model_name.lower():
            # Mistral format
            prompt = f"[INST] {system_prompt}\n\n{user_prompt} [/INST]"
        
        elif "phi" in self.model_name.lower():
            # Phi format
            prompt = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{user_prompt}<|end|>\n<|assistant|>\n"
        
        elif "gemma" in self.model_name.lower():
            # Gemma format
            prompt = f"<start_of_turn>user\n{system_prompt}\n\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"
        
        elif "qwen" in self.model_name.lower():
            # Qwen format
            prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        else:
            # Generic format
            prompt = f"{system_prompt}\n\n{user_prompt}\n\nResponse:"
        
        return prompt
    
    def _extract_json(self, text: str) -> Dict:
        """Extract JSON from model output with robust error handling"""
        # Try to find JSON in code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        
        # Try to find JSON object
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            text = json_match.group(0)
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fallback: try to clean and parse
            text = text.strip()
            text = re.sub(r',(\s*[}\]])', r'\1', text)  # Remove trailing commas
            try:
                return json.loads(text)
            except:
                # Return default structure if parsing fails
                return {
                    "scores": {
                        "specific": 5.0,
                        "measurable": 5.0,
                        "achievable": 5.0,
                        "relevant": 5.0,
                        "time_bound": 5.0
                    },
                    "feedback": {
                        "specific": "Unable to parse model output",
                        "measurable": "Unable to parse model output",
                        "achievable": "Unable to parse model output",
                        "relevant": "Unable to parse model output",
                        "time_bound": "Unable to parse model output"
                    },
                    "strengths": ["Parse error occurred"],
                    "improvements": ["Please try again"],
                    "suggested_revision": None
                }
    
    def score_objective(
        self, 
        objective: str, 
        context: Optional[Dict] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.3
    ) -> SMARTScore:
        """
        Score an objective using the loaded model
        
        Args:
            objective: The objective text to evaluate
            context: Optional contextual information
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            
        Returns:
            SMARTScore object with scores and feedback
        """
        prompt = self._format_prompt(objective, context)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract response (remove prompt)
        response = generated_text[len(self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):]
        
        # Parse JSON
        try:
            result = self._extract_json(response)
            scores = result['scores']
            feedback = result['feedback']
            
            # Calculate overall score (weighted average)
            overall = (
                scores['specific'] * 0.20 +
                scores['measurable'] * 0.25 +
                scores['achievable'] * 0.20 +
                scores['relevant'] * 0.15 +
                scores['time_bound'] * 0.20
            )
            
            # Add additional fields
            feedback['strengths'] = result.get('strengths', [])
            feedback['improvements'] = result.get('improvements', [])
            feedback['suggested_revision'] = result.get('suggested_revision')
            
            return SMARTScore(
                specific=float(scores['specific']),
                measurable=float(scores['measurable']),
                achievable=float(scores['achievable']),
                relevant=float(scores['relevant']),
                time_bound=float(scores['time_bound']),
                overall=round(overall, 2),
                feedback=feedback
            )
        
        except Exception as e:
            print(f"Error parsing response: {e}")
            print(f"Raw response: {response[:500]}")
            raise
    
    def batch_score_objectives(
        self, 
        objectives: List[Dict],
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Score multiple objectives in batch
        
        Args:
            objectives: List of dicts with 'objective' and optional 'context' keys
            show_progress: Show progress bar
            
        Returns:
            List of results with scores and feedback
        """
        results = []
        
        for idx, obj_data in enumerate(objectives):
            if show_progress:
                print(f"Processing {idx+1}/{len(objectives)}...", end='\r')
            
            objective = obj_data['objective']
            context = obj_data.get('context')
            
            try:
                score = self.score_objective(objective, context)
                results.append({
                    'id': obj_data.get('id', idx),
                    'objective': objective,
                    'scores': score.to_dict(),
                    'status': 'success'
                })
            except Exception as e:
                results.append({
                    'id': obj_data.get('id', idx),
                    'objective': objective,
                    'error': str(e),
                    'status': 'error'
                })
        
        if show_progress:
            print(f"\nCompleted {len(results)} objectives")
        
        return results
    
    def generate_report(self, score: SMARTScore, objective: str) -> str:
        """Generate a formatted report for the scored objective"""
        
        def get_rating(score_val):
            if score_val >= 8: return "Excellent ✓"
            elif score_val >= 6: return "Good"
            elif score_val >= 4: return "Fair ⚠"
            else: return "Needs Work ✗"
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║              SMART OBJECTIVE EVALUATION REPORT               ║
╚══════════════════════════════════════════════════════════════╝

OBJECTIVE:
{objective}

OVERALL SCORE: {score.overall}/10 - {get_rating(score.overall)}

INDIVIDUAL SCORES:
├─ Specific:     {score.specific}/10  {'█' * int(score.specific)}{'░' * (10-int(score.specific))} {get_rating(score.specific)}
├─ Measurable:   {score.measurable}/10  {'█' * int(score.measurable)}{'░' * (10-int(score.measurable))} {get_rating(score.measurable)}
├─ Achievable:   {score.achievable}/10  {'█' * int(score.achievable)}{'░' * (10-int(score.achievable))} {get_rating(score.achievable)}
├─ Relevant:     {score.relevant}/10  {'█' * int(score.relevant)}{'░' * (10-int(score.relevant))} {get_rating(score.relevant)}
└─ Time-bound:   {score.time_bound}/10  {'█' * int(score.time_bound)}{'░' * (10-int(score.time_bound))} {get_rating(score.time_bound)}

DETAILED FEEDBACK:

📝 Specific: {score.feedback['specific']}

📊 Measurable: {score.feedback['measurable']}

🎯 Achievable: {score.feedback['achievable']}

🔗 Relevant: {score.feedback['relevant']}

⏰ Time-bound: {score.feedback['time_bound']}

STRENGTHS:
"""
        for strength in score.feedback.get('strengths', []):
            report += f"  ✓ {strength}\n"
        
        report += "\nAREAS FOR IMPROVEMENT:\n"
        for improvement in score.feedback.get('improvements', []):
            report += f"  → {improvement}\n"
        
        if score.feedback.get('suggested_revision'):
            report += f"\n💡 SUGGESTED REVISION:\n{score.feedback['suggested_revision']}\n"
        
        report += "\n" + "="*64 + "\n"
        
        return report


# Example usage and testing
if __name__ == "__main__":
    # Initialize scorer with your preferred model
    print("Initializing SMART Scorer...")
    
    # Choose model based on your hardware:
    # - GPU with 8GB+ VRAM: 'mistralai/Mistral-7B-Instruct-v0.3'
    # - GPU with 4GB VRAM: 'microsoft/Phi-3-mini-4k-instruct'
    # - CPU only: 'google/gemma-2-2b-it'
    
    scorer = OpenSourceSMARTScorer(
        model_name='microsoft/Phi-3-mini-4k-instruct',  # Fast and efficient
        quantization=True  # Reduces memory usage
    )
    
    # Test objectives
    test_objectives = [
        {
            'id': 1,
            'objective': "Increase sales",
            'context': {
                'role': 'Sales Manager',
                'department': 'Sales',
                'period': 'Q1 2025'
            }
        },
        {
            'id': 2,
            'objective': "Increase customer satisfaction score from 7.2 to 8.5 by implementing a new feedback system and training program by June 30, 2025",
            'context': {
                'role': 'Customer Success Manager',
                'department': 'Customer Service',
                'period': 'H1 2025'
            }
        },
        {
            'id': 3,
            'objective': "Launch 3 new product features related to user analytics by Q2 end, validated through A/B testing with 1000+ users, achieving 15% engagement improvement",
            'context': {
                'role': 'Product Manager',
                'department': 'Product',
                'period': 'Q1-Q2 2025'
            }
        }
    ]
    
    # Score individual objective
    print("\n" + "="*64)
    print("Scoring individual objective...")
    print("="*64)
    
    score = scorer.score_objective(
        test_objectives[1]['objective'],
        test_objectives[1]['context']
    )
    print(scorer.generate_report(score, test_objectives[1]['objective']))
    
    # Batch scoring
    print("\n" + "="*64)
    print("Batch scoring all objectives...")
    print("="*64 + "\n")
    
    results = scorer.batch_score_objectives(test_objectives)
    
    for result in results:
        if result['status'] == 'success':
            print(f"✓ Objective {result['id']}: Overall Score = {result['scores']['overall']}/10")
        else:
            print(f"✗ Objective {result['id']}: Error - {result['error']}")
