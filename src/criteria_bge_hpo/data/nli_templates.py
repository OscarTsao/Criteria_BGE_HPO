"""NLI Template Generation for DSM-5 Criteria Matching.

This module provides utilities for converting DSM-5 criteria into Natural Language
Inference (NLI) hypothesis templates. The NLI formulation treats:
- Premise: Reddit sentence (evidence text)
- Hypothesis: Templated statement about the criterion
- Label: 1 if premise entails hypothesis (person has symptom), 0 otherwise
"""

import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class TemplateType(str, Enum):
    """Template types for generating NLI hypotheses."""

    DIRECT = "direct"  # Use criterion text directly
    ENTAILMENT = "entailment"  # "This text indicates [symptom]"
    DESCRIPTION = "description"  # "This person is experiencing [symptom]"
    EVIDENCE = "evidence"  # "This text describes [symptom]"


# Short symptom descriptions for each DSM-5 MDD criterion
CRITERION_SHORT_FORMS = {
    "A.1": "depressed mood or feelings of sadness and hopelessness",
    "A.2": "loss of interest or pleasure in activities",
    "A.3": "significant changes in weight or appetite",
    "A.4": "sleep problems or insomnia",
    "A.5": "psychomotor agitation or retardation",
    "A.6": "fatigue or loss of energy",
    "A.7": "feelings of worthlessness or excessive guilt",
    "A.8": "difficulty thinking, concentrating, or making decisions",
    "A.9": "thoughts of death or suicide",
}


def create_entailment_template(symptom_desc: str) -> str:
    """Create an entailment-style hypothesis template.

    Args:
        symptom_desc: Short description of the symptom

    Returns:
        Formatted hypothesis string

    Example:
        >>> create_entailment_template("depressed mood")
        'This text indicates depressed mood'
    """
    return f"This text indicates {symptom_desc}"


def create_description_template(symptom_desc: str) -> str:
    """Create a description-style hypothesis template.

    Args:
        symptom_desc: Short description of the symptom

    Returns:
        Formatted hypothesis string

    Example:
        >>> create_description_template("depressed mood")
        'This person is experiencing depressed mood'
    """
    return f"This person is experiencing {symptom_desc}"


def create_evidence_template(symptom_desc: str) -> str:
    """Create an evidence-style hypothesis template.

    Args:
        symptom_desc: Short description of the symptom

    Returns:
        Formatted hypothesis string

    Example:
        >>> create_evidence_template("depressed mood")
        'This text describes depressed mood'
    """
    return f"This text describes {symptom_desc}"


class NLITemplateGenerator:
    """Generator for NLI hypothesis templates from DSM-5 criteria.

    This class provides methods to convert DSM-5 Major Depressive Disorder
    criteria into hypothesis templates suitable for Natural Language Inference
    tasks. It supports multiple template styles and handles criterion loading
    from JSON files.

    Attributes:
        dsm5_criteria: Dictionary of loaded DSM-5 criteria
        template_type: Type of template to generate (direct, entailment, etc.)
        criterion_short_forms: Short symptom descriptions for each criterion

    Example:
        >>> generator = NLITemplateGenerator(
        ...     dsm5_json_path="data/DSM5/MDD_Criteria.json",
        ...     template_type="entailment"
        ... )
        >>> hypothesis = generator.generate_hypothesis("A.1")
        >>> print(hypothesis)
        'This text indicates depressed mood or feelings of sadness and hopelessness'
    """

    def __init__(
        self,
        dsm5_json_path: str,
        template_type: str = "entailment",
        criterion_short_forms: Optional[Dict[str, str]] = None,
    ):
        """Initialize the NLI template generator.

        Args:
            dsm5_json_path: Path to DSM-5 criteria JSON file
            template_type: Type of template to generate. Must be one of:
                - "direct": Use criterion text directly
                - "entailment": "This text indicates [symptom]"
                - "description": "This person is experiencing [symptom]"
                - "evidence": "This text describes [symptom]"
            criterion_short_forms: Optional custom short forms for criteria.
                If None, uses default CRITERION_SHORT_FORMS.

        Raises:
            ValueError: If template_type is not valid
            FileNotFoundError: If DSM-5 JSON file does not exist
            json.JSONDecodeError: If JSON file is malformed
        """
        # Validate template type
        try:
            self.template_type = TemplateType(template_type)
        except ValueError:
            valid_types = [t.value for t in TemplateType]
            raise ValueError(
                f"Invalid template_type '{template_type}'. "
                f"Must be one of: {valid_types}"
            )

        # Load DSM-5 criteria
        self.dsm5_criteria = self.load_criteria(dsm5_json_path)

        # Use provided or default criterion short forms
        self.criterion_short_forms = (
            criterion_short_forms if criterion_short_forms is not None
            else CRITERION_SHORT_FORMS.copy()
        )

    def load_criteria(self, json_path: str) -> Dict:
        """Load DSM-5 criteria from JSON file.

        Args:
            json_path: Path to DSM-5 criteria JSON file

        Returns:
            Dictionary containing DSM-5 criteria data

        Raises:
            FileNotFoundError: If JSON file does not exist
            json.JSONDecodeError: If JSON file is malformed

        Example:
            >>> criteria = generator.load_criteria("data/DSM5/MDD_Criteria.json")
            >>> print(criteria.keys())
            dict_keys(['diagnosis', 'criteria'])
        """
        json_path_obj = Path(json_path)

        if not json_path_obj.exists():
            raise FileNotFoundError(
                f"DSM-5 criteria JSON file not found: {json_path}"
            )

        try:
            with open(json_path_obj, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Failed to parse DSM-5 JSON file: {e.msg}",
                e.doc,
                e.pos
            )

        # Convert list of criteria to dict keyed by id
        criteria_dict = {}
        if "criteria" in data:
            for criterion in data["criteria"]:
                criteria_dict[criterion["id"]] = criterion

        return criteria_dict

    def generate_hypothesis(self, criterion_id: str) -> str:
        """Generate NLI hypothesis for a specific criterion.

        Args:
            criterion_id: DSM-5 criterion identifier (e.g., "A.1", "A.2")

        Returns:
            Generated hypothesis string formatted according to template_type

        Raises:
            KeyError: If criterion_id is not found in short forms or DSM-5 data

        Example:
            >>> hypothesis = generator.generate_hypothesis("A.1")
            >>> print(hypothesis)
            'This text indicates depressed mood or feelings of sadness and hopelessness'
        """
        # Check if criterion exists in short forms
        if criterion_id not in self.criterion_short_forms:
            raise KeyError(
                f"Criterion '{criterion_id}' not found in criterion_short_forms. "
                f"Available criteria: {list(self.criterion_short_forms.keys())}"
            )

        symptom_desc = self.criterion_short_forms[criterion_id]

        # Generate hypothesis based on template type
        if self.template_type == TemplateType.DIRECT:
            # Use full criterion text from DSM-5 JSON if available
            if criterion_id in self.dsm5_criteria:
                criterion_data = self.dsm5_criteria[criterion_id]
                # Try to get full text, fall back to short form
                return criterion_data.get('text', symptom_desc)
            return symptom_desc

        elif self.template_type == TemplateType.ENTAILMENT:
            return create_entailment_template(symptom_desc)

        elif self.template_type == TemplateType.DESCRIPTION:
            return create_description_template(symptom_desc)

        elif self.template_type == TemplateType.EVIDENCE:
            return create_evidence_template(symptom_desc)

        # Fallback (should never reach here due to enum validation)
        return symptom_desc

    def get_all_templates(self) -> Dict[str, str]:
        """Generate hypothesis templates for all available criteria.

        Returns:
            Dictionary mapping criterion IDs to hypothesis templates

        Example:
            >>> templates = generator.get_all_templates()
            >>> print(templates["A.1"])
            'This text indicates depressed mood or feelings of sadness and hopelessness'
            >>> print(templates["A.2"])
            'This text indicates loss of interest or pleasure in activities'
        """
        return {
            criterion_id: self.generate_hypothesis(criterion_id)
            for criterion_id in self.criterion_short_forms.keys()
        }

    def format_nli_pair(
        self,
        premise: str,
        criterion_id: str,
        label: int
    ) -> Dict[str, Any]:
        """Format a complete NLI premise-hypothesis pair.

        Args:
            premise: The premise text (e.g., Reddit sentence)
            criterion_id: DSM-5 criterion identifier
            label: NLI label (1 for entailment, 0 for non-entailment)

        Returns:
            Dictionary containing premise, hypothesis, label, and criterion_id

        Raises:
            KeyError: If criterion_id is not found
            ValueError: If label is not 0 or 1

        Example:
            >>> nli_pair = generator.format_nli_pair(
            ...     premise="I've been feeling so sad and empty lately",
            ...     criterion_id="A.1",
            ...     label=1
            ... )
            >>> print(nli_pair)
            {
                'premise': "I've been feeling so sad and empty lately",
                'hypothesis': 'This text indicates depressed mood or feelings...',
                'label': 1,
                'criterion_id': 'A.1'
            }
        """
        # Validate label
        if label not in (0, 1):
            raise ValueError(
                f"Invalid label '{label}'. Must be 0 or 1 for binary NLI."
            )

        # Generate hypothesis
        hypothesis = self.generate_hypothesis(criterion_id)

        return {
            "premise": premise,
            "hypothesis": hypothesis,
            "label": label,
            "criterion_id": criterion_id
        }

    def batch_format_nli_pairs(
        self,
        pairs: List[Tuple[str, str, int]]
    ) -> List[Dict[str, Any]]:
        """Format multiple NLI pairs in batch.

        Args:
            pairs: List of (premise, criterion_id, label) tuples

        Returns:
            List of formatted NLI pair dictionaries

        Example:
            >>> pairs = [
            ...     ("I feel sad", "A.1", 1),
            ...     ("I can't sleep", "A.4", 1),
            ... ]
            >>> nli_pairs = generator.batch_format_nli_pairs(pairs)
            >>> len(nli_pairs)
            2
        """
        return [
            self.format_nli_pair(premise, criterion_id, label)
            for premise, criterion_id, label in pairs
        ]
