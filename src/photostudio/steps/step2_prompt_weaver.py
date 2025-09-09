#!/usr/bin/env python3
"""
step2_prompt_weaver.py - Phase 2: Prompt Weaver
===============================================

Convert JSON analysis into sectioned natural-language prompts with guardrails.

Sections:
1. Template inheritance & verification flags
2. Non-negotiable facts (JSON paraphrased)
3. Mandatory style guide (white bg, high-key, 2048², sRGB, frame %)
4. Garment-aware rules (zipper logic, hollow openings, etc.)
5. Forbidden edits (no invented features, no label relocation)
6. Critical quality checks (ΔE match, resolution, hollow effect)

Dependencies: PyYAML pathlib
"""

from __future__ import annotations

import json
import argparse
import logging
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import yaml

logger = logging.getLogger("photostudio.prompt_weaver")

# ---------------------------------------------------------------------------
# Template System & Prompt Builder
# ---------------------------------------------------------------------------

class PromptWeaver:
    """
    Converts Step-1 JSON analysis into sectioned natural-language prompts
    with style guardrails and verification flags.
    """

    def __init__(self, style_config_path: Optional[Path] = None, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        
        # Load style configuration
        if style_config_path and style_config_path.exists():
            with open(style_config_path, 'r') as f:
                self.style_config = yaml.safe_load(f)
        else:
            self.style_config = self._default_style_config()
        
        logger.info(f"PromptWeaver initialized with seed {seed}")

    def weave_prompt(
        self,
        json_analysis: Dict[str, Any],
        template_id: str = "photostudio_ghost_mannequin_v2",
        temperature: float = 0.0,
        top_p: float = 0.15,
        verification_level: str = "high"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Convert JSON analysis to sectioned prompt.
        Returns: (prompt_text, meta_dict)
        """
        
        # Extract key information from JSON
        garment_info = self._extract_garment_info(json_analysis)
        
        # Build sectioned prompt
        sections = []
        
        # Section 1: Template inheritance & verification
        sections.append(self._build_template_section(template_id, verification_level))
        
        # Section 2: Non-negotiable facts (paraphrased JSON)
        sections.append(self._build_facts_section(garment_info))
        
        # Section 3: Mandatory style guide
        sections.append(self._build_style_guide_section())
        
        # Section 4: Garment-aware rules
        sections.append(self._build_garment_rules_section(garment_info))
        
        # Section 5: Forbidden edits
        sections.append(self._build_forbidden_section())
        
        # Section 6: Critical quality checks
        sections.append(self._build_quality_checks_section(garment_info))
        
        # Join sections
        prompt_text = "\n\n".join(sections)
        
        # Build metadata
        meta = {
            "controls": {
                "temperature": temperature,
                "top_p": top_p,
                "seed": self.seed,
                "verification_level": verification_level
            },
            "notes": {
                "template_id": template_id,
                "inheritance_chain": ["base_product_photography", "ghost_mannequin_specific"],
                "category": garment_info.get("category", "unknown"),
                "specific_type": garment_info.get("specific_type", "unknown"),
                "primary_hex": garment_info.get("primary_hex", "#000000")
            }
        }
        
        return prompt_text, meta

    def _extract_garment_info(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key garment information from JSON analysis."""
        
        # Primary classification
        garment_cat = json_data.get("garment_categorization", {})
        primary_class = garment_cat.get("primary_classification", {})
        style_attrs = garment_cat.get("style_attributes", {})
        
        # Color information
        color_data = json_data.get("color_extraction", {})
        primary_color = color_data.get("primary_color", {})
        
        # Construction details
        construction = json_data.get("construction_details", {})
        closures = construction.get("closures", {})
        features = construction.get("critical_features", {})
        
        # Fabric analysis
        fabric = json_data.get("fabric_analysis", {})
        texture = fabric.get("texture_classification", {})
        material = fabric.get("material_detection", {})
        
        return {
            "category": primary_class.get("category", "unknown"),
            "specific_type": primary_class.get("specific_type", "unknown"),
            "subcategory": primary_class.get("subcategory", "unknown"),
            "silhouette": style_attrs.get("silhouette", "unknown"),
            "neckline": style_attrs.get("neckline", "unknown"),
            "sleeves": style_attrs.get("sleeves", "unknown"),
            "hemline": style_attrs.get("hemline", "unknown"),
            "fit_type": style_attrs.get("fit_type", "unknown"),
            "primary_hex": primary_color.get("hex_value", "#000000"),
            "color_name": primary_color.get("color_name", "unknown"),
            "closure_type": closures.get("primary_closure", "unknown"),
            "closure_count": closures.get("closure_count", 0),
            "has_pockets": features.get("pockets", {}).get("present", False),
            "pocket_type": features.get("pockets", {}).get("type", "none"),
            "primary_texture": texture.get("primary_texture", "unknown"),
            "fabric_weight": texture.get("fabric_weight", "unknown"),
            "material_type": material.get("material_type", "unknown"),
            "drape_behavior": material.get("drape_behavior", "unknown"),
        }

    def _build_template_section(self, template_id: str, verification_level: str) -> str:
        """Section 1: Template inheritance & verification flags."""
        return f"""TEMPLATE: {template_id}
INHERITANCE: base_product_photography → ghost_mannequin_specific
VERIFICATION_LEVEL: {verification_level}
DETERMINISTIC_SEED: {self.seed}"""

    def _build_facts_section(self, info: Dict[str, Any]) -> str:
        """Section 2: Non-negotiable facts (JSON paraphrased)."""
        facts = []
        
        # Basic classification
        if info["category"] != "unknown":
            facts.append(f"GARMENT_CATEGORY: {info['category']}")
        if info["specific_type"] != "unknown":
            facts.append(f"GARMENT_TYPE: {info['specific_type']}")
        if info["subcategory"] != "unknown":
            facts.append(f"STYLE_CLASSIFICATION: {info['subcategory']}")
            
        # Physical attributes
        if info["silhouette"] != "unknown":
            facts.append(f"SILHOUETTE: {info['silhouette']}")
        if info["fit_type"] != "unknown":
            facts.append(f"FIT: {info['fit_type']}")
            
        # Specific garment features
        if info["neckline"] != "unknown":
            facts.append(f"NECKLINE: {info['neckline']}")
        if info["sleeves"] != "unknown":
            facts.append(f"SLEEVES: {info['sleeves']}")
        if info["hemline"] != "unknown":
            facts.append(f"HEMLINE: {info['hemline']}")
            
        # Color specification
        facts.append(f"PRIMARY_COLOR: {info['primary_hex']} ({info['color_name']})")
        
        # Construction details
        if info["closure_type"] != "unknown":
            if info["closure_count"] > 0:
                facts.append(f"CLOSURES: {info['closure_type']} (count: {info['closure_count']})")
            else:
                facts.append(f"CLOSURES: {info['closure_type']}")
                
        if info["has_pockets"] and info["pocket_type"] != "none":
            facts.append(f"POCKETS: {info['pocket_type']}")
            
        # Fabric characteristics
        if info["primary_texture"] != "unknown":
            facts.append(f"TEXTURE: {info['primary_texture']}")
        if info["fabric_weight"] != "unknown":
            facts.append(f"FABRIC_WEIGHT: {info['fabric_weight']}")
        if info["material_type"] != "unknown":
            facts.append(f"MATERIAL: {info['material_type']}")
        if info["drape_behavior"] != "unknown":
            facts.append(f"DRAPE: {info['drape_behavior']}")
            
        return "NON-NEGOTIABLE FACTS:\n" + "\n".join(f"- {fact}" for fact in facts)

    def _build_style_guide_section(self) -> str:
        """Section 3: Mandatory style guide."""
        style = self.style_config.get("style_guide", {})
        
        return f"""MANDATORY STYLE GUIDE:
- BACKGROUND: {style.get('background', 'pure white (#FFFFFF), seamless, no shadows')}
- LIGHTING: {style.get('lighting', 'high-key, even, diffused, professional studio lighting')}
- RESOLUTION: {style.get('resolution', '2048×2048 pixels minimum')}
- COLOR_SPACE: {style.get('color_space', 'sRGB')}
- FRAME_FILL: {style.get('frame_fill', '85% garment coverage, centered')}
- PERSPECTIVE: {style.get('perspective', 'front-facing, straight-on view')}
- FOCUS: {style.get('focus', 'sharp focus throughout, no blur')}
- PRESENTATION: {style.get('presentation', 'ghost mannequin effect, hollow interior visible')}"""

    def _build_garment_rules_section(self, info: Dict[str, Any]) -> str:
        """Section 4: Garment-aware rules."""
        rules = []
        
        # Category-specific rules
        category = info["category"]
        if category == "tops":
            rules.append("HOLLOW_NECK: Neckline must show clear hollow interior opening")
            rules.append("SLEEVE_OPENINGS: Sleeve ends must show hollow ghost effect")
            rules.append("HEM_STRUCTURE: Bottom hem should maintain garment's natural drape")
        elif category == "bottoms":
            rules.append("WAISTBAND: Must show proper waistband structure and hollow opening")
            rules.append("LEG_OPENINGS: Leg ends must show hollow ghost effect")
        elif category == "dresses":
            rules.append("HOLLOW_NECK: Neckline must show clear hollow interior opening") 
            rules.append("SLEEVE_TREATMENT: Sleeves (if present) must show hollow openings")
            rules.append("SKIRT_DRAPE: Lower portion must show natural dress drape")
        elif category == "outerwear":
            rules.append("CLOSURE_VISIBILITY: Zippers/buttons must remain clearly visible")
            rules.append("COLLAR_STRUCTURE: Collar must maintain proper shape and hollow effect")
            rules.append("POCKET_INTEGRITY: Existing pockets must remain visible and structured")
            
        # Closure-specific rules
        if info["closure_type"] == "zipper":
            rules.append("ZIPPER_INTEGRITY: Zipper must remain straight, unmodified, fully visible")
        elif info["closure_type"] == "buttons":
            if info["closure_count"] > 0:
                rules.append(f"BUTTON_PRESERVATION: All {info['closure_count']} buttons must remain visible and aligned")
        
        # Pocket rules
        if info["has_pockets"]:
            rules.append("POCKET_STRUCTURE: Existing pockets must maintain their shape and visibility")
            
        # Texture-specific rules
        texture = info["primary_texture"]
        if texture == "ribbed" or "ribbed" in texture:
            rules.append("RIBBING_PRESERVATION: Ribbed texture must remain clearly visible")
        elif texture == "smooth":
            rules.append("SMOOTH_FINISH: Surface must maintain smooth, clean appearance")
            
        return "GARMENT-AWARE RULES:\n" + "\n".join(f"- {rule}" for rule in rules)

    def _build_forbidden_section(self) -> str:
        """Section 5: Forbidden edits."""
        forbidden = self.style_config.get("forbidden_edits", [])
        
        default_forbidden = [
            "DO NOT invent or add features not present in original garment",
            "DO NOT relocate or modify brand labels or tags", 
            "DO NOT alter the primary color specified in facts",
            "DO NOT add patterns, prints, or graphics not in original",
            "DO NOT modify closure count or type",
            "DO NOT add or remove pockets",
            "DO NOT change neckline style or shape",
            "DO NOT alter sleeve length or style",
            "DO NOT add embellishments not present in original",
            "DO NOT modify garment proportions or silhouette"
        ]
        
        all_forbidden = forbidden + default_forbidden
        return "FORBIDDEN EDITS:\n" + "\n".join(f"- {rule}" for rule in all_forbidden)

    def _build_quality_checks_section(self, info: Dict[str, Any]) -> str:
        """Section 6: Critical quality checks."""
        quality_checks = self.style_config.get("quality_checks", {})
        
        primary_hex = info["primary_hex"]
        
        return f"""CRITICAL QUALITY CHECKS:
- COLOR_ACCURACY: Final output must match primary color {primary_hex} within ΔE2000 < 2.0
- RESOLUTION_CHECK: Output must be exactly 2048×2048 pixels
- HOLLOW_EFFECT: Ghost mannequin interior must be clearly visible and properly lit
- BACKGROUND_PURITY: Background must be pure white with no shadows or artifacts
- EDGE_QUALITY: Garment edges must be clean and sharp, no halos or artifacts
- LABEL_PRESERVATION: Any visible brand labels must remain legible and unmodified
- SYMMETRY_CHECK: Garment must appear properly centered and symmetrical
- LIGHTING_CONSISTENCY: No harsh shadows, even illumination throughout"""

    def _default_style_config(self) -> Dict[str, Any]:
        """Default style configuration when no config file is provided."""
        return {
            "style_guide": {
                "background": "pure white (#FFFFFF), seamless, no shadows",
                "lighting": "high-key, even, diffused, professional studio lighting",
                "resolution": "2048×2048 pixels minimum",
                "color_space": "sRGB",
                "frame_fill": "85% garment coverage, centered", 
                "perspective": "front-facing, straight-on view",
                "focus": "sharp focus throughout, no blur",
                "presentation": "ghost mannequin effect, hollow interior visible"
            },
            "forbidden_edits": [],
            "quality_checks": {
                "delta_e_threshold": 2.0,
                "resolution_exact": True,
                "hollow_effect_required": True
            }
        }

# ---------------------------------------------------------------------------
# Template Variations for A/B Testing
# ---------------------------------------------------------------------------

class PromptVariationGenerator:
    """Generate prompt variants that keep constraints identical but vary phrasing."""
    
    def __init__(self, base_weaver: PromptWeaver):
        self.base_weaver = base_weaver
        
    def generate_variants(
        self,
        json_analysis: Dict[str, Any],
        count: int = 3,
        **weave_kwargs
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Generate multiple prompt variants with different phrasing."""
        
        variants = []
        for i in range(count):
            # Use different random seed for each variant to vary phrasing
            original_seed = self.base_weaver.seed
            self.base_weaver.seed = original_seed + i
            random.seed(self.base_weaver.seed)
            
            prompt, meta = self.base_weaver.weave_prompt(json_analysis, **weave_kwargs)
            
            # Apply phrasing variations
            varied_prompt = self._apply_phrasing_variations(prompt, i)
            
            # Update metadata to indicate variant
            meta["notes"]["variant_id"] = i
            meta["notes"]["base_seed"] = original_seed
            meta["controls"]["seed"] = self.base_weaver.seed
            
            variants.append((varied_prompt, meta))
            
        # Restore original seed
        self.base_weaver.seed = original_seed
        random.seed(original_seed)
        
        return variants
        
    def _apply_phrasing_variations(self, prompt: str, variant_id: int) -> str:
        """Apply subtle phrasing variations without changing constraints."""
        
        variations = {
            0: {  # Base variant - no changes
            },
            1: {  # Formal variant
                "must": "shall",
                "should": "must",
                "ensure": "guarantee",
                "maintain": "preserve",
                "clearly": "distinctly"
            },
            2: {  # Action-oriented variant  
                "must show": "should display",
                "must remain": "needs to stay",
                "must be": "should be",
                "maintain": "keep",
                "preserve": "retain"
            }
        }
        
        if variant_id not in variations:
            return prompt
            
        var_map = variations[variant_id]
        varied_prompt = prompt
        
        for original, replacement in var_map.items():
            varied_prompt = varied_prompt.replace(original, replacement)
            
        return varied_prompt

# ---------------------------------------------------------------------------
# CLI Interface  
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Step 2/6: Prompt Weaver - Convert JSON analysis to sectioned prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", required=True, help="Path to garment_analysis.json from Step 1")
    parser.add_argument("--out", default="./step2", help="Output directory")
    parser.add_argument("--style-config", help="Path to style_kit.yml configuration file")
    parser.add_argument("--template-id", default="photostudio_ghost_mannequin_v2", help="Template identifier")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for deterministic output")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--top-p", type=float, default=0.15, help="Generation top-p")
    parser.add_argument("--verification-level", default="high", choices=["low", "medium", "high"], 
                       help="Verification level for quality checks")
    parser.add_argument("--variants", type=int, default=1, help="Number of prompt variants to generate")

    args = parser.parse_args()

    # Create output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load JSON analysis
    with open(args.json, 'r', encoding='utf-8') as f:
        json_analysis = json.load(f)

    # Load style configuration if provided
    style_config_path = Path(args.style_config) if args.style_config else None

    # Initialize prompt weaver
    weaver = PromptWeaver(
        style_config_path=style_config_path,
        seed=args.seed
    )

    try:
        if args.variants == 1:
            # Single prompt
            prompt, meta = weaver.weave_prompt(
                json_analysis=json_analysis,
                template_id=args.template_id,
                temperature=args.temperature,
                top_p=args.top_p,
                verification_level=args.verification_level
            )

            # Save outputs
            with open(out_dir / "prompt.txt", 'w', encoding='utf-8') as f:
                f.write(prompt)

            with open(out_dir / "prompt.meta.json", 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2)

            print(f"✅ Prompt generated! Output saved to: {args.out}")
            print(f"📝 Template: {args.template_id}")
            print(f"🎯 Category: {meta['notes']['category']}")
            print(f"🎨 Primary Color: {meta['notes']['primary_hex']}")

        else:
            # Multiple variants
            variant_gen = PromptVariationGenerator(weaver)
            variants = variant_gen.generate_variants(
                json_analysis=json_analysis,
                count=args.variants,
                template_id=args.template_id,
                temperature=args.temperature,
                top_p=args.top_p,
                verification_level=args.verification_level
            )

            for i, (prompt, meta) in enumerate(variants):
                # Save each variant
                with open(out_dir / f"prompt_variant_{i}.txt", 'w', encoding='utf-8') as f:
                    f.write(prompt)

                with open(out_dir / f"prompt_variant_{i}.meta.json", 'w', encoding='utf-8') as f:
                    json.dump(meta, f, indent=2)

            print(f"✅ {len(variants)} prompt variants generated! Output saved to: {args.out}")
            print(f"📝 Template: {args.template_id}")
            print(f"🎯 Category: {variants[0][1]['notes']['category']}")
            print(f"🎨 Primary Color: {variants[0][1]['notes']['primary_hex']}")

    except Exception as e:
        logger.error(f"Prompt generation failed: {e}")
        raise


if __name__ == "__main__":
    main()
