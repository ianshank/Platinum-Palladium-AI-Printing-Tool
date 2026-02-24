"""
Domain-specific prompts for Pt/Pd calibration assistance.
"""

SYSTEM_PROMPT = """You are an expert in platinum/palladium alternative photographic printing processes.
You have deep knowledge of:

1. **Sensitizer Chemistry**
   - Platinum and palladium metal salts (ferric oxalate, platinum/palladium solutions)
   - The difference between "traditional" Pt/Pd (separate solutions) and pre-mixed solutions
   - Contrast control agents like Na2 (sodium chloroplatinate), potassium chlorate
   - How metal ratios affect image tone (Pt = cooler/neutral, Pd = warmer)

2. **Paper Selection**
   - Cotton rag papers (Arches Platine, Bergger COT320, Hahnemühle Platinum Rag)
   - Japanese papers (Gampi, Kozo, Mitsumata)
   - Sizing requirements and effects on print quality
   - Paper characteristics that affect Dmax, clearing, and tonal range

3. **Exposure & Development**
   - UV light sources (sun, BLB tubes, metal halide, LED arrays)
   - Exposure calculation and calibration
   - Developers (potassium oxalate, ammonium citrate) and their characteristics
   - Clearing chemistry and procedures

4. **Calibration & Digital Negatives**
   - Step tablet reading and density measurement
   - Linearization curves for digital negatives
   - QTR (QuadTone RIP) and Piezography workflows
   - Color management for alternative processes

5. **Troubleshooting**
   - Common problems: bronzing, staining, solarization, blocked shadows
   - Chemical contamination issues
   - Paper-related problems
   - Environmental factors (humidity, temperature)

When helping users:
- Be specific and technical when appropriate
- Provide practical, actionable advice
- Reference specific products and materials when relevant
- Explain the "why" behind recommendations
- Acknowledge uncertainty when you don't have definitive answers
- Consider safety when discussing chemicals

Always prioritize helping the user achieve beautiful prints while working safely."""


def get_analysis_prompt(calibration_data: dict) -> str:
    """
    Generate prompt for analyzing a calibration result.

    Args:
        calibration_data: Dictionary with calibration details.

    Returns:
        Formatted prompt for analysis.
    """
    return f"""Please analyze this Pt/Pd calibration result and provide insights:

**Paper:** {calibration_data.get("paper_type", "Unknown")}
**Chemistry:** {calibration_data.get("chemistry_type", "Unknown")}
**Metal Ratio:** {calibration_data.get("metal_ratio", 0.5):.0%} Pt / {1 - calibration_data.get("metal_ratio", 0.5):.0%} Pd
**Contrast Agent:** {calibration_data.get("contrast_agent", "None")} ({calibration_data.get("contrast_amount", 0)} drops)
**Developer:** {calibration_data.get("developer", "Unknown")}
**Exposure Time:** {calibration_data.get("exposure_time", 0):.0f} seconds

**Measured Densities:**
- Dmin (paper base): {calibration_data.get("dmin", 0):.2f}
- Dmax: {calibration_data.get("dmax", 0):.2f}
- Density Range: {calibration_data.get("density_range", 0):.2f}

**Additional Observations:**
{calibration_data.get("notes", "None provided")}

Please provide:
1. Assessment of the overall calibration quality
2. Analysis of the density range and tonal characteristics
3. Suggestions for improvement if needed
4. Any concerns or potential issues identified"""


def get_recipe_prompt(paper_type: str, desired_characteristics: str) -> str:
    """
    Generate prompt for recipe suggestion.

    Args:
        paper_type: Target paper type.
        desired_characteristics: Desired print characteristics.

    Returns:
        Formatted prompt for recipe suggestion.
    """
    return f"""Please suggest a starting Pt/Pd coating recipe for:

**Paper:** {paper_type}
**Desired Characteristics:** {desired_characteristics}

Please provide:
1. Recommended metal ratio (Pt:Pd)
2. Suggested contrast agent and amount
3. Recommended developer
4. Starting exposure time estimate
5. Any paper-specific preparation needed
6. Tips for this particular combination

Include your reasoning for each recommendation."""


def get_troubleshooting_prompt(problem_description: str) -> str:
    """
    Generate prompt for troubleshooting.

    Args:
        problem_description: Description of the problem.

    Returns:
        Formatted prompt for troubleshooting.
    """
    return f"""I'm having a problem with my Pt/Pd print:

**Problem Description:**
{problem_description}

Please help me troubleshoot by:
1. Identifying possible causes (list in order of likelihood)
2. Providing diagnostic steps to narrow down the issue
3. Suggesting solutions for each possible cause
4. Recommending preventive measures for the future

If you need more information to diagnose the problem, please ask specific questions."""


def get_comparison_prompt(record1: dict, record2: dict) -> str:
    """
    Generate prompt for comparing two calibrations.

    Args:
        record1: First calibration record.
        record2: Second calibration record.

    Returns:
        Formatted prompt for comparison.
    """
    return f"""Please compare these two Pt/Pd calibration results:

**Calibration 1:**
- Paper: {record1.get("paper_type", "Unknown")}
- Metal Ratio: {record1.get("metal_ratio", 0.5):.0%} Pt
- Contrast Agent: {record1.get("contrast_agent", "None")} ({record1.get("contrast_amount", 0)} drops)
- Exposure: {record1.get("exposure_time", 0):.0f}s
- Dmax: {record1.get("dmax", 0):.2f}
- Density Range: {record1.get("density_range", 0):.2f}

**Calibration 2:**
- Paper: {record2.get("paper_type", "Unknown")}
- Metal Ratio: {record2.get("metal_ratio", 0.5):.0%} Pt
- Contrast Agent: {record2.get("contrast_agent", "None")} ({record2.get("contrast_amount", 0)} drops)
- Exposure: {record2.get("exposure_time", 0):.0f}s
- Dmax: {record2.get("dmax", 0):.2f}
- Density Range: {record2.get("density_range", 0):.2f}

Please provide:
1. Key differences between the two calibrations
2. What caused the differences in results
3. Which approach is better for different use cases
4. Recommendations for achieving specific goals"""


def get_paper_recommendation_prompt(requirements: dict) -> str:
    """
    Generate prompt for paper recommendation.

    Args:
        requirements: Dictionary with user requirements.

    Returns:
        Formatted prompt for paper recommendation.
    """
    return f"""I need help choosing a paper for Pt/Pd printing:

**Requirements:**
- Budget: {requirements.get("budget", "Not specified")}
- Print size: {requirements.get("print_size", "Not specified")}
- Desired tone: {requirements.get("tone", "Not specified")}
- Experience level: {requirements.get("experience", "Not specified")}
- Desired Dmax: {requirements.get("dmax", "Not specified")}
- Special requirements: {requirements.get("special", "None")}

Please recommend:
1. Top 3 paper choices with pros and cons
2. Which paper is best for my specific needs
3. Preparation requirements for each paper
4. Expected results and characteristics
5. Where to purchase (if commonly available)"""
