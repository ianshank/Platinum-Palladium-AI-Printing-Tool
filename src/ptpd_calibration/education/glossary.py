"""
Glossary of Pt/Pd printing terminology and concepts.

Provides comprehensive reference for technical terms, processes, and concepts
used in platinum/palladium printing.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TermCategory(str, Enum):
    """Categories for glossary terms."""

    CHEMISTRY = "chemistry"
    PROCESS = "process"
    MEASUREMENT = "measurement"
    MATERIALS = "materials"
    EQUIPMENT = "equipment"
    QUALITY = "quality"
    ARTISTIC = "artistic"
    TROUBLESHOOTING = "troubleshooting"


class GlossaryTerm(BaseModel):
    """A single glossary term with definition and metadata."""

    term: str = Field(..., min_length=1, max_length=200, description="The term itself")
    definition: str = Field(..., min_length=1, description="Detailed definition")
    category: TermCategory = Field(..., description="Term category")
    related_terms: list[str] = Field(
        default_factory=list, description="Related terminology"
    )
    examples: list[str] = Field(
        default_factory=list, description="Usage examples or notes"
    )
    synonyms: list[str] = Field(default_factory=list, description="Alternative names")
    see_also: list[str] = Field(
        default_factory=list, description="Cross-references to other terms"
    )


# Comprehensive glossary database
GLOSSARY_DATA = {
    # Chemistry Terms
    "ferric_oxalate": {
        "term": "Ferric Oxalate",
        "definition": (
            "Iron(III) oxalate, the primary sensitizer in Pt/Pd printing. A light-sensitive "
            "compound that, when exposed to UV light, reduces to ferrous oxalate, which then "
            "reduces platinum and palladium salts to metallic form. Typically used as a 24-27% "
            "solution."
        ),
        "category": TermCategory.CHEMISTRY,
        "related_terms": ["sensitizer", "ferrous_oxalate", "reduction"],
        "examples": [
            "Standard concentration is 24%",
            "Store in amber bottle away from light",
            "Mix fresh solution every few months for best results",
        ],
        "synonyms": ["iron oxalate"],
        "see_also": ["sensitizer", "light_sensitive"],
    },
    "platinum_chloride": {
        "term": "Platinum Chloride",
        "definition": (
            "Chloroplatinic acid (H2PtCl6), one of the two precious metal salts used in Pt/Pd "
            "printing. Produces cooler, more neutral tones. Typically used as 15-20% solution. "
            "More expensive than palladium but provides higher contrast and neutral tones."
        ),
        "category": TermCategory.CHEMISTRY,
        "related_terms": ["palladium_chloride", "metal_ratio", "contrast"],
        "examples": [
            "Pure platinum prints have cool, neutral tones",
            "Standard stock solution is 15%",
            "More contrast than palladium alone",
        ],
        "synonyms": ["chloroplatinic acid", "Pt salt"],
        "see_also": ["palladium_chloride", "metal_ratio"],
    },
    "palladium_chloride": {
        "term": "Palladium Chloride",
        "definition": (
            "Palladium(II) chloride (PdCl2), one of the two precious metal salts used in Pt/Pd "
            "printing. Produces warmer, brown tones. Typically used as 7-15% solution. Less "
            "expensive than platinum with gentler contrast characteristics."
        ),
        "category": TermCategory.CHEMISTRY,
        "related_terms": ["platinum_chloride", "metal_ratio", "tone"],
        "examples": [
            "Pure palladium prints have warm, brown tones",
            "Common stock solution is 7%",
            "More affordable option for practice prints",
        ],
        "synonyms": ["Pd salt"],
        "see_also": ["platinum_chloride", "metal_ratio"],
    },
    "metal_ratio": {
        "term": "Metal Ratio",
        "definition": (
            "The proportion of platinum to palladium in the sensitizer mixture, expressed as "
            "percentage or ratio. Common ratios: 100/0 (pure Pt), 50/50 (equal mix), 0/100 "
            "(pure Pd). Affects print tone, contrast, and cost. Higher Pt = cooler, higher "
            "contrast; higher Pd = warmer, gentler contrast."
        ),
        "category": TermCategory.CHEMISTRY,
        "related_terms": ["platinum_chloride", "palladium_chloride", "tone", "contrast"],
        "examples": [
            "50/50 is most popular for balanced characteristics",
            "70/30 Pt/Pd for cooler tones",
            "30/70 Pt/Pd for warmer, more economical prints",
        ],
        "see_also": ["platinum_chloride", "palladium_chloride"],
    },
    "potassium_oxalate": {
        "term": "Potassium Oxalate",
        "definition": (
            "The most common developer for Pt/Pd prints. Typically used as 20-25% solution at "
            "68-75°F (20-24°C). Reduces the exposed ferrous oxalate to metallic platinum and "
            "palladium, making the image visible. Development typically completes in 1-3 minutes."
        ),
        "category": TermCategory.CHEMISTRY,
        "related_terms": ["developer", "ammonium_citrate", "development"],
        "examples": [
            "Mix 200g potassium oxalate in 1L water for 20% solution",
            "Warmer developer = warmer print tones",
            "Use fresh developer for best results",
        ],
        "synonyms": ["K2C2O4", "developer"],
        "see_also": ["developer", "development"],
    },
    "ammonium_citrate": {
        "term": "Ammonium Citrate",
        "definition": (
            "Alternative developer to potassium oxalate, producing warmer print tones. Used at "
            "similar concentrations (20-25%) but gives slightly different tonal characteristics. "
            "Some printers prefer it for warmer, more luminous prints."
        ),
        "category": TermCategory.CHEMISTRY,
        "related_terms": ["developer", "potassium_oxalate", "sodium_citrate"],
        "examples": [
            "Produces warmer tones than potassium oxalate",
            "Popular for portrait work",
            "Temperature affects tone more than with oxalate",
        ],
        "synonyms": ["ammonium citrate developer"],
        "see_also": ["developer", "potassium_oxalate"],
    },
    "edta": {
        "term": "EDTA",
        "definition": (
            "Ethylenediaminetetraacetic acid, the preferred clearing agent for modern Pt/Pd "
            "printing. Used as 1-2% solution to remove unexposed sensitizer and iron compounds "
            "from the print. More effective and archivally sound than traditional citric acid "
            "clearing. Typical clearing time is 5-10 minutes with agitation."
        ),
        "category": TermCategory.CHEMISTRY,
        "related_terms": ["clearing", "citric_acid", "archival"],
        "examples": [
            "Use 1% solution for clearing",
            "Clears more thoroughly than citric acid",
            "Essential for archival permanence",
        ],
        "synonyms": ["clearing agent", "tetrasodium EDTA"],
        "see_also": ["clearing", "yellow_staining"],
    },
    "contrast_agent": {
        "term": "Contrast Agent",
        "definition": (
            "Chemicals added to sensitizer to increase print contrast and maximum density. "
            "Common agents: NA2 (sodium chloroplatinate), potassium chlorate, hydrogen peroxide, "
            "dichromate. Use sparingly - start with 1-4 drops per standard sensitizer batch. "
            "Affects exposure time and print character."
        ),
        "category": TermCategory.CHEMISTRY,
        "related_terms": ["na2", "contrast", "dmax"],
        "examples": [
            "NA2 is most powerful, start with 1 drop",
            "Potassium chlorate gives coolest tones",
            "Dichromate produces warmest tones",
        ],
        "see_also": ["na2", "contrast"],
    },
    "na2": {
        "term": "NA2",
        "definition": (
            "Sodium chloroplatinate (Na2PtCl6), a very powerful contrast-enhancing agent. "
            "Dramatically increases maximum density and contrast. Start with minimal amounts "
            "(1-2 drops per 4ml sensitizer) and increase carefully. Can also be used as primary "
            "metal salt in specialized high-contrast processes."
        ),
        "category": TermCategory.CHEMISTRY,
        "related_terms": ["contrast_agent", "dmax", "contrast"],
        "examples": [
            "Start with 1 drop and test",
            "Can double or triple Dmax",
            "Affects exposure time - may need longer exposure",
        ],
        "synonyms": ["sodium chloroplatinate"],
        "see_also": ["contrast_agent", "contrast"],
    },
    # Process Terms
    "sizing": {
        "term": "Sizing",
        "definition": (
            "Surface treatment applied to paper to control absorbency and allow even sensitizer "
            "coating. Common sizing agents: gelatin (traditional), arrowroot starch (vegan), "
            "commercially sized papers. Proper sizing is essential for even coating and preventing "
            "sensitizer from sinking into paper fibers."
        ),
        "category": TermCategory.PROCESS,
        "related_terms": ["gelatin_sizing", "arrowroot", "coating"],
        "examples": [
            "2-3% gelatin solution is traditional",
            "Arrowroot is popular vegan alternative",
            "Arches Platine comes pre-sized",
        ],
        "see_also": ["coating", "paper"],
    },
    "coating": {
        "term": "Coating",
        "definition": (
            "Application of light-sensitive emulsion to paper. In Pt/Pd printing, sensitizer is "
            "coated onto sized paper using glass rod, brush, or spray. Must be done in subdued "
            "lighting. Even coating is critical for uniform print quality. Paper must dry "
            "completely before exposure."
        ),
        "category": TermCategory.PROCESS,
        "related_terms": ["sizing", "glass_rod", "sensitizer"],
        "examples": [
            "Glass rod coating is most common method",
            "Work quickly to prevent drying during application",
            "Single smooth stroke produces best results",
        ],
        "see_also": ["glass_rod", "sizing"],
    },
    "exposure": {
        "term": "Exposure",
        "definition": (
            "UV light exposure of coated paper through negative to create latent image. UV light "
            "reduces ferric oxalate to ferrous, which then reduces metal salts during development. "
            "Exposure time varies by UV source, chemistry, humidity, and negative density. "
            "Proper exposure is critical for full density range."
        ),
        "category": TermCategory.PROCESS,
        "related_terms": ["uv_light", "negative", "exposure_time"],
        "examples": [
            "Sunlight: typically 5-15 minutes",
            "UV LED: typically 3-10 minutes",
            "Test with step tablet to determine time",
        ],
        "see_also": ["uv_light", "contact_printing"],
    },
    "development": {
        "term": "Development",
        "definition": (
            "Chemical process that makes the latent image visible. Coated and exposed paper is "
            "immersed in developer solution (typically potassium oxalate), which reduces the "
            "exposed metal salts to metallic platinum and palladium. Development is complete when "
            "no further image change occurs, typically 1-3 minutes."
        ),
        "category": TermCategory.PROCESS,
        "related_terms": ["developer", "potassium_oxalate", "agitation"],
        "examples": [
            "Immerse print face-down in developer",
            "Agitate gently and continuously",
            "Image appears and deepens during development",
        ],
        "see_also": ["potassium_oxalate", "developer"],
    },
    "clearing": {
        "term": "Clearing",
        "definition": (
            "Removal of unexposed sensitizer and iron compounds from developed print. Essential "
            "for archival stability and clean highlights. Modern clearing uses EDTA (1-2% solution) "
            "for 5-15 minutes with agitation, followed by thorough water rinses. Incomplete "
            "clearing causes yellow staining and poor archival stability."
        ),
        "category": TermCategory.PROCESS,
        "related_terms": ["edta", "yellow_staining", "archival"],
        "examples": [
            "Initial water rinse: 2-5 minutes",
            "EDTA clearing: 5-10 minutes",
            "Final wash: 15-20 minutes running water",
        ],
        "see_also": ["edta", "yellow_staining"],
    },
    "contact_printing": {
        "term": "Contact Printing",
        "definition": (
            "Printing method where negative is placed in direct contact with coated paper during "
            "exposure. Requires same-size negative as desired final print. Perfect contact is "
            "essential for sharp prints. Uses contact printing frame with spring back or vacuum "
            "contact to ensure uniform pressure."
        ),
        "category": TermCategory.PROCESS,
        "related_terms": ["negative", "exposure", "printing_frame"],
        "examples": [
            "Negative must be same size as final print",
            "Check for air bubbles or gaps",
            "Use weighted glass or sprung frame for pressure",
        ],
        "see_also": ["exposure", "negative"],
    },
    # Measurement Terms
    "dmax": {
        "term": "Dmax",
        "definition": (
            "Maximum density - the darkest value a print can achieve. In Pt/Pd printing, Dmax is "
            "affected by metal ratio, contrast agents, exposure, and paper characteristics. "
            "Typical Pt/Pd Dmax ranges from 1.6-2.1 for visual density. Higher values indicate "
            "deeper blacks and longer tonal range."
        ),
        "category": TermCategory.MEASUREMENT,
        "related_terms": ["dmin", "density_range", "contrast"],
        "examples": [
            "Pure Pt typically achieves higher Dmax than pure Pd",
            "NA2 can increase Dmax significantly",
            "Measured with densitometer or derived from scans",
        ],
        "see_also": ["dmin", "density_range"],
    },
    "dmin": {
        "term": "Dmin",
        "definition": (
            "Minimum density - the lightest value in a print, typically the paper base with any "
            "residual sensitizer staining. Lower Dmin indicates cleaner highlights. Proper clearing "
            "is essential for achieving low Dmin. Paper choice and base color also affect Dmin."
        ),
        "category": TermCategory.MEASUREMENT,
        "related_terms": ["dmax", "density_range", "clearing"],
        "examples": [
            "Ideal Dmin should be close to unsized paper base",
            "Yellow staining increases Dmin",
            "Typical Dmin: 0.05-0.15",
        ],
        "see_also": ["dmax", "density_range", "clearing"],
    },
    "density_range": {
        "term": "Density Range",
        "definition": (
            "The difference between Dmax and Dmin - represents the total tonal range available "
            "in a print. Calculated as Dmax - Dmin. Typical Pt/Pd density range: 1.5-2.0. Wider "
            "range allows for more tonal gradation and 'depth' in prints. Also called Dynamic Range."
        ),
        "category": TermCategory.MEASUREMENT,
        "related_terms": ["dmax", "dmin", "contrast"],
        "examples": [
            "If Dmax=1.9 and Dmin=0.1, density range = 1.8",
            "Wider range = more tonal gradation possible",
            "Negative should match or be less than print density range",
        ],
        "synonyms": ["dynamic range", "tonal range"],
        "see_also": ["dmax", "dmin"],
    },
    "visual_density": {
        "term": "Visual Density",
        "definition": (
            "Density measurement weighted to match human visual perception, using the photopic "
            "luminosity function. Standard for Pt/Pd printing measurements. Different from Status A "
            "(color negative) or Status M (color transparency) density standards. Most densitometers "
            "and scanners can provide visual density readings."
        ),
        "category": TermCategory.MEASUREMENT,
        "related_terms": ["density", "dmax", "dmin"],
        "examples": [
            "Used for all Pt/Pd density measurements",
            "Matches how eye perceives tone",
            "Scanner readings can approximate visual density",
        ],
        "see_also": ["dmax", "dmin", "density_range"],
    },
    "linearization": {
        "term": "Linearization",
        "definition": (
            "Process of creating a correction curve so input tonal values map predictably to "
            "output densities. Essential because Pt/Pd printing is non-linear - 50% input doesn't "
            "produce 50% output density. Linearization involves printing step tablet, measuring "
            "results, and generating compensating curve to apply to images before printing."
        ),
        "category": TermCategory.MEASUREMENT,
        "related_terms": ["calibration", "curve", "step_tablet"],
        "examples": [
            "Print 21-step tablet to create curve",
            "Apply curve to images before printing",
            "Different curves needed for each paper/chemistry combination",
        ],
        "see_also": ["calibration", "curve"],
    },
    "step_tablet": {
        "term": "Step Tablet",
        "definition": (
            "Test image with series of uniform density steps, typically 21 steps from 0-100% in "
            "5% increments. Used for creating linearization curves and testing process variables. "
            "Printed, measured, and analyzed to determine actual density response of paper/chemistry "
            "combination."
        ),
        "category": TermCategory.MEASUREMENT,
        "related_terms": ["linearization", "calibration", "curve"],
        "examples": [
            "21-step tablet is standard (0%, 5%, 10%... 100%)",
            "Print at same size as final prints",
            "Scan and measure each step",
        ],
        "synonyms": ["density wedge", "gray scale"],
        "see_also": ["linearization", "calibration"],
    },
    "calibration": {
        "term": "Calibration",
        "definition": (
            "Process of characterizing and optimizing the printing system for predictable, "
            "consistent results. Involves printing step tablets, measuring output, creating "
            "linearization curves, and applying curves to images. Separate calibration needed "
            "for each paper/chemistry/process combination."
        ),
        "category": TermCategory.MEASUREMENT,
        "related_terms": ["linearization", "step_tablet", "curve"],
        "examples": [
            "Calibrate for each paper type",
            "Re-calibrate when changing chemistry ratios",
            "Environmental changes may require recalibration",
        ],
        "see_also": ["linearization", "step_tablet"],
    },
    # Material Terms
    "paper": {
        "term": "Paper",
        "definition": (
            "Substrate for Pt/Pd printing. Must be 100% cotton or other natural fiber. Paper "
            "choice affects tone, texture, Dmax, and working characteristics. Popular choices: "
            "Arches Platine (pre-sized), Bergger COT320, Hahnemühle Platinum Rag, Revere Platinum. "
            "Paper weight typically 250-320 GSM for handling during processing."
        ),
        "category": TermCategory.MATERIALS,
        "related_terms": ["sizing", "cotton", "substrate"],
        "examples": [
            "Arches Platine is most popular pre-sized paper",
            "Bergger COT320 for very smooth finish",
            "Cheaper papers good for testing and practice",
        ],
        "see_also": ["sizing", "coating"],
    },
    "negative": {
        "term": "Negative",
        "definition": (
            "Inverted tonal image used for contact printing. In Pt/Pd printing, typically a "
            "digital negative - inkjet printed on transparency film with calibrated curve applied. "
            "Must have sufficient density range (typically 1.4-1.8 Dmax) and be same size as "
            "desired final print. Quality of negative directly affects print quality."
        ),
        "category": TermCategory.MATERIALS,
        "related_terms": ["digital_negative", "contact_printing", "density_range"],
        "examples": [
            "Print with dedicated inkjet printer using pigment ink",
            "Pictorico OHP film is popular transparency",
            "Apply linearization curve before printing negative",
        ],
        "synonyms": ["digital negative"],
        "see_also": ["contact_printing", "linearization"],
    },
    "digital_negative": {
        "term": "Digital Negative",
        "definition": (
            "Negative created by inkjet printing digital image onto transparency film. Modern "
            "alternative to traditional film negatives. Requires: inkjet printer (Epson is standard), "
            "transparency film, proper ICC profiles or linearization curves, and pigment-based inks "
            "for density and longevity."
        ),
        "category": TermCategory.MATERIALS,
        "related_terms": ["negative", "linearization", "transparency"],
        "examples": [
            "Epson printers with K3 inks are popular",
            "Print at 1440 DPI or higher",
            "Must apply curve before printing",
        ],
        "synonyms": ["digital neg"],
        "see_also": ["negative", "linearization"],
    },
    # Equipment Terms
    "glass_rod": {
        "term": "Glass Rod",
        "definition": (
            "Cylindrical glass rod used to spread sensitizer evenly across paper. Standard coating "
            "tool for Pt/Pd printing. Typical diameter: 10-15mm. Must be perfectly smooth and clean. "
            "Technique involves pouring sensitizer across top of paper and drawing rod down in "
            "single smooth motion."
        ),
        "category": TermCategory.EQUIPMENT,
        "related_terms": ["coating", "puddle_pusher"],
        "examples": [
            "Clean thoroughly between coatings",
            "Single smooth stroke prevents streaking",
            "Can substitute acrylic rod or puddle pusher",
        ],
        "synonyms": ["coating rod"],
        "see_also": ["coating", "puddle_pusher"],
    },
    "printing_frame": {
        "term": "Contact Printing Frame",
        "definition": (
            "Frame used to hold negative in perfect contact with coated paper during UV exposure. "
            "Essential for sharp prints. Types: hinged back with springs, vacuum frame, weighted "
            "glass. Must provide even pressure across entire image area to prevent contact gaps "
            "that cause unsharpness."
        ),
        "category": TermCategory.EQUIPMENT,
        "related_terms": ["contact_printing", "exposure"],
        "examples": [
            "Spring-back frames provide good even pressure",
            "Vacuum frames ideal for large prints",
            "Check for perfect contact before exposure",
        ],
        "synonyms": ["contact frame"],
        "see_also": ["contact_printing", "exposure"],
    },
    "uv_light": {
        "term": "UV Light Source",
        "definition": (
            "Light source providing ultraviolet radiation for exposing Pt/Pd prints. Options: "
            "sunlight (free but variable), UV LED units (consistent, efficient), fluorescent UV "
            "tubes (traditional), metal halide (professional). UV wavelength around 365-405nm "
            "is optimal. Consistent output is important for repeatable results."
        ),
        "category": TermCategory.EQUIPMENT,
        "related_terms": ["exposure", "uv_meter"],
        "examples": [
            "UV LED units are popular for consistency",
            "Sunlight works but requires testing each session",
            "UV meter helps maintain consistent exposure",
        ],
        "synonyms": ["UV source", "exposure unit"],
        "see_also": ["exposure", "uv_meter"],
    },
    "uv_meter": {
        "term": "UV Meter",
        "definition": (
            "Device for measuring UV light intensity. Helps maintain consistent exposures by "
            "compensating for variations in UV source strength. Measures in units like μW/cm². "
            "Not essential but helpful for achieving repeatable results, especially with natural "
            "light or aging UV sources."
        ),
        "category": TermCategory.EQUIPMENT,
        "related_terms": ["uv_light", "exposure"],
        "examples": [
            "Compensate exposure time based on readings",
            "Track UV tube degradation over time",
            "Essential for scientific repeatability",
        ],
        "see_also": ["uv_light", "exposure"],
    },
    # Quality Terms
    "contrast": {
        "term": "Contrast",
        "definition": (
            "The difference between light and dark tones. In Pt/Pd printing, affected by metal "
            "ratio (Pt = higher contrast), contrast agents, development, and negative characteristics. "
            "Higher contrast = greater tonal separation but less gradation. Lower contrast = "
            "gentler tonal transitions."
        ),
        "category": TermCategory.QUALITY,
        "related_terms": ["metal_ratio", "contrast_agent", "density_range"],
        "examples": [
            "Pure platinum has higher contrast than palladium",
            "NA2 dramatically increases contrast",
            "Match contrast to image content",
        ],
        "see_also": ["metal_ratio", "contrast_agent"],
    },
    "tone": {
        "term": "Tone",
        "definition": (
            "The color character of a print - from warm brown to cool neutral. In Pt/Pd printing, "
            "primarily controlled by metal ratio: pure Pd = warm brown, 50/50 = neutral/slightly "
            "warm, pure Pt = cool neutral. Also affected by developer type, temperature, paper, "
            "and aging."
        ),
        "category": TermCategory.QUALITY,
        "related_terms": ["metal_ratio", "developer"],
        "examples": [
            "Palladium produces warmer, brown tones",
            "Platinum produces cooler, neutral tones",
            "Prints warm slightly with aging",
        ],
        "see_also": ["metal_ratio", "developer"],
    },
    "archival": {
        "term": "Archival Permanence",
        "definition": (
            "Long-term stability and longevity of prints. Pt/Pd prints are among the most archivally "
            "stable of all photographic processes when properly processed. Metallic platinum and "
            "palladium are extremely stable. Critical factors: complete clearing, proper washing, "
            "100% cotton paper, quality chemistry. Properly made Pt/Pd prints can last centuries."
        ),
        "category": TermCategory.QUALITY,
        "related_terms": ["clearing", "edta", "paper"],
        "examples": [
            "Pt/Pd prints from 1900s still in excellent condition",
            "Complete clearing is essential",
            "Use archival papers and proper washing",
        ],
        "see_also": ["clearing", "edta"],
    },
    # Troubleshooting Terms
    "yellow_staining": {
        "term": "Yellow Staining",
        "definition": (
            "Unwanted yellow coloration in print highlights caused by incomplete clearing of iron "
            "compounds. Indicates inadequate EDTA clearing time, exhausted clearing bath, or "
            "insufficient washing. Affects both aesthetics and archival stability. Prevention: "
            "adequate clearing time, fresh EDTA, thorough washing."
        ),
        "category": TermCategory.TROUBLESHOOTING,
        "related_terms": ["clearing", "edta", "dmin"],
        "examples": [
            "Extend EDTA clearing to 15 minutes if staining occurs",
            "Use fresh EDTA solution",
            "Can sometimes re-clear stained prints",
        ],
        "see_also": ["clearing", "edta"],
    },
    "fogging": {
        "term": "Fogging",
        "definition": (
            "Unwanted overall density or edge density caused by light exposure of sensitized paper "
            "before or during processing. Pt/Pd sensitizer is extremely light-sensitive. Causes: "
            "darkroom light leaks, improper storage of coated paper, light during coating/drying, "
            "or safelight too bright. Prevention: work in complete darkness or dim amber light."
        ),
        "category": TermCategory.TROUBLESHOOTING,
        "related_terms": ["light_sensitive", "darkroom"],
        "examples": [
            "Check darkroom for light leaks",
            "Store coated paper in darkness",
            "Even dim light can cause fogging",
        ],
        "see_also": ["light_sensitive", "coating"],
    },
    "streaking": {
        "term": "Streaking",
        "definition": (
            "Visible lines or uneven density in print caused by coating problems. Usually results "
            "from multiple glass rod passes, uneven pressure, poor sizing, or sensitizer issues. "
            "Prevention: single smooth coating stroke, even pressure, quality sizing, well-mixed "
            "sensitizer."
        ),
        "category": TermCategory.TROUBLESHOOTING,
        "related_terms": ["coating", "glass_rod", "sizing"],
        "examples": [
            "Practice coating technique on scrap paper",
            "Use single smooth stroke",
            "Check sizing quality",
        ],
        "see_also": ["coating", "glass_rod"],
    },
    "mottle": {
        "term": "Mottle",
        "definition": (
            "Blotchy or uneven appearance in print, typically in areas of uniform tone. Causes: "
            "uneven coating, inadequate development agitation, exhausted developer, hard water, "
            "or temperature variations. Prevention: even coating, continuous gentle agitation, "
            "fresh developer, distilled water."
        ),
        "category": TermCategory.TROUBLESHOOTING,
        "related_terms": ["development", "coating", "agitation"],
        "examples": [
            "Agitate continuously during development",
            "Use fresh developer",
            "Ensure even sensitizer coating",
        ],
        "see_also": ["development", "coating"],
    },
    # Additional process and artistic terms
    "sensitizer": {
        "term": "Sensitizer",
        "definition": (
            "The light-sensitive solution coated onto paper, consisting of ferric oxalate and "
            "platinum/palladium salts. When exposed to UV light, ferric oxalate reduces to ferrous, "
            "which then reduces the metal salts during development. Mixed immediately before use "
            "and applied in subdued lighting."
        ),
        "category": TermCategory.CHEMISTRY,
        "related_terms": ["ferric_oxalate", "coating", "light_sensitive"],
        "examples": [
            "Mix sensitizer immediately before coating",
            "Typically 2-4ml for 8x10 print",
            "Keep away from all light sources until dry",
        ],
        "see_also": ["ferric_oxalate", "coating"],
    },
    "light_sensitive": {
        "term": "Light Sensitive",
        "definition": (
            "Property of Pt/Pd sensitizer reacting to light exposure. Ferric oxalate is reduced by "
            "UV light to ferrous form. Coated paper must be kept in darkness until exposure. Even "
            "brief light exposure can cause fogging. Work under amber/red safelight or in complete "
            "darkness."
        ),
        "category": TermCategory.CHEMISTRY,
        "related_terms": ["sensitizer", "fogging", "uv_light"],
        "examples": [
            "Store coated paper in complete darkness",
            "UV light causes photochemical reaction",
            "Visible light can cause fogging",
        ],
        "see_also": ["sensitizer", "fogging"],
    },
    "developer": {
        "term": "Developer",
        "definition": (
            "Chemical solution that reduces exposed metal salts to metallic form, making the image "
            "visible. Common developers: potassium oxalate (neutral tones), ammonium citrate "
            "(warmer tones), sodium citrate. Typically 20-25% solution at 68-75°F. Development "
            "time: 1-3 minutes with continuous agitation."
        ),
        "category": TermCategory.CHEMISTRY,
        "related_terms": ["potassium_oxalate", "ammonium_citrate", "development"],
        "examples": [
            "Potassium oxalate is most common",
            "Temperature affects tone",
            "Use fresh developer for important prints",
        ],
        "see_also": ["development", "potassium_oxalate"],
    },
    "humidity": {
        "term": "Humidity Control",
        "definition": (
            "Relative humidity significantly affects Pt/Pd printing. Optimal range: 50-65%. High "
            "humidity (>65%) requires longer exposure and can cause coating problems. Low humidity "
            "(<45%) makes paper brittle and affects chemistry performance. Monitor with hygrometer "
            "and adjust exposure times accordingly."
        ),
        "category": TermCategory.PROCESS,
        "related_terms": ["exposure", "coating"],
        "examples": [
            "Use dehumidifier if consistently high humidity",
            "Adjust exposure time based on humidity",
            "Track humidity in process log",
        ],
        "see_also": ["exposure", "coating"],
    },
    "curve": {
        "term": "Calibration Curve",
        "definition": (
            "Correction curve applied to digital images to compensate for non-linear response of "
            "Pt/Pd process. Created by measuring step tablet print and calculating inverse of actual "
            "response. Applied in Photoshop or other software before printing negative. Different "
            "curve needed for each paper/chemistry combination."
        ),
        "category": TermCategory.MEASUREMENT,
        "related_terms": ["linearization", "calibration", "step_tablet"],
        "examples": [
            "Generated from step tablet measurements",
            "Save as .acv file for Photoshop",
            "Apply before all creative adjustments",
        ],
        "synonyms": ["linearization curve"],
        "see_also": ["linearization", "calibration"],
    },
    "agitation": {
        "term": "Agitation",
        "definition": (
            "Gentle movement of chemistry during development and clearing to ensure even processing. "
            "Continuous agitation prevents streaking and mottle. Use gentle rocking or tilting "
            "motion - avoid vigorous agitation which can cause staining or uneven density."
        ),
        "category": TermCategory.PROCESS,
        "related_terms": ["development", "clearing"],
        "examples": [
            "Rock tray gently throughout development",
            "Continuous motion prevents mottle",
            "Avoid splashing or vigorous movement",
        ],
        "see_also": ["development", "clearing", "mottle"],
    },
    "puddle_pusher": {
        "term": "Puddle Pusher",
        "definition": (
            "Alternative to glass rod for coating - typically an acrylic or glass bar with flat "
            "edge. Pushes 'puddle' of sensitizer across paper. Some printers prefer over cylindrical "
            "rod for control of coating thickness. Requires similar technique to glass rod."
        ),
        "category": TermCategory.EQUIPMENT,
        "related_terms": ["glass_rod", "coating"],
        "examples": [
            "Provides good control over coating thickness",
            "Similar technique to glass rod",
            "Popular alternative coating tool",
        ],
        "see_also": ["glass_rod", "coating"],
    },
    # =====================================================
    # CYANOTYPE PROCESS TERMS
    # =====================================================
    "cyanotype": {
        "term": "Cyanotype",
        "definition": (
            "Iron-based alternative photographic printing process producing characteristic Prussian blue "
            "images. Invented by Sir John Herschel in 1842. Uses ferric ammonium citrate and potassium "
            "ferricyanide as sensitizers. Developed in water. One of the simplest and most accessible "
            "alternative processes. Often called 'blueprint' due to historical use in architectural drawings."
        ),
        "category": TermCategory.PROCESS,
        "related_terms": ["ferric_ammonium_citrate", "potassium_ferricyanide", "prussian_blue"],
        "examples": [
            "Anna Atkins created first photographically illustrated book using cyanotypes (1843)",
            "Exposure typically 10-30 minutes in direct sunlight",
            "Development in running water for 5-10 minutes",
            "Final image is Prussian blue (ferric ferrocyanide)",
        ],
        "synonyms": ["blueprint", "blue print", "sun print"],
        "see_also": ["ferric_ammonium_citrate", "new_cyanotype"],
    },
    "ferric_ammonium_citrate": {
        "term": "Ferric Ammonium Citrate",
        "definition": (
            "Iron(III) ammonium citrate, also known as FAC. Primary sensitizer for cyanotype printing. "
            "Available in 'green' and 'brown' forms - green is more common and light-sensitive. "
            "Mixed as 25% solution (Solution A) in classic cyanotype formula. Light-sensitive iron "
            "compound that reduces to ferrous form when exposed to UV light."
        ),
        "category": TermCategory.CHEMISTRY,
        "related_terms": ["cyanotype", "potassium_ferricyanide", "sensitizer"],
        "examples": [
            "Use 'green' form for best results",
            "Mix 25g per 100ml distilled water for standard solution",
            "Store in amber bottle, keeps 6+ months",
            "Green color indicates proper iron oxidation state",
        ],
        "synonyms": ["FAC", "ammonium iron citrate", "green FAC"],
        "see_also": ["cyanotype", "potassium_ferricyanide"],
    },
    "potassium_ferricyanide": {
        "term": "Potassium Ferricyanide",
        "definition": (
            "K3[Fe(CN)6], the second component of cyanotype sensitizer. Forms Prussian blue when "
            "combined with ferrous (reduced) iron. Mixed as 10% solution (Solution B) in classic "
            "cyanotype. Despite 'cyanide' in name, is relatively safe when not heated or mixed with "
            "strong acids. Produces characteristic blue color of cyanotypes."
        ),
        "category": TermCategory.CHEMISTRY,
        "related_terms": ["cyanotype", "ferric_ammonium_citrate", "prussian_blue"],
        "examples": [
            "Mix 10g per 100ml distilled water",
            "Orange-red crystals in solid form",
            "Safe when used properly - avoid acids and heat",
            "Also used in print bleaching and toning",
        ],
        "synonyms": ["red prussiate of potash", "potassium hexacyanoferrate(III)"],
        "see_also": ["cyanotype", "ferric_ammonium_citrate"],
    },
    "prussian_blue": {
        "term": "Prussian Blue",
        "definition": (
            "Ferric ferrocyanide (Fe4[Fe(CN)6]3), the deep blue pigment that forms the final image "
            "in cyanotype prints. One of the first synthetic pigments. Extremely stable and lightfast "
            "when fully oxidized. Forms when ferrous iron (from light exposure) reacts with ferricyanide "
            "during development/washing."
        ),
        "category": TermCategory.CHEMISTRY,
        "related_terms": ["cyanotype", "potassium_ferricyanide"],
        "examples": [
            "Final cyanotype image is Prussian blue pigment",
            "Can be toned to other colors with various baths",
            "Bleaches in alkaline conditions (washing soda)",
            "Deep blue color intensifies over 24-48 hours",
        ],
        "synonyms": ["iron blue", "Berlin blue", "Milori blue"],
        "see_also": ["cyanotype", "potassium_ferricyanide"],
    },
    "new_cyanotype": {
        "term": "New Cyanotype",
        "definition": (
            "Improved cyanotype formula developed by Mike Ware using ammonium iron(III) oxalate "
            "instead of ferric ammonium citrate. Produces higher Dmax, faster exposures, and better "
            "tonal separation than classic formula. More expensive but preferred by serious practitioners "
            "for fine art work."
        ),
        "category": TermCategory.PROCESS,
        "related_terms": ["cyanotype", "ammonium_iron_oxalate"],
        "examples": [
            "30-50% shorter exposure times than classic",
            "Higher maximum density achievable",
            "Better highlight detail",
            "Requires ammonium iron(III) oxalate",
        ],
        "synonyms": ["Ware cyanotype", "Mike Ware cyanotype"],
        "see_also": ["cyanotype"],
    },
    "cyanotype_toning": {
        "term": "Cyanotype Toning",
        "definition": (
            "Post-processing technique to change cyanotype color from blue to other hues. Common "
            "toners: tannic acid/tea (brown/black), wine (purple), ammonia (yellow temporarily), "
            "sodium carbonate bleach followed by redevelopment. Allows artistic variation while "
            "maintaining archival quality."
        ),
        "category": TermCategory.ARTISTIC,
        "related_terms": ["cyanotype", "prussian_blue"],
        "examples": [
            "Tea/tannic acid toning produces warm brown-black",
            "Bleach with sodium carbonate, then tone",
            "Some tones fade if not properly fixed",
            "Experiment on test prints first",
        ],
        "see_also": ["cyanotype"],
    },
    # =====================================================
    # SILVER GELATIN PROCESS TERMS
    # =====================================================
    "silver_gelatin": {
        "term": "Silver Gelatin",
        "definition": (
            "Traditional photographic printing process using factory-prepared papers coated with "
            "light-sensitive silver halides in gelatin. The standard black-and-white darkroom process. "
            "Paper is exposed under an enlarger, developed in chemical developer, stopped, fixed, "
            "and washed. Available in fiber-based (FB) and resin-coated (RC) varieties."
        ),
        "category": TermCategory.PROCESS,
        "related_terms": ["fiber_based", "resin_coated", "enlarger", "darkroom"],
        "examples": [
            "Standard process for darkroom printing since 1870s",
            "Paper is pre-sensitized - no coating required",
            "Processing at 68°F (20°C) is standard",
            "Available in graded or variable contrast",
        ],
        "synonyms": ["gelatin silver", "silver print", "darkroom print"],
        "see_also": ["fiber_based", "resin_coated", "variable_contrast"],
    },
    "fiber_based": {
        "term": "Fiber Based Paper",
        "definition": (
            "Traditional silver gelatin paper with emulsion coated on baryta (barium sulfate) layer "
            "over cotton/alpha-cellulose fiber base. Considered archival standard. Requires longer "
            "processing times: development 2-3 min, fix 5-10 min, wash 1 hour. Beautiful tonal range "
            "and surface quality. Can be toned for permanence and color shifts."
        ),
        "category": TermCategory.MATERIALS,
        "related_terms": ["silver_gelatin", "resin_coated", "baryta"],
        "examples": [
            "Museum-quality archival prints use FB paper",
            "Longer wash time (60+ minutes) required",
            "Can be air-dried, ferrotyped, or heat dried",
            "Responds well to selenium and gold toning",
        ],
        "synonyms": ["FB paper", "baryta paper"],
        "see_also": ["silver_gelatin", "resin_coated"],
    },
    "resin_coated": {
        "term": "Resin Coated Paper",
        "definition": (
            "Silver gelatin paper with emulsion on plastic-coated paper base. Faster processing "
            "than fiber-based: development 1-2 min, fix 2-3 min, wash 4-5 min. Easier to handle "
            "and dries flat. Not considered archival but excellent for proof prints and general work. "
            "More resistant to curling and damage during processing."
        ),
        "category": TermCategory.MATERIALS,
        "related_terms": ["silver_gelatin", "fiber_based"],
        "examples": [
            "Quick processing for proof prints",
            "4-minute wash is sufficient",
            "Air dries flat in 30 minutes",
            "Not recommended for long-term archival storage",
        ],
        "synonyms": ["RC paper", "PE paper"],
        "see_also": ["silver_gelatin", "fiber_based"],
    },
    "variable_contrast": {
        "term": "Variable Contrast Paper",
        "definition": (
            "Silver gelatin paper containing two emulsion layers sensitive to different colors. "
            "Contrast is controlled by filtering the enlarger light: magenta filters increase contrast, "
            "yellow filters decrease it. Standard grades 00-5 available. Allows local contrast control "
            "through split-grade printing technique."
        ),
        "category": TermCategory.MATERIALS,
        "related_terms": ["silver_gelatin", "split_grade", "multigrade_filter"],
        "examples": [
            "Ilford Multigrade is most popular brand",
            "Grade 2 is normal contrast (no filter)",
            "Magenta = high contrast, Yellow = low contrast",
            "Split-grade printing uses both for local control",
        ],
        "synonyms": ["multigrade", "VC paper", "multicontrast"],
        "see_also": ["silver_gelatin", "split_grade"],
    },
    "split_grade": {
        "term": "Split Grade Printing",
        "definition": (
            "Darkroom technique using separate exposures through high and low contrast filters on "
            "variable contrast paper. Shadows exposed through high contrast filter (grade 4-5), "
            "highlights through low contrast filter (grade 0-1). Provides superior tonal control "
            "and allows local contrast manipulation."
        ),
        "category": TermCategory.PROCESS,
        "related_terms": ["variable_contrast", "dodging_burning"],
        "examples": [
            "First exposure: high contrast for shadows",
            "Second exposure: low contrast for highlights",
            "Adjust ratio to control overall contrast",
            "Can dodge/burn each exposure separately",
        ],
        "synonyms": ["split-filter printing", "split filtering"],
        "see_also": ["variable_contrast", "dodging_burning"],
    },
    "enlarger": {
        "term": "Enlarger",
        "definition": (
            "Optical device for projecting negative image onto photographic paper. Light source "
            "illuminates negative, lens focuses and magnifies image onto paper below. Types: "
            "condenser (high contrast, sharp), diffusion (lower contrast, forgiving), cold light "
            "(cool light, traditional favorite). Height adjustment controls print size."
        ),
        "category": TermCategory.EQUIPMENT,
        "related_terms": ["silver_gelatin", "negative", "darkroom"],
        "examples": [
            "Condenser enlargers produce higher contrast",
            "Diffusion enlargers hide dust and scratches",
            "Raise head for larger prints (longer exposure)",
            "Critical to maintain alignment and focus",
        ],
        "see_also": ["silver_gelatin", "darkroom"],
    },
    "darkroom": {
        "term": "Darkroom",
        "definition": (
            "Light-tight workspace for processing light-sensitive photographic materials. For silver "
            "gelatin printing: amber or red safelight illumination, wet and dry areas separated, "
            "temperature-controlled water supply, ventilation. Essential for handling unexposed paper "
            "and processing prints."
        ),
        "category": TermCategory.EQUIPMENT,
        "related_terms": ["silver_gelatin", "enlarger", "safelight"],
        "examples": [
            "Amber safelight (OC filter) safe for most papers",
            "Wet side for trays, dry side for enlarger",
            "Running water at 68°F (20°C) ideal",
            "Good ventilation essential for chemistry fumes",
        ],
        "see_also": ["silver_gelatin", "safelight"],
    },
    "safelight": {
        "term": "Safelight",
        "definition": (
            "Filtered light source providing illumination in darkroom without fogging light-sensitive "
            "materials. Silver gelatin papers typically require amber or red filters. Different filters "
            "for different materials (ortho vs pan). Must be appropriate wattage and distance from "
            "work surface."
        ),
        "category": TermCategory.EQUIPMENT,
        "related_terms": ["darkroom", "silver_gelatin"],
        "examples": [
            "Kodak OC filter (amber) for most B&W papers",
            "15-watt bulb, minimum 4 feet from paper",
            "Test periodically for safety",
            "Red safer but harder to work under",
        ],
        "see_also": ["darkroom", "silver_gelatin"],
    },
    "paper_developer": {
        "term": "Paper Developer",
        "definition": (
            "Chemical solution that reduces exposed silver halides to metallic silver, making the "
            "latent image visible. Standard types: Dektol/D-72 (neutral), Selectol (warm tone), "
            "Amidol (cold tone). Typically used at 68°F (20°C) for 1-3 minutes. Developer choice "
            "affects image tone and contrast."
        ),
        "category": TermCategory.CHEMISTRY,
        "related_terms": ["silver_gelatin", "dektol", "development"],
        "examples": [
            "Dektol 1:2 dilution is standard",
            "Warmer developer = warmer print tones",
            "Exhausted developer produces flat prints",
            "Development time affects density and contrast",
        ],
        "synonyms": ["print developer"],
        "see_also": ["silver_gelatin", "dektol"],
    },
    "dektol": {
        "term": "Dektol",
        "definition": (
            "Kodak's standard paper developer, similar to formula D-72. Produces neutral to slightly "
            "cool black tones. Typically diluted 1:2 with water. Development time 1.5-2 minutes at "
            "68°F. Industry standard developer for silver gelatin printing. Versatile and consistent "
            "results."
        ),
        "category": TermCategory.CHEMISTRY,
        "related_terms": ["paper_developer", "silver_gelatin"],
        "examples": [
            "Mix 1 part stock to 2 parts water",
            "90 seconds development at 68°F",
            "Produces neutral black tones",
            "Can use longer for more density",
        ],
        "synonyms": ["D-72", "Kodak Dektol"],
        "see_also": ["paper_developer", "silver_gelatin"],
    },
    "stop_bath": {
        "term": "Stop Bath",
        "definition": (
            "Acidic solution (typically dilute acetic acid) used between developer and fixer to "
            "immediately halt development. Prevents developer carryover into fixer and extends "
            "fixer life. 30 seconds immersion typical. Indicator stop baths change color when "
            "exhausted."
        ),
        "category": TermCategory.CHEMISTRY,
        "related_terms": ["silver_gelatin", "fixer"],
        "examples": [
            "1% acetic acid or commercial indicator bath",
            "30 seconds with agitation",
            "Indicator turns purple when exhausted",
            "Water rinse can substitute in pinch",
        ],
        "see_also": ["silver_gelatin", "fixer"],
    },
    "fixer": {
        "term": "Fixer",
        "definition": (
            "Chemical solution that removes unexposed silver halides from print, making it light-stable. "
            "Sodium thiosulfate (traditional) or ammonium thiosulfate (rapid). FB paper: 5-10 minutes, "
            "RC paper: 2-3 minutes. Over-fixing can bleach highlights. Two-bath fixing recommended "
            "for archival work."
        ),
        "category": TermCategory.CHEMISTRY,
        "related_terms": ["silver_gelatin", "archival", "hypo_clear"],
        "examples": [
            "Ammonium thiosulfate (rapid fix) is faster",
            "Two-bath fixing improves archival quality",
            "Over-fixing causes image bleaching",
            "Must be completely washed out",
        ],
        "synonyms": ["hypo", "sodium thiosulfate"],
        "see_also": ["silver_gelatin", "hypo_clear"],
    },
    "hypo_clear": {
        "term": "Hypo Clear",
        "definition": (
            "Wash aid solution used after fixing to reduce wash time for fiber-based papers. "
            "Converts residual fixer to more easily washed compounds. Reduces FB wash from 60+ "
            "minutes to 20-30 minutes. Essential for archival processing. Also called 'wash aid' "
            "or 'fixer remover'."
        ),
        "category": TermCategory.CHEMISTRY,
        "related_terms": ["fixer", "fiber_based", "archival"],
        "examples": [
            "Use after fixing, before final wash",
            "2-3 minutes immersion",
            "Dramatically reduces wash time for FB",
            "Not necessary for RC papers",
        ],
        "synonyms": ["wash aid", "fixer remover", "hypo eliminator"],
        "see_also": ["fixer", "fiber_based"],
    },
    "dodging_burning": {
        "term": "Dodging and Burning",
        "definition": (
            "Darkroom techniques for local exposure control. Dodging: blocking light from areas "
            "to make them lighter (hand, card on wire). Burning: giving additional exposure to "
            "areas to make them darker (card with hole). Essential for controlling tonal balance "
            "and directing viewer attention."
        ),
        "category": TermCategory.ARTISTIC,
        "related_terms": ["silver_gelatin", "exposure"],
        "examples": [
            "Dodge shadows to retain detail",
            "Burn highlights to add density",
            "Keep tool moving to avoid hard edges",
            "Plan adjustments before starting print",
        ],
        "synonyms": ["dodging", "burning", "burning in"],
        "see_also": ["silver_gelatin", "split_grade"],
    },
    "test_strip": {
        "term": "Test Strip",
        "definition": (
            "Series of progressive exposures on single piece of paper to determine correct "
            "exposure time. Paper is exposed in strips with increasing time, then processed. "
            "Provides visual guide for choosing base exposure. Essential step before making "
            "final print."
        ),
        "category": TermCategory.PROCESS,
        "related_terms": ["silver_gelatin", "exposure"],
        "examples": [
            "Start with 3-second increments",
            "Process test strip normally",
            "Choose exposure with good highlight detail",
            "Make additional tests if needed",
        ],
        "see_also": ["silver_gelatin", "exposure"],
    },
    "toning_silver": {
        "term": "Silver Print Toning",
        "definition": (
            "Post-processing treatment that changes silver gelatin print color and/or improves "
            "permanence. Common toners: selenium (neutral to purple, archival), gold (blue-black, "
            "highly archival), sepia (warm brown), selenium/gold split (multiple tones). Toning "
            "converts metallic silver to more stable compounds."
        ),
        "category": TermCategory.PROCESS,
        "related_terms": ["silver_gelatin", "archival"],
        "examples": [
            "Selenium 1:20 for subtle permanence boost",
            "Gold toner for blue-black and permanence",
            "Sepia requires bleach step first",
            "Tone after complete washing",
        ],
        "see_also": ["silver_gelatin", "archival"],
    },
}


class Glossary:
    """Manages glossary of Pt/Pd printing terms."""

    def __init__(self):
        """Initialize glossary."""
        self.terms: dict[str, GlossaryTerm] = {}
        self._load_terms()

    def _load_terms(self) -> None:
        """Load glossary data into GlossaryTerm objects."""
        for key, data in GLOSSARY_DATA.items():
            term = GlossaryTerm(**data)
            # Index by lowercase version of term for case-insensitive lookup
            self.terms[term.term.lower()] = term
            # Also index synonyms
            for synonym in term.synonyms:
                self.terms[synonym.lower()] = term

    def lookup(self, term: str) -> Optional[GlossaryTerm]:
        """
        Look up a specific term.

        Args:
            term: Term to look up (case-insensitive)

        Returns:
            GlossaryTerm object or None if not found
        """
        return self.terms.get(term.lower())

    def search(self, query: str) -> list[GlossaryTerm]:
        """
        Search for terms matching query.

        Searches in term names, definitions, and synonyms.

        Args:
            query: Search string (case-insensitive)

        Returns:
            List of matching GlossaryTerm objects
        """
        query_lower = query.lower()
        results = []
        seen_terms = set()

        for term in self.terms.values():
            # Avoid duplicates from synonym indexing
            if term.term in seen_terms:
                continue

            # Search in term, definition, and synonyms
            if (
                query_lower in term.term.lower()
                or query_lower in term.definition.lower()
                or any(query_lower in syn.lower() for syn in term.synonyms)
            ):
                results.append(term)
                seen_terms.add(term.term)

        return sorted(results, key=lambda t: t.term)

    def get_by_category(self, category: TermCategory) -> list[GlossaryTerm]:
        """
        Get all terms in a specific category.

        Args:
            category: Term category

        Returns:
            List of GlossaryTerm objects in category
        """
        results = []
        seen_terms = set()

        for term in self.terms.values():
            if term.category == category and term.term not in seen_terms:
                results.append(term)
                seen_terms.add(term.term)

        return sorted(results, key=lambda t: t.term)

    def get_related(self, term: str) -> list[GlossaryTerm]:
        """
        Get terms related to a specific term.

        Args:
            term: Term to find related terms for

        Returns:
            List of related GlossaryTerm objects
        """
        main_term = self.lookup(term)
        if not main_term:
            return []

        related = []
        for related_term_name in main_term.related_terms + main_term.see_also:
            related_term = self.lookup(related_term_name.replace("_", " "))
            if related_term and related_term.term != main_term.term:
                related.append(related_term)

        return related

    def add_term(self, term: GlossaryTerm) -> None:
        """
        Add new term to glossary.

        Args:
            term: GlossaryTerm object to add
        """
        self.terms[term.term.lower()] = term
        for synonym in term.synonyms:
            self.terms[synonym.lower()] = term

    def get_all_categories(self) -> list[TermCategory]:
        """
        Get list of all categories in use.

        Returns:
            List of TermCategory values
        """
        categories = set(term.category for term in self.terms.values())
        return sorted(categories, key=lambda c: c.value)

    def export_to_dict(self) -> dict[str, dict]:
        """
        Export glossary to dictionary format.

        Returns:
            Dictionary of all terms
        """
        result = {}
        seen_terms = set()

        for term in self.terms.values():
            if term.term not in seen_terms:
                result[term.term] = term.model_dump()
                seen_terms.add(term.term)

        return result
