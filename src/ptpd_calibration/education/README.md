# Educational Components for Pt/Pd Printing Tool

This directory contains comprehensive educational resources for platinum/palladium printing, including tutorials, glossary, and tips systems.

## Overview

The education module provides three main components:

1. **Tutorial System** - Interactive, step-by-step tutorials for all skill levels
2. **Glossary** - Comprehensive terminology reference with cross-references
3. **Tips System** - Contextual best practices and helpful advice

## Components

### 1. Tutorial System (`tutorials.py`)

Interactive tutorials guide users through Pt/Pd printing processes from beginner to advanced levels.

#### Features
- 7 comprehensive tutorials covering all aspects of Pt/Pd printing
- Difficulty levels: Beginner, Intermediate, Advanced
- User progress tracking
- Step validation
- Estimated time and prerequisites
- Materials lists and learning objectives

#### Available Tutorials

1. **Your First Platinum/Palladium Print** (Beginner, 180 min)
   - Complete end-to-end guide for first-time printers
   - Covers safety, sizing, chemistry, coating, exposure, development, and clearing

2. **Calibration and Linearization Workflow** (Intermediate, 120 min)
   - Creating and using calibration curves
   - Step tablet printing and measurement
   - Achieving consistent results

3. **Chemistry Mixing and Ratios** (Intermediate, 60 min)
   - Understanding Pt/Pd ratios and their effects
   - Mixing accurate solutions
   - Using contrast agents

4. **Paper Coating Techniques** (Intermediate, 45 min)
   - Glass rod coating mastery
   - Troubleshooting coating issues
   - Alternative coating methods

5. **UV Exposure Techniques and Control** (Intermediate, 60 min)
   - Working with different UV sources
   - Exposure testing methods
   - Environmental compensation

6. **Troubleshooting Common Problems** (Intermediate, 90 min)
   - Diagnosing print defects
   - Fixing common issues
   - Preventive measures

7. **Advanced Printing Techniques** (Advanced, 120 min)
   - Split-grade printing
   - Contrast masking
   - Creative chemistry variations

#### Usage Example

```python
from ptpd_calibration.education import TutorialManager, UserProgress

# Initialize manager
tm = TutorialManager()

# List available tutorials
tutorials = tm.get_available_tutorials()
for tutorial in tutorials:
    print(f"{tutorial.display_name} - {tutorial.difficulty.value}")

# Get specific tutorial
first_print = tm.get_tutorial("first_print")
print(f"Steps: {len(first_print.steps)}")
print(f"Time: {first_print.estimated_time} minutes")

# Track user progress
progress = tm.start_tutorial("first_print")
print(f"Current step: {progress.current_step}")

# Get progress info
progress_info = tm.get_progress("first_print", progress)
print(f"Completion: {progress_info['progress_percent']:.0f}%")
```

### 2. Glossary System (`glossary.py`)

Comprehensive terminology reference with 43+ terms covering all aspects of Pt/Pd printing.

#### Features
- Detailed definitions for all key terms
- Category organization (Chemistry, Process, Measurement, etc.)
- Cross-references and related terms
- Synonyms and alternative names
- Usage examples
- Case-insensitive search

#### Term Categories

- **Chemistry** - Sensitizers, developers, clearing agents, metal salts
- **Process** - Coating, exposure, development, clearing
- **Measurement** - Dmax, Dmin, density range, linearization
- **Materials** - Paper, negatives, substrates
- **Equipment** - Glass rods, printing frames, UV sources
- **Quality** - Contrast, tone, archival stability
- **Troubleshooting** - Common problems and solutions

#### Usage Example

```python
from ptpd_calibration.education import Glossary, TermCategory

# Initialize glossary
glossary = Glossary()

# Look up specific term
term = glossary.lookup("ferric oxalate")
print(f"{term.term}: {term.definition}")
print(f"Related: {term.related_terms}")

# Search for terms
results = glossary.search("exposure")
for term in results:
    print(f"{term.term} ({term.category.value})")

# Browse by category
chemistry_terms = glossary.get_by_category(TermCategory.CHEMISTRY)
for term in chemistry_terms:
    print(f"• {term.term}")

# Get related terms
related = glossary.get_related("platinum chloride")
for term in related:
    print(f"Related: {term.term}")
```

### 3. Tips System (`tips.py`)

Contextual tips and best practices organized by category and difficulty.

#### Features
- 58+ practical tips covering all aspects of Pt/Pd printing
- Contextual recommendations based on current operation
- Priority levels (1-5) for importance
- Category-based organization
- Difficulty filtering (Beginner, Intermediate, Advanced, All)
- Seen/unseen tracking
- Random tip generation

#### Tip Categories

- **Safety** - Critical safety information
- **Chemistry** - Mixing and handling chemistry
- **Coating** - Application techniques
- **Exposure** - UV exposure control
- **Development** - Development best practices
- **Clearing** - Proper clearing techniques
- **Troubleshooting** - Problem solving
- **Workflow** - Process optimization
- **Cost Saving** - Economical practices
- **Quality** - Achieving best results
- **Beginner** - New printer guidance
- **Advanced** - Advanced techniques

#### Usage Example

```python
from ptpd_calibration.education import TipsManager, TipCategory, TipDifficulty

# Initialize manager
tips_mgr = TipsManager()

# Get contextual tips for current operation
coating_tips = tips_mgr.get_contextual_tips("coating", limit=5)
for tip in coating_tips:
    print(f"[{tip.category.value}] {tip.content}")

# Get tips by category
safety_tips = tips_mgr.get_tips_by_category(TipCategory.SAFETY)
for tip in safety_tips:
    print(f"Priority {tip.priority}: {tip.content}")

# Get random tip
random_tip = tips_mgr.get_random_tip(difficulty=TipDifficulty.BEGINNER)
print(random_tip.content)

# Get high-priority (critical) tips
critical = tips_mgr.get_high_priority_tips(min_priority=5)
for tip in critical:
    print(f"CRITICAL: {tip.content}")

# Search tips
results = tips_mgr.search_tips("glass rod")
for tip in results:
    print(tip.content)

# Get statistics
stats = tips_mgr.get_statistics()
print(f"Total tips: {stats['total_tips']}")
print(f"High priority: {stats['high_priority_count']}")
```

## Data Structure

All tutorial content, glossary terms, and tips are stored as structured data (Python dictionaries) within the module files, making them easy to maintain, update, and potentially export to JSON/YAML if needed.

### Tutorial Data Structure
```python
{
    "tutorial_name": {
        "display_name": "Tutorial Title",
        "description": "Overview...",
        "difficulty": TutorialDifficulty.BEGINNER,
        "estimated_time": 60,
        "prerequisites": ["other_tutorial"],
        "learning_objectives": [...],
        "materials_needed": [...],
        "steps": [
            {
                "step_number": 1,
                "title": "Step Title",
                "content": "Detailed instructions...",
                "action": ActionType.PRACTICE,
                "validation": "Expected outcome",
                "tips": [...],
                "warnings": [...],
            }
        ]
    }
}
```

### Glossary Data Structure
```python
{
    "term_key": {
        "term": "Display Term",
        "definition": "Detailed definition...",
        "category": TermCategory.CHEMISTRY,
        "related_terms": [...],
        "examples": [...],
        "synonyms": [...],
    }
}
```

### Tips Data Structure
```python
{
    "content": "The tip content...",
    "category": TipCategory.COATING,
    "difficulty": TipDifficulty.ALL,
    "priority": 4,
    "conditions": ["coating", "practice"],
    "related_terms": ["glass_rod", "sizing"],
}
```

## Integration Examples

### With UI/CLI

```python
# Display contextual help during coating
def show_coating_help():
    tips_mgr = TipsManager()
    tips = tips_mgr.get_contextual_tips("coating", limit=3)

    print("💡 Coating Tips:")
    for tip in tips:
        print(f"  • {tip.content}")

# Start tutorial from UI
def begin_tutorial(tutorial_name):
    tm = TutorialManager()
    tutorial = tm.get_tutorial(tutorial_name)

    print(f"Starting: {tutorial.display_name}")
    print(f"Materials needed:")
    for material in tutorial.materials_needed:
        print(f"  ☐ {material}")
```

### With LLM Assistant

```python
# Provide glossary context to LLM
def get_term_definition(term):
    glossary = Glossary()
    term_obj = glossary.lookup(term)

    if term_obj:
        return {
            "term": term_obj.term,
            "definition": term_obj.definition,
            "examples": term_obj.examples,
            "related": [glossary.lookup(t).term for t in term_obj.related_terms]
        }
```

## Testing

Run the demonstration script to see all components in action:

```bash
PYTHONPATH=/path/to/src python examples/education_demo.py
```

## Extending the System

### Adding New Tutorials

Edit `tutorials.py` and add to `TUTORIALS_DATA` dictionary:

```python
TUTORIALS_DATA["new_tutorial"] = {
    "display_name": "New Tutorial Title",
    "description": "Tutorial description...",
    "difficulty": TutorialDifficulty.INTERMEDIATE,
    "estimated_time": 60,
    "steps": [...],
    # ... other fields
}
```

### Adding New Glossary Terms

Edit `glossary.py` and add to `GLOSSARY_DATA` dictionary:

```python
GLOSSARY_DATA["new_term"] = {
    "term": "New Term",
    "definition": "Definition...",
    "category": TermCategory.CHEMISTRY,
    "related_terms": [...],
    # ... other fields
}
```

### Adding New Tips

Edit `tips.py` and add to `TIPS_DATA` list:

```python
TIPS_DATA.append({
    "content": "New tip content...",
    "category": TipCategory.COATING,
    "difficulty": TipDifficulty.ALL,
    "priority": 3,
    "conditions": ["coating"],
})
```

## License

Part of the Platinum-Palladium AI Printing Tool project.

## Contributing

Contributions to expand tutorials, glossary terms, and tips are welcome! Please ensure:
- Accurate technical information
- Clear, concise language
- Proper categorization
- Cross-references where appropriate
