# 🧠 Intelligent Report Generation with Granular LLM-Based Editing: A Targeted Modification System

**Author:** Rishi Renchen  
**Framework:** LangGraph (Python) / Azure OpenAI  

---

## 📘 Overview

This project explores a next-generation approach to document editing using Large Language Models (LLMs).  
Traditional LLM-based document generation systems require **complete regeneration** of a report whenever users request an edit — even for small, localized modifications.  

This leads to:
- 🚀 **High latency and token cost** (entire document reprocessed)
- ⚠️ **Unpredictable outputs** (unintended changes to other sections)
- 😕 **Poor user experience** (loss of structure and context)

**Goal:**  
To build an **intelligent, targeted modification system** that applies *surgical edits* to specific document sections while preserving all other content.

---

## 🎯 Core Problem Statement

> “Traditional LLM-based document generation requires complete regeneration for any modification, which is:”

- **Inefficient** (high token usage, latency)
- **Unpredictable** (may alter unintended sections)
- **Poor UX** (users lose context during full rewrites)

**Our Solution:** Build a system that applies surgical edits to specific document sections while preserving the rest.

---

## 🔧 Technical Approach

The system implements four core capabilities:

### 1. Document Structure Understanding
- **Parse reports** into logical sections (headings, paragraphs, lists)
- **Maintain a section tree/map** with unique identifiers
- **Track section boundaries** and hierarchical relationships
- Support for markdown, HTML, and structured document formats

### 2. Intent Recognition from Prompts
- **Parse user commands** like:
  - "remove section X"
  - "add between A and B"
  - "update section Y with new requirements"
- **Identify target sections** and modification types:
  - `ADD` - Insert new content
  - `REMOVE` - Delete sections
  - `UPDATE` - Modify existing content
  - `EXPAND` - Enhance existing sections
  - `REORDER` - Rearrange sections
- **Extract context requirements** (what content to add/modify)

### 3. Targeted LLM Application
- **Send only relevant context** to LLM (surrounding sections for coherence)
- **Generate only the modified portions** (not the entire document)
- **Implement different strategies** for different edit types:
  - **Removal**: Direct deletion with transition smoothing
  - **Addition**: Generate new section with context awareness
  - **Update**: Regenerate specific section with new requirements
  - **Expansion**: Enhance existing content while preserving structure

### 4. Document Reassembly
- **Merge modified sections** back into document
- **Ensure coherence** at section boundaries
- **Track changes** for user review (similar to Git diff view)
- **Preserve formatting** and document structure

---

## ⚙️ System Architecture

The architecture is implemented as a **LangGraph pipeline** (with Azure OpenAI backend), where each node represents a distinct functional stage.

### 🧩 Node Flow

```plaintext
User Command
   ↓
Intent Recognition
   ↓
Document Structure Analyzer
   ↓
Context Extractor
   ↓
Targeted Edit Executor (LLM)
   ↓
Document Reassembler
   ↓
Coherence Smoothing + Diff Viewer
   ↓
Final Output
```

---

## 📊 Success Metrics

The system is evaluated against the following key performance indicators:

### Token Reduction
- **Compare vs. full regeneration**: Measure token usage for targeted edits vs. complete document regeneration
- **Target**: Achieve 60-90% token reduction for localized edits

### Edit Accuracy
- **Does it modify the right sections?**: Verify that only intended sections are changed
- **Precision**: Measure false positive modifications (unintended changes)
- **Recall**: Ensure all requested changes are applied

### Coherence Preservation
- **Quality of unchanged sections**: Verify that unmodified sections remain intact
- **Boundary coherence**: Ensure smooth transitions between modified and unmodified sections
- **Formatting consistency**: Maintain document structure and styling

### Latency Improvement
- **Time saved vs. full regeneration**: Measure end-to-end processing time
- **Target**: Achieve 50-80% latency reduction for targeted edits

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Azure OpenAI account with API credentials
- Required packages (see `requirements.txt`)

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
# Create a .env file with:
# AZURE_OPENAI_API_KEY=your-api-key
# AZURE_OPENAI_ENDPOINT=your-endpoint
# AZURE_OPENAI_API_VERSION=2024-02-15-preview
# AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=your-deployment-name
```

### Usage

The system can be used via:
- **Jupyter Notebook**: `granular_document_editor.ipynb` - Interactive exploration
- **Python Script**: `startup.py` - Command-line interface
- **API Server**: FastAPI backend (coming soon)
- **Web UI**: Streamlit interface (coming soon)

---

## 📁 Project Structure

```
Use_Case_2/
├── granular_document_editor.ipynb  # Main implementation notebook
├── startup.py                       # CLI entry point
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── .env                            # Environment variables (not in repo)
```

---

## 🔬 Implementation Details

### Core Components

- **DocumentParser**: Parses markdown/structured documents into hierarchical sections
- **IntentRecognizer**: Analyzes user commands and extracts edit intents
- **ContextExtractor**: Identifies relevant surrounding context for LLM calls
- **TargetedEditor**: Applies surgical edits using Azure OpenAI
- **DocumentReassembler**: Merges modified sections back into complete document
- **DiffTracker**: Tracks and visualizes changes (Git-style diff view)

### Data Structures

- `DocumentSection`: Represents a section with metadata (id, type, level, content)
- `EditIntent`: Parsed user command with action type and target information
- `EditMetrics`: Performance tracking (tokens, latency, accuracy)

---

## 🎯 Future Enhancements

- [ ] Support for additional document formats (PDF, DOCX)
- [ ] Multi-section batch editing
- [ ] Version history and rollback capabilities
- [ ] Collaborative editing with conflict resolution
- [ ] Advanced diff visualization UI
- [ ] Integration with document management systems

---

## 📝 Notes

This project is part of a broader exploration into agentic AI systems and represents a parallel development track based on customer interest in efficient document editing workflows.
