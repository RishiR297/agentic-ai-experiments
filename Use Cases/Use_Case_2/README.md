# 🧠 Intelligent Report Generation with Granular LLM-Based Editing

**Author:** Rishi Renchen  
**Framework:** LangGraph (Python)  

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

> “How can we apply precise, section-level LLM edits to large reports — without regenerating the entire document?”

The system should:
- Understand the **structure** of a document (sections, subsections, paragraphs).  
- Interpret **user edit intents** (add, remove, update, reorder).  
- Apply **localized LLM edits** only to relevant sections.  
- Preserve overall **coherence and formatting**.  
- Support **diff viewing** and **version tracking** for transparency.

---

## ⚙️ System Architecture

The architecture is implemented as a **LangGraph pipeline**, where each node represents a distinct functional stage.

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
