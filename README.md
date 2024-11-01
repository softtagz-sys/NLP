# NLP Assignment with spaCy

This repository contains the completed assignment for Natural Language Processing (NLP) using the spaCy library. The assignment consists of three parts, each focusing on different aspects of text processing, analysis, and custom NLP model development.

## Assignment Overview

### Part A: Basic Text Processing
In this part, I completed the first tutorial of spaCy and performed the following tasks:

1. **Tutorial Completion:**
   - Followed the spaCy tutorial (Part 1) and executed the provided Python code in a Jupyter Notebook.

2. **Text Loading:**
   - Loaded a small extract from *Moby Dick* by Herman Melville into a spaCy document.
   - Loaded two additional text files, `ai_forecast1.txt` and `ai_forecast2.txt`, into separate spaCy documents.

3. **Tokenization and Analysis:**
   - Performed tokenization on all three documents.
   - Printed out the Part-of-Speech (POS) tags and grammatical structures of sentences.
   - Discovered and displayed named entities in both technical reports.
   - Explained the meaning of grammar and POS tags using spaCy's features.

4. **Matcher Construction:**
   - Created a matcher to identify occurrences of “Artificial Intelligence” in the Key Market Insights text.
   - Developed a matcher to find instances of the word “AI” followed by a verb.
   - Constructed a matcher to identify numbers followed by a percentage sign (e.g., `129%`).
   - Built a matcher to locate company names within the texts.

### Part B: Document Comparison and Custom Entity Recognition
In this part, I completed the spaCy tutorial (Chapter 2) and engaged in the following tasks:

1. **Document Comparison:**
   - Compared the three documents: *Moby Dick*, `ai_forecast1.txt`, and `ai_forecast2.txt`.
   - Compared the first 100 tokens of each document to analyze similarities and differences.

2. **Custom Entity Recognition:**
   - Started from a blank spaCy NLP model and added my name as an entity.
   - Utilized a span to extract and add the first and last name (e.g., Remco Evenepoel) as a `PERSON` entity in the custom model.

### Part C: Custom Pipeline Component
In this final part, I completed the spaCy tutorial (Chapter 3) and performed the following tasks:

1. **Pipeline Components:**
   - Printed the names of the components in the spaCy NLP pipeline to understand the architecture.

2. **Custom Component Development:**
   - Added a custom component to the pipeline that prints the longest token found in the document, showcasing the flexibility of spaCy for custom NLP tasks.
