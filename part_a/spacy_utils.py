# spacy_utils.py
import spacy
import pandas as pd
from spacy.matcher import Matcher

# Load Spacy model
nlp = spacy.load("en_core_web_sm")

def create_doc(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        text = file.read()
    return nlp(text)

def get_tokenization_and_pos(doc):
    df = pd.DataFrame(columns=['SENTENCE', 'WORD', 'POS', 'DEP', 'DEP_ON', 'POS_EXPLAIN', 'DEP_EXPLAIN'])

    for sentence in doc.sents:
        df.loc[len(df)] = {'SENTENCE': sentence.text}
        for ent in sentence:
            df.loc[len(df)] = {'WORD': ent.text,
                               'POS': ent.pos_,
                               'DEP': ent.dep_,
                               'DEP_ON': ent.head.text,
                               'POS_EXPLAIN': spacy.explain(ent.pos_),
                               'DEP_EXPLAIN': spacy.explain(ent.dep_)
                               }

    return df

def get_named_entities(doc):
    data = []
    for ent in doc.ents:
        data.append({
            "Entity": ent.text,
            "Label": ent.label_
        })
    return pd.DataFrame(data)

def explain_tags(doc):
    data = []
    for token in doc:
        data.append({
            "Token": token.text,
            "POS": token.pos_,
            "POS Explanation": spacy.explain(token.pos_),
            "Dependency": token.dep_,
            "Dependency Explanation": spacy.explain(token.dep_)
        })
    return pd.DataFrame(data)

def find_matches(doc, patterns):
    matcher = Matcher(nlp.vocab)
    for pattern_name, pattern in patterns.items():
        matcher.add(pattern_name, [pattern])

    data = []
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        data.append({
            "Match": span.text,
            "Start": start,
            "End": end
        })
    return pd.DataFrame(data)