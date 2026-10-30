"""
dataset.py
Parallel Text Dataset Preprocessing and Loading Pipeline
Course: Foundations of Generative AI (ICT4403) - Assignment 3
Language Pair: English (Source) -> French (Target)
"""

import os
import re
import unicodedata
import pandas as pd
from typing import Tuple, List, Dict, Optional
from collections import Counter

# Fallback clean bilingual sample pairs for immediate offline execution or sanity testing
OFFLINE_SAMPLE_PAIRS = [
    ("The quick brown fox jumps over the lazy dog.", "Le renard brun rapide saute par-dessus le chien paresseux."),
    ("Artificial intelligence is transforming modern science and technology.", "L'intelligence artificielle transforme la science et la technologie modernes."),
    ("She has been reading this interesting book for three hours.", "Elle lit ce livre intéressant depuis trois heures."),
    ("The weather is surprisingly pleasant in the mountains today.", "Le temps est étonnamment agréable dans les montagnes aujourd'hui."),
    ("Can you please help me translate these complex documents?", "Pouvez-vous s'il vous plaît m'aider à traduire ces documents complexes ?"),
    ("He works as a senior software engineer in Paris.", "Il travaille comme ingénieur logiciel senior à Paris."),
    ("Education is the most powerful weapon which you can use to change the world.", "L'éducation est l'arme la plus puissante que vous puissiez utiliser pour changer le monde."),
    ("We must protect our environment for future generations.", "Nous devons protéger notre environnement pour les générations futures."),
    ("They arrived at the railway station just before the train departed.", "Ils sont arrivés à la gare ferroviaire juste avant que le train ne parte."),
    ("Knowledge is of no value unless you put it into practice.", "Le savoir n'a aucune valeur si vous ne le mettez pas en pratique."),
    ("The doctor carefully examined the patient in the hospital.", "Le médecin a soigneusement examiné le patient à l'hôpital."),
    ("Music expresses that which cannot be put into words and that which cannot remain silent.", "La musique exprime ce qui ne peut être dit et ce qui ne peut rester silencieux."),
    ("A journey of a thousand miles begins with a single step.", "Un voyage de mille lieues commence par un seul pas."),
    ("The library offers access to millions of digital academic articles.", "La bibliothèque offre l'accès à des millions d'articles universitaires numériques."),
    ("She speaks three European languages with remarkable fluency.", "Elle parle trois langues européennes avec une fluidité remarquable."),
    ("Scientists discovered a new species of marine life in the deep ocean.", "Les scientifiques ont découvert une nouvelle espèce de vie marine dans les profondeurs de l'océan."),
    ("Patience is a key element of success in difficult endeavors.", "La patience est un élément clé de la réussite dans les entreprises difficiles."),
    ("The international conference will take place in Geneva next month.", "La conférence internationale aura lieu à Genève le mois prochain."),
    ("Digital privacy is a fundamental human right in the modern era.", "La confidentialité numérique est un droit fondamental à l'ère moderne."),
    ("He solved the challenging mathematical equation within minutes.", "Il a résolu l'équation mathématique difficile en quelques minutes."),
    ("The artist painted a magnificent landscape of the sunrise over the river.", "L'artiste a peint un magnifique paysage du lever de soleil sur la rivière."),
    ("Effective communication requires both speaking clearly and listening attentively.", "Une communication efficace exige à la fois de parler clairement et d'écouter attentivement."),
    ("Clean renewable energy sources are crucial to combat global warming.", "Les sources d'énergie renouvelable propre sont cruciales pour lutter contre le réchauffement climatique."),
    ("Students should review all lecture materials prior to the final exam.", "Les étudiants devraient réviser toutes les notes de cours avant l'examen final."),
    ("The museum showcases historic artifacts dating back to ancient Rome.", "Le musée présente des artefacts historiques remontant à la Rome antique."),
    ("Artificial neural networks learn complex representations through hierarchical layers.", "Les réseaux neuronaux artificiels apprennent des représentations complexes grâce à des couches hiérarchiques."),
    ("Language models require substantial computational resources for pretraining.", "Les modèles linguistiques nécessitent des ressources de calcul substantielles pour le pré-entraînement."),
    ("Good food brings people together around the dinner table.", "La bonne nourriture rassemble les gens autour de la table."),
    ("The government announced new policies to support small local businesses.", "Le gouvernement a annoncé de nouvelles politiques pour soutenir les petites entreprises locales."),
    ("Curiosity and perseverance are essential qualities for research researchers.", "La curiosité et la persévérance sont des qualités essentielles pour les chercheurs.")
]


def clean_text(text: str) -> str:
    """Normalize unicode, strip whitespace, and normalize punctuation."""
    if not isinstance(text, str):
        return ""
    # Unicode NFKC normalization
    text = unicodedata.normalize('NFKC', text)
    # Strip excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_and_preprocess_dataset(
    dataset_name: str = "opus_books",
    lang_pair: str = "en-fr",
    max_samples: int = 12000,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    min_len: int = 3,
    max_len: int = 60,
    save_dir: str = "data"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load parallel dataset, clean text, filter by sequence length,
    split into train/val/test, and persist locally.
    """
    os.makedirs(save_dir, exist_ok=True)
    src_lang, tgt_lang = lang_pair.split("-")
    
    pairs = []
    hf_loaded = False
    
    try:
        from datasets import load_dataset
        print(f"[*] Attempting to download '{dataset_name}' ({lang_pair}) via HuggingFace Datasets...")
        raw_data = load_dataset(dataset_name, lang_pair, split="train")
        print(f"[+] Loaded {len(raw_data)} raw pairs from HuggingFace.")
        
        for item in raw_data:
            trans = item.get("translation", {})
            src = clean_text(trans.get(src_lang, ""))
            tgt = clean_text(trans.get(tgt_lang, ""))
            if src and tgt:
                src_words = src.split()
                tgt_words = tgt.split()
                if min_len <= len(src_words) <= max_len and min_len <= len(tgt_words) <= max_len:
                    pairs.append((src, tgt))
            if len(pairs) >= max_samples:
                break
        hf_loaded = True
    except Exception as e:
        print(f"[!] Hugging Face dataset download failed or network unavailable: {e}")
        print("[*] Falling back to offline augmented parallel corpus.")
        pairs = OFFLINE_SAMPLE_PAIRS * (max_samples // len(OFFLINE_SAMPLE_PAIRS) + 1)
        pairs = pairs[:max_samples]

    df = pd.DataFrame(pairs, columns=["source_en", "target_fr"]).drop_duplicates().reset_index(drop=True)
    total_len = len(df)
    print(f"[+] Filtered dataset contains {total_len} high-quality bilingual sentence pairs.")
    
    # Shuffle deterministically
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    n_train = int(total_len * train_ratio)
    n_val = int(total_len * val_ratio)
    
    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_df = df.iloc[n_train + n_val:].reset_index(drop=True)
    
    print(f"    - Train split:      {len(train_df)} samples")
    print(f"    - Validation split: {len(val_df)} samples")
    print(f"    - Test split:       {len(test_df)} samples")
    
    # Persist splits for reproducible cross-model evaluation
    train_path = os.path.join(save_dir, "train.csv")
    val_path = os.path.join(save_dir, "val.csv")
    test_path = os.path.join(save_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False, encoding="utf-8")
    val_df.to_csv(val_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")
    print(f"[+] Dataset splits saved to '{save_dir}/'")
    
    return train_df, val_df, test_df


class SimpleVocabulary:
    """Word-level / subword vocabulary with special tokens for the Basic Transformer."""
    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"
    
    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.special_tokens = [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        self.token2id: Dict[str, int] = {tok: idx for idx, tok in enumerate(self.special_tokens)}
        self.id2token: Dict[int, str] = {idx: tok for tok, idx in self.token2id.items()}
        self.pad_id = self.token2id[self.PAD_TOKEN]
        self.sos_id = self.token2id[self.SOS_TOKEN]
        self.eos_id = self.token2id[self.EOS_TOKEN]
        self.unk_id = self.token2id[self.UNK_TOKEN]
        
    def tokenize(self, text: str) -> List[str]:
        # Simple punctuation-aware regex tokenizer
        text = text.lower()
        return re.findall(r"\w+|[^\w\s]", text)
        
    def build_vocab(self, texts: List[str]):
        counter = Counter()
        for t in texts:
            counter.update(self.tokenize(t))
        for token, freq in counter.items():
            if freq >= self.min_freq and token not in self.token2id:
                idx = len(self.token2id)
                self.token2id[token] = idx
                self.id2token[idx] = token
        print(f"[+] Vocabulary built with {len(self.token2id)} tokens.")
        
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = self.tokenize(text)
        ids = [self.token2id.get(tok, self.unk_id) for tok in tokens]
        if add_special_tokens:
            ids = [self.sos_id] + ids + [self.eos_id]
        return ids
        
    def decode(self, ids: List[int], remove_special_tokens: bool = True) -> str:
        tokens = []
        for i in ids:
            if remove_special_tokens and i in (self.pad_id, self.sos_id, self.eos_id):
                continue
            tok = self.id2token.get(i, self.UNK_TOKEN)
            tokens.append(tok)
        # Detokenize nicely
        text = " ".join(tokens)
        text = re.sub(r'\s+([,.:;?!])', r'\1', text)
        text = re.sub(r"\s+'\s+", "'", text)
        return text

    def __len__(self):
        return len(self.token2id)


if __name__ == "__main__":
    train, val, test = load_and_preprocess_dataset(max_samples=2000, save_dir="../data")
    vocab_en = SimpleVocabulary()
    vocab_en.build_vocab(train["source_en"].tolist())
    sample_text = "The quick brown fox jumps."
    enc = vocab_en.encode(sample_text)
    dec = vocab_en.decode(enc)
    print(f"Test encode: {sample_text} -> {enc}")
    print(f"Test decode: {dec}")
