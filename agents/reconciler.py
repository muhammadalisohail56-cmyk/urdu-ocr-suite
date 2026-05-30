import difflib
import re
from typing import List, Dict, Any
from collections import Counter

def normalize_urdu_word(word: str) -> str:
    """Normalize for comparison purposes only. Never shown to user."""
    # remove diacritics (zer, zabar, pesh, etc)
    word = re.sub(r'[\u064B-\u065F]', '', word)
    # normalize yeh
    word = word.replace('ي', 'ی')
    # normalize kaf
    word = word.replace('ك', 'ک')
    # remove zero-width joiner/non-joiner
    word = word.replace('\u200c', '').replace('\u200d', '')
    return word.strip()

def tokenize(text: str) -> List[str]:
    # split by whitespace but keep words intact
    return [w for w in text.split() if w]

class ReconcilerAgent:
    def __init__(self):
        pass

    def reconcile(self, agent_outputs: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        agent_outputs: e.g. {"gemini": ["line 1", "line 2"], "openai": ["line 1..."]}
        Returns a list of tokens with confidence, and overall confidence.
        """
        # flatten into words
        tokenized_outputs = {}
        for provider, lines in agent_outputs.items():
            text = " ".join(lines)
            tokenized_outputs[provider] = tokenize(text)

        if not tokenized_outputs:
            return {"tokens": [], "confidence": 0.0}

        # pick the longest sequence as the anchor
        anchor_provider = max(tokenized_outputs.keys(), key=lambda k: len(tokenized_outputs[k]))
        anchor_tokens = tokenized_outputs[anchor_provider]
        
        # for each token in anchor, we will align tokens from other providers
        aligned_results = []
        for i in range(len(anchor_tokens)):
            aligned_results.append({
                anchor_provider: anchor_tokens[i]
            })

        # align others to anchor
        for provider, tokens in tokenized_outputs.items():
            if provider == anchor_provider:
                continue
            
            # normalize for matching
            anchor_norm = [normalize_urdu_word(t) for t in anchor_tokens]
            provider_norm = [normalize_urdu_word(t) for t in tokens]
            
            sm = difflib.SequenceMatcher(None, anchor_norm, provider_norm)
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag in ('equal', 'replace'):
                    # map provider tokens to anchor tokens where possible
                    # for 'replace', we just assume they align positionally
                    for idx in range(i2 - i1):
                        anchor_idx = i1 + idx
                        provider_idx = j1 + min(idx, (j2 - j1) - 1)
                        if provider_idx < j2:
                            aligned_results[anchor_idx][provider] = tokens[provider_idx]
                elif tag == 'insert':
                    # tokens in provider not in anchor. For simplicity in this demo,
                    # we ignore insertions or attach them to the previous anchor token.
                    # A robust approach would create new slots, but anchor is the longest.
                    pass
                elif tag == 'delete':
                    # tokens in anchor not in provider. They just don't get a vote from this provider.
                    pass

        # Adjudicate
        final_tokens = []
        total_confidence = 0.0
        
        num_providers = len(tokenized_outputs)
        
        for slot in aligned_results:
            # slot is e.g. {"gemini": "word", "openai": "word", "claude": "word2"}
            votes = list(slot.values())
            # to count votes, normalize them
            norm_votes = [normalize_urdu_word(v) for v in votes]
            counter = Counter(norm_votes)
            
            # most common normalized word
            best_norm, count = counter.most_common(1)[0]
            
            # find the original word corresponding to best_norm
            best_word = next(v for v in votes if normalize_urdu_word(v) == best_norm)
            
            # Determine confidence
            # all agree -> 1.0 (high)
            # majority -> 0.6 (medium)
            # tie / alone -> 0.3 (low)
            if count == num_providers:
                conf = 1.0
            elif count > num_providers / 2:
                conf = 0.6
            else:
                conf = 0.3
                
            final_tokens.append({
                "text": best_word,
                "confidence": conf,
                "votes": slot  # store who voted for what
            })
            total_confidence += conf
            
        avg_confidence = total_confidence / len(final_tokens) if final_tokens else 0.0
        return {
            "tokens": final_tokens,
            "confidence": avg_confidence
        }
