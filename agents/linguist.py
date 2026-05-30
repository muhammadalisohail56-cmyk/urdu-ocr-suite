import re
from typing import Dict, Any

class LinguistAgent:
    def __init__(self):
        pass

    def process(self, token: Dict[str, Any]) -> Dict[str, Any]:
        """Applies Urdu-specific structural cleanup without changing meaning."""
        original_text = token.get("text", "")
        new_text = original_text
        
        changes = []
        
        # Normalize Arabic Yeh to Urdu Yeh
        if 'ي' in new_text:
            new_text = new_text.replace('ي', 'ی')
            changes.append("Normalized Yeh")
            
        # Normalize Arabic Kaf to Urdu Kaf
        if 'ك' in new_text:
            new_text = new_text.replace('ك', 'ک')
            changes.append("Normalized Kaf")
            
        # Remove stray zero-width joiners/non-joiners at boundaries
        if new_text.startswith('\u200c') or new_text.endswith('\u200c') or new_text.startswith('\u200d') or new_text.endswith('\u200d'):
            new_text = new_text.strip('\u200c\u200d')
            changes.append("Removed stray ZWNJ/ZWJ")

        if new_text != original_text:
            token["text"] = new_text
            # Lower confidence slightly because we auto-edited it
            token["confidence"] = max(0.1, token.get("confidence", 1.0) - 0.05)
            
            rationale = ", ".join(changes)
            existing_rationale = token.get("rationale", "")
            if existing_rationale:
                token["rationale"] = f"{existing_rationale} | Linguist: {rationale}"
            else:
                token["rationale"] = f"Linguist: {rationale}"
                
            flags = token.get("flags", [])
            if "linguist_edited" not in flags:
                flags.append("linguist_edited")
            token["flags"] = flags
            
        return token
