"""
Persistent Custom Rules & User Feedback Manager
-----------------------------------------------
Enables users to define custom merchant-to-category overrides that persist
in a JSON store and take top precedence (Tier 0) in the classification pipeline.
"""

import json
import os
import re
from typing import Dict, List, Optional

DEFAULT_RULES_PATH = "data/user_rules.json"


class UserRuleManager:
    """Manages persistent custom merchant rules defined by the user."""

    def __init__(self, rules_path: str = DEFAULT_RULES_PATH):
        self.rules_path = rules_path
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        os.makedirs(os.path.dirname(self.rules_path), exist_ok=True)
        if not os.path.exists(self.rules_path):
            with open(self.rules_path, "w") as f:
                json.dump({}, f, indent=2)

    def load_rules(self) -> Dict[str, str]:
        """Load all custom rules as a dictionary {normalized_keyword: category}."""
        try:
            with open(self.rules_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def add_rule(self, keyword: str, category: str) -> bool:
        """Add or update a custom rule mapping."""
        clean_kw = keyword.strip().lower()
        if not clean_kw:
            return False

        rules = self.load_rules()
        rules[clean_kw] = category

        with open(self.rules_path, "w") as f:
            json.dump(rules, f, indent=2)
        return True

    def delete_rule(self, keyword: str) -> bool:
        """Remove a custom rule mapping."""
        clean_kw = keyword.strip().lower()
        rules = self.load_rules()
        if clean_kw in rules:
            del rules[clean_kw]
            with open(self.rules_path, "w") as f:
                json.dump(rules, f, indent=2)
            return True
        return False

    def match(self, cleaned_text: str) -> Optional[str]:
        """
        Check if any custom rule keyword appears in the transaction text.
        Returns category if matched, else None.
        """
        rules = self.load_rules()
        if not rules:
            return None

        for kw, category in rules.items():
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, cleaned_text, re.IGNORECASE):
                return category
        return None


if __name__ == "__main__":
    mgr = UserRuleManager()
    mgr.add_rule("iron den gym", "Health & Medical")
    print(f"Loaded rules: {mgr.load_rules()}")
    matched = mgr.match("monthly sub iron den gym blr")
    print(f"Match test: {matched}")

