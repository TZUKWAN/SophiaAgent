"""SophiaAgent academic paper review system.

Six-dimension review engine for post-writing quality assurance:
1. Authenticity — citation verification, data consistency, fact checking
2. Logic — methodology match, evidence support, chain integrity
3. Citations — format validation, list matching, phantom detection
4. Language — academic style, grammar, redundancy
5. Statistics — p-value consistency, effect size completeness, assumption reporting
6. Ethics — data fabrication detection, plagiarism flagging
"""

from sophia.review.engine import ReviewEngine

__all__ = ["ReviewEngine"]
