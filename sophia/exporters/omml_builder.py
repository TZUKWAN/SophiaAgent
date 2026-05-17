"""OMML (Office Math Markup Language) formula builder for Word documents.

Generates native editable math formulas that can be inserted into .docx
via python-docx.  Formulas are fully editable in Word's equation editor.

Reference: MiniMax Office Skills philosophy — direct XML manipulation
for production-grade fidelity.
"""

import html
import re
from typing import List, Optional, Tuple, Union

# OMML namespace declaration for use in parse_xml
OMML_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
OMML_NS_DECL = 'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"'

# Greek letter mapping: name → Unicode char
GREEK_LETTERS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "zeta": "ζ", "eta": "η", "theta": "θ", "iota": "ι", "kappa": "κ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "omicron": "ο",
    "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "upsilon": "υ",
    "phi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Alpha": "Α", "Beta": "Β", "Gamma": "Γ", "Delta": "Δ", "Epsilon": "Ε",
    "Theta": "Θ", "Lambda": "Λ", "Mu": "Μ", "Pi": "Π", "Sigma": "Σ",
    "Phi": "Φ", "Chi": "Χ", "Psi": "Ψ", "Omega": "Ω",
}

# Unicode Greek chars for reverse lookup
GREEK_UNICODE = {v: k for k, v in GREEK_LETTERS.items()}

# Common statistical symbols
STAT_SYMBOLS = {
    "<": "<", ">": ">", "<=": "≤", ">=": "≥", "!=": "≠",
    "+-": "±", "~": "~", "*": "*", "x": "×", "inf": "∞",
    "": "→", "": "⇒", "": "⇔",
}


class OMMLFragment:
    """A fragment of OMML XML that can be combined with others."""

    def __init__(self, xml: str):
        self.xml = xml

    def __str__(self) -> str:
        return self.xml

    def __add__(self, other) -> "OMMLFragment":
        if isinstance(other, OMMLFragment):
            return OMMLFragment(self.xml + other.xml)
        return OMMLFragment(self.xml + _escape_text(str(other)))


class OMMLBuilder:
    """Build OMML formulas for insertion into Word documents."""

    # ------------------------------------------------------------------
    # Low-level builders
    # ------------------------------------------------------------------

    @staticmethod
    def text(s: str) -> OMMLFragment:
        """Plain text run."""
        escaped = _escape_text(s)
        return OMMLFragment(f'<m:r><m:t>{escaped}</m:t></m:r>')

    @staticmethod
    def superscript(base: Union[str, OMMLFragment], sup: Union[str, OMMLFragment]) -> OMMLFragment:
        """Superscript: base^sup."""
        b = base.xml if isinstance(base, OMMLFragment) else OMMLBuilder.text(base).xml
        s = sup.xml if isinstance(sup, OMMLFragment) else OMMLBuilder.text(sup).xml
        return OMMLFragment(f"<m:sSup><m:e>{b}</m:e><m:sup>{s}</m:sup></m:sSup>")

    @staticmethod
    def subscript(base: Union[str, OMMLFragment], sub: Union[str, OMMLFragment]) -> OMMLFragment:
        """Subscript: base_sub."""
        b = base.xml if isinstance(base, OMMLFragment) else OMMLBuilder.text(base).xml
        s = sub.xml if isinstance(sub, OMMLFragment) else OMMLBuilder.text(sub).xml
        return OMMLFragment(f"<m:sSub><m:e>{b}</m:e><m:sub>{s}</m:sub></m:sSub>")

    @staticmethod
    def subsup(base: Union[str, OMMLFragment], sub: Union[str, OMMLFragment], sup: Union[str, OMMLFragment]) -> OMMLFragment:
        """Combined subscript + superscript."""
        b = base.xml if isinstance(base, OMMLFragment) else OMMLBuilder.text(base).xml
        s = sub.xml if isinstance(sub, OMMLFragment) else OMMLBuilder.text(sub).xml
        p = sup.xml if isinstance(sup, OMMLFragment) else OMMLBuilder.text(sup).xml
        return OMMLFragment(
            f"<m:sSubSup><m:e>{b}</m:e><m:sub>{s}</m:sub><m:sup>{p}</m:sup></m:sSubSup>"
        )

    @staticmethod
    def fraction(num: Union[str, OMMLFragment], den: Union[str, OMMLFragment]) -> OMMLFragment:
        """Fraction num/den."""
        n = num.xml if isinstance(num, OMMLFragment) else OMMLBuilder.text(num).xml
        d = den.xml if isinstance(den, OMMLFragment) else OMMLBuilder.text(den).xml
        return OMMLFragment(f"<m:f><m:num>{n}</m:num><m:den>{d}</m:den></m:f>")

    @staticmethod
    def sqrt(content: Union[str, OMMLFragment], degree: Optional[str] = None) -> OMMLFragment:
        """Square root or n-th root."""
        c = content.xml if isinstance(content, OMMLFragment) else OMMLBuilder.text(content).xml
        if degree is None:
            return OMMLFragment(f'<m:rad><m:radPr><m:degHide m:val="1"/></m:radPr><m:e>{c}</m:e></m:rad>')
        d = OMMLBuilder.text(degree).xml
        return OMMLFragment(f"<m:rad><m:deg>{d}</m:deg><m:e>{c}</m:e></m:rad>")

    @staticmethod
    def bracket(content: Union[str, OMMLFragment], left: str = "(", right: str = ")") -> OMMLFragment:
        """Bracketed expression with auto-sizing delimiters."""
        c = content.xml if isinstance(content, OMMLFragment) else OMMLBuilder.text(content).xml
        l = _escape_text(left)
        r = _escape_text(right)
        return OMMLFragment(
            f'<m:d><m:dPr><m:begChr m:val="{l}"/><m:endChr m:val="{r}"/></m:dPr>'
            f"<m:e>{c}</m:e></m:d>"
        )

    @staticmethod
    def greek(name: str) -> OMMLFragment:
        """Greek letter by name (e.g., 'beta', 'chi')."""
        char = GREEK_LETTERS.get(name, "?")
        return OMMLFragment(f'<m:r><m:t>{char}</m:t></m:r>')

    @staticmethod
    def operator(op: str) -> OMMLFragment:
        """Math operator or symbol."""
        char = STAT_SYMBOLS.get(op, op)
        return OMMLFragment(f'<m:r><m:t>{_escape_text(char)}</m:t></m:r>')

    # ------------------------------------------------------------------
    # High-level: common statistical expressions
    # ------------------------------------------------------------------

    @staticmethod
    def t_stat(df: Union[int, str], value: float) -> OMMLFragment:
        """t(df) = value, e.g. t(98) = 3.24."""
        return OMMLBuilder.text("t") + OMMLBuilder.bracket(str(df)) + OMMLBuilder.text(f" = {value:.2f}")

    @staticmethod
    def f_stat(df1: Union[int, str], df2: Union[int, str], value: float) -> OMMLFragment:
        """F(df1, df2) = value."""
        return OMMLBuilder.text("F") + OMMLBuilder.bracket(f"{df1}, {df2}") + OMMLBuilder.text(f" = {value:.2f}")

    @staticmethod
    def chi_sq(df: Union[int, str], value: float) -> OMMLFragment:
        """χ²(df) = value."""
        chi = OMMLBuilder.greek("chi")
        sup = OMMLBuilder.superscript(chi, "2")
        return sup + OMMLBuilder.bracket(str(df)) + OMMLBuilder.text(f" = {value:.2f}")

    @staticmethod
    def p_value(value: float) -> OMMLFragment:
        """p < .001 or p = .042."""
        if value < 0.001:
            return OMMLBuilder.text("p < .001")
        p_str = f"{value:.3f}"
        if p_str.startswith("0"):
            p_str = p_str[1:]  # "0.042" -> ".042"
        return OMMLBuilder.text(f"p = {p_str}")

    @staticmethod
    def cohens_d(value: float) -> OMMLFragment:
        """Cohen's d = value."""
        return OMMLBuilder.text("Cohen's d = ") + OMMLBuilder.text(f"{value:.2f}")

    @staticmethod
    def r_squared(value: float, adjusted: Optional[float] = None) -> OMMLFragment:
        """R² = value (adj. R² = adj)."""
        r2 = OMMLBuilder.superscript("R", "2")
        frag = r2 + OMMLBuilder.text(f" = {value:.3f}")
        if adjusted is not None:
            adj = OMMLBuilder.superscript("R", "2") + OMMLBuilder.subscript("", "adj")
            frag = frag + OMMLBuilder.text(", ") + OMMLBuilder.text(f"adj. R² = {adjusted:.3f}")
        return frag

    @staticmethod
    def beta_coeff(name: str, value: float, se: Optional[float] = None) -> OMMLFragment:
        """β_name = value (SE = se)."""
        if "_" in name:
            base, sub = name.split("_", 1)
            beta = OMMLBuilder.greek("beta") + OMMLBuilder.subscript("", sub)
        else:
            beta = OMMLBuilder.greek("beta") + OMMLBuilder.subscript("", name)
        frag = beta + OMMLBuilder.text(f" = {value:.3f}")
        if se is not None:
            frag = frag + OMMLBuilder.text(f", SE = {se:.3f}")
        return frag

    @staticmethod
    def ci(interval: Tuple[float, float]) -> OMMLFragment:
        """95% CI [low, high]."""
        return OMMLBuilder.text(f"95% CI [{interval[0]:.2f}, {interval[1]:.2f}]")

    @staticmethod
    def eta_squared(value: float) -> OMMLFragment:
        """η² = value."""
        eta2 = OMMLBuilder.superscript(OMMLBuilder.greek("eta"), "2")
        return eta2 + OMMLBuilder.text(f" = {value:.3f}")

    @staticmethod
    def omega_squared(value: float) -> OMMLFragment:
        """ω² = value."""
        omega2 = OMMLBuilder.superscript(OMMLBuilder.greek("omega"), "2")
        return omega2 + OMMLBuilder.text(f" = {value:.3f}")

    @staticmethod
    def sem(value: float) -> OMMLFragment:
        """SEM = value."""
        return OMMLBuilder.text(f"SEM = {value:.3f}")

    @staticmethod
    def cronbach_alpha(value: float) -> OMMLFragment:
        """Cronbach's α = value."""
        return OMMLBuilder.text("Cronbach's ") + OMMLBuilder.greek("alpha") + OMMLBuilder.text(f" = {value:.3f}")

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    @staticmethod
    def join(*parts: Union[str, OMMLFragment], separator: str = " ") -> OMMLFragment:
        """Join multiple fragments with a separator."""
        result = OMMLFragment("")
        sep = OMMLBuilder.text(separator)
        for i, part in enumerate(parts):
            if i > 0:
                result = OMMLFragment(result.xml + sep.xml)
            p = part.xml if isinstance(part, OMMLFragment) else OMMLBuilder.text(part).xml
            result = OMMLFragment(result.xml + p)
        return result

    @staticmethod
    def wrap_math(fragment: Union[str, OMMLFragment]) -> str:
        """Wrap a fragment in m:oMath with namespace declaration for parse_xml."""
        xml = fragment.xml if isinstance(fragment, OMMLFragment) else OMMLBuilder.text(fragment).xml
        return f'<m:oMath {OMML_NS_DECL}>{xml}</m:oMath>'


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _escape_text(s: str) -> str:
    """Escape XML special characters in text content."""
    s = str(s)
    s = s.replace("&", "&amp;")
    s = s.replace("<", "&lt;")
    s = s.replace(">", "&gt;")
    s = s.replace('"', "&quot;")
    return s


def insert_omml(paragraph, fragment: Union[str, OMMLFragment]):
    """Insert an OMML formula into a python-docx paragraph.

    Args:
        paragraph: A docx.paragraph.Paragraph object.
        fragment: OMMLFragment or string to wrap as math.
    """
    from docx.oxml import parse_xml

    xml = OMMLBuilder.wrap_math(fragment)
    element = parse_xml(xml)
    paragraph._p.append(element)
