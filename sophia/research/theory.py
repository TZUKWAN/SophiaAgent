"""Theory mapping, concept history, and school comparison for SophiaAgent.

Provides TheoryMapper with built-in knowledge base, LLM-augmented analysis,
concept evolution tracing, and multi-school comparison.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_KB_PATH = os.path.join(os.path.dirname(__file__), "data", "theory_kb.json")

_CONCEPT_CACHE: Dict[str, Dict[str, Any]] = {}

_RELATION_TYPES = {"extends", "contradicts", "complements", "influences"}

# Pre-computed concept histories (zh)
_PRECOMPUTED_CONCEPTS = {
    "社会资本": {
        "concept": "社会资本",
        "evolution_stages": [
            {
                "period": "1916-1970s",
                "discipline": "经济学/社会学",
                "definition": "Hanifan (1916) 首次提出社会资本概念，指社区中 goodwill、友谊和同情心的集合",
                "key_authors": ["L. J. Hanifan"],
                "seminal_works": ["Hanifan, L. J. (1916). The Rural School Community Center."],
                "shift_description": "概念萌芽期，主要用于社区发展和教育领域"
            },
            {
                "period": "1970s-1980s",
                "discipline": "社会学",
                "definition": "Bourdieu 将社会资本定义为实际或潜在资源的集合，与持久网络相联系",
                "key_authors": ["Pierre Bourdieu"],
                "seminal_works": ["Bourdieu, P. (1980). Le capital social."],
                "shift_description": "从社区层面转向阶级再生产分析，强调资本的阶级属性"
            },
            {
                "period": "1988-1990s",
                "discipline": "社会学/经济学",
                "definition": "Coleman 将社会资本定义为社会结构中有利于行动者行动的方面，强调其功能属性",
                "key_authors": ["James Coleman"],
                "seminal_works": ["Coleman, J. (1988). Social Capital in the Creation of Human Capital."],
                "shift_description": "功能主义转向，社会资本被视为促进人力资本积累的工具"
            },
            {
                "period": "1993-2000s",
                "discipline": "政治学/社会学",
                "definition": "Putnam 将社会资本定义为促进合作的网络、规范和信任，强调其民主功能",
                "key_authors": ["Robert Putnam"],
                "seminal_works": ["Putnam, R. (1993). Making Democracy Work.", "Putnam, R. (2000). Bowling Alone."],
                "shift_description": "民主化转向，社会资本与公民参与、民主绩效关联"
            },
            {
                "period": "2000s-至今",
                "discipline": "多学科",
                "definition": "社会资本概念进一步分化：桥接型/黏合型/连接型社会资本，在线社会资本等",
                "key_authors": ["Robert Putnam", "Alejandro Portes", "Nan Lin"],
                "seminal_works": ["Putnam, R. (2000). Bowling Alone.", "Portes, A. (1998). Social Capital: Its Origins and Applications."],
                "shift_description": "概念精细化与批判，关注负面效应（封闭、排他）和数字时代的新形态"
            }
        ],
        "current_debates": [
            "社会资本是否必然产生积极效应？（Portes 的负面社会资本批判）",
            "在线社会资本与传统社会资本有何异同？",
            "社会资本测量方法的可靠性问题",
            "社会资本与制度质量之间的因果关系方向"
        ],
        "cross_disciplinary_usage": {
            "社会学": "社会分层、社区研究、网络分析",
            "政治学": "民主化、公民参与、治理绩效",
            "经济学": "发展经济学、创业研究、劳动力市场",
            "教育学": "教育不平等、学业成就、学校社区",
            "传播学": "社交媒体、信息传播、数字参与"
        }
    },
    "内卷": {
        "concept": "内卷",
        "evolution_stages": [
            {
                "period": "1963",
                "discipline": "人类学",
                "definition": "Geertz 借用 involution 描述爪哇水稻农业在边际报酬递减下的精细化过程",
                "key_authors": ["Clifford Geertz"],
                "seminal_works": ["Geertz, C. (1963). Agricultural Involution."],
                "shift_description": "概念引入期，描述农业经济中的自我复制型增长"
            },
            {
                "period": "1980s-1990s",
                "discipline": "历史学/社会学",
                "definition": "黄宗智将 involution 译为'内卷'，描述明清时期小农经济在人口压力下劳动边际报酬递减的现象",
                "key_authors": ["Philip Huang (黄宗智)"],
                "seminal_works": ["Huang, P. C. C. (1985). The Peasant Economy and Social Change in North China."],
                "shift_description": "概念中国化，与'过密化'并用，成为理解中国小农经济的核心概念"
            },
            {
                "period": "2010s-至今",
                "discipline": "社会学/文化研究",
                "definition": "内卷成为描述教育、职场等领域非理性竞争的流行语，指无发展的增长和无效努力",
                "key_authors": ["项飙", " various Chinese scholars"],
                "seminal_works": ["项飙访谈（2020）关于'内卷'作为社会心态的讨论"],
                "shift_description": "概念泛化与流行化，从学术概念转变为社会文化关键词，描述竞争加剧下的焦虑"
            }
        ],
        "current_debates": [
            "内卷概念在流行语境中是否被过度泛化，失去学术精确性？",
            "内卷与'躺平'之间的辩证关系",
            "内卷是结构性问题还是个体选择问题？",
            "如何区分健康的竞争与内卷？"
        ],
        "cross_disciplinary_usage": {
            "人类学": "农业经济、文化演化",
            "历史学": "明清经济史、小农经济",
            "社会学": "教育竞争、职场文化、社会分层",
            "教育学": "学业竞争、教育焦虑、择校",
            "文化研究": "青年亚文化、社会心态"
        }
    },
    "数字劳动": {
        "concept": "数字劳动",
        "evolution_stages": [
            {
                "period": "1990s-2000s",
                "discipline": "传播学/社会学",
                "definition": "Dallas Smythe 提出'受众商品论'，认为受众的注意力是媒体生产的商品",
                "key_authors": ["Dallas Smythe"],
                "seminal_works": ["Smythe, D. (1977). Communications: Blindspot of Western Marxism."],
                "shift_description": "数字劳动的思想源头，将受众观看行为定义为一种劳动"
            },
            {
                "period": "2000s-2010s",
                "discipline": "传播学/政治经济学",
                "definition": "Tiziana Terranova 提出'免费劳动'概念，描述互联网用户无偿的内容生产",
                "key_authors": ["Tiziana Terranova"],
                "seminal_works": ["Terranova, T. (2000). Free Labor: Producing Culture for the Digital Economy."],
                "shift_description": "从受众商品到免费劳动，关注互联网经济中的无偿数字生产"
            },
            {
                "period": "2010s-至今",
                "discipline": "多学科",
                "definition": "数字劳动概念扩展为平台劳动、众包劳动、玩劳动、情感劳动等多种形式",
                "key_authors": ["Nick Dyer-Witheford", "Ursula Huws", "孙萍"],
                "seminal_works": ["Dyer-Witheford, N. (2015). Cyber-Proletariat.", "孙萍 (2019). 过渡劳动：平台经济下的外卖骑手"],
                "shift_description": "概念体系化，关注平台经济中的劳动者权益、算法管理和劳动过程"
            }
        ],
        "current_debates": [
            "数字劳动是否构成马克思意义上的'劳动'？",
            "平台劳动者的法律身份认定问题",
            "算法管理对传统劳动控制的替代与强化",
            "全球数字劳动分工中的南北不平等"
        ],
        "cross_disciplinary_usage": {
            "传播学": "平台政治经济学、受众劳动",
            "社会学": "劳动社会学、平台经济",
            "法学": "劳动法、平台治理",
            "经济学": "平台经济、零工经济",
            "地理学": "数字地理、空间劳动"
        }
    },
    "文化资本": {
        "concept": "文化资本",
        "evolution_stages": [
            {
                "period": "1970s",
                "discipline": "社会学",
                "definition": "Bourdieu 提出文化资本概念，指通过家庭和社会环境获得的文化能力和品味",
                "key_authors": ["Pierre Bourdieu"],
                "seminal_works": ["Bourdieu, P. (1979). La Distinction."],
                "shift_description": "概念创立，区分身体化、客观化和制度化三种形态"
            },
            {
                "period": "1980s-1990s",
                "discipline": "教育学/社会学",
                "definition": "文化资本被广泛用于解释教育不平等，强调家庭文化环境对学业成就的影响",
                "key_authors": ["Annette Lareau", "Paul DiMaggio"],
                "seminal_works": ["Lareau, A. (1989). Home Advantage.", "DiMaggio, P. (1982). Cultural Capital and School Success."],
                "shift_description": "教育社会学转向，文化资本成为解释教育分层的核心变量"
            },
            {
                "period": "2000s-至今",
                "discipline": "多学科",
                "definition": "文化资本概念扩展至数字文化资本、全球文化资本等新形态",
                "key_authors": ["Tony Bennett", "Mike Savage"],
                "seminal_works": ["Bennett, T. (2009). Culture, Class, Distinction.", "Savage, M. (2015). Social Class in the 21st Century."],
                "shift_description": "概念更新与批判，关注文化资本在数字时代和全球化背景下的新形态"
            }
        ],
        "current_debates": [
            "文化资本与人力资本的概念边界",
            "文化资本测量方法的争议",
            "数字文化资本是否构成新的资本形态？",
            "文化资本概念在不同文化语境中的适用性"
        ],
        "cross_disciplinary_usage": {
            "社会学": "社会分层、文化消费、品味研究",
            "教育学": "教育不平等、学业成就、家庭教育",
            "传播学": "媒介素养、数字文化",
            "艺术学": "艺术消费、文化政策"
        }
    },
    "治理": {
        "concept": "治理",
        "evolution_stages": [
            {
                "period": "1980s-1990s",
                "discipline": "政治学/公共管理",
                "definition": "世界银行提出'治理'概念，指国家管理经济和社会资源的方式",
                "key_authors": ["世界银行", "James Rosenau"],
                "seminal_works": ["World Bank (1989). Sub-Saharan Africa: From Crisis to Sustainable Growth.", "Rosenau, J. (1992). Governance, Order, and Change in World Politics."],
                "shift_description": "概念引入期，治理作为'良好政府管理'的替代话语"
            },
            {
                "period": "1990s-2000s",
                "discipline": "政治学/社会学",
                "definition": "治理理论发展为多元主体协同的网络治理，超越政府单一中心",
                "key_authors": ["R. A. W. Rhodes", "Stephen Osborne"],
                "seminal_works": ["Rhodes, R. A. W. (1996). The New Governance.", "Osborne, S. (2010). The New Public Governance."],
                "shift_description": "理论深化，从政府管理转向多元网络治理"
            },
            {
                "period": "2000s-至今",
                "discipline": "多学科",
                "definition": "治理概念进一步扩展为全球治理、数字治理、环境治理、城市治理等",
                "key_authors": ["Anne-Marie Slaughter", "Helen Milner"],
                "seminal_works": ["Slaughter, A. M. (2004). A New World Order.", "Milner, H. (2021). Digital Governance."],
                "shift_description": "概念泛化与精细化，治理成为跨学科的核心分析框架"
            }
        ],
        "current_debates": [
            "治理概念是否因过度泛化而失去分析力？",
            "国家与社会在治理中的权力关系",
            "数字技术如何重塑治理模式？",
            "全球治理的民主赤字问题"
        ],
        "cross_disciplinary_usage": {
            "政治学": "公共政策、民主治理、国家能力",
            "社会学": "社区治理、社会组织",
            "管理学": "公司治理、公共管理",
            "环境科学": "环境治理、可持续发展",
            "传播学": "数字治理、平台治理"
        }
    },
    "全球化": {
        "concept": "全球化",
        "evolution_stages": [
            {
                "period": "1980s-1990s",
                "discipline": "经济学/社会学",
                "definition": "全球化指商品、资本、技术和劳动力在全球范围内的加速流动",
                "key_authors": ["Theodore Levitt", "Anthony Giddens"],
                "seminal_works": ["Levitt, T. (1983). The Globalization of Markets.", "Giddens, A. (1990). The Consequences of Modernity."],
                "shift_description": "概念兴起期，全球化被视为不可逆转的历史趋势"
            },
            {
                "period": "1990s-2000s",
                "discipline": "社会学/政治学",
                "definition": "全球化理论分化，出现'超全球化论者'与'怀疑论者'的争论",
                "key_authors": ["David Held", "Joseph Stiglitz", "Arjun Appadurai"],
                "seminal_works": ["Held, D. (1999). Global Transformations.", "Stiglitz, J. (2002). Globalization and Its Discontents."],
                "shift_description": "理论辩论期，批判视角兴起，关注全球化的不平等效应"
            },
            {
                "period": "2010s-至今",
                "discipline": "多学科",
                "definition": "全球化进入'慢全球化'或'逆全球化'阶段，数字全球化成为新维度",
                "key_authors": ["Dani Rodrik", "Thomas Friedman"],
                "seminal_works": ["Rodrik, D. (2011). The Globalization Paradox.", "Friedman, T. (2005). The World Is Flat."],
                "shift_description": "现实转向，逆全球化、民粹主义与数字全球化并存"
            }
        ],
        "current_debates": [
            "全球化是否正在逆转？",
            "全球化与不平等的关系",
            "数字全球化与传统全球化的差异",
            "全球化与民族国家主权的张力"
        ],
        "cross_disciplinary_usage": {
            "经济学": "国际贸易、全球价值链、金融全球化",
            "社会学": "跨国移民、文化全球化、全球城市",
            "政治学": "全球治理、国际关系、主权",
            "传播学": "全球媒介、跨文化传播",
            "文化研究": "文化混杂、全球本土化"
        }
    },
    "身份认同": {
        "concept": "身份认同",
        "evolution_stages": [
            {
                "period": "1950s-1960s",
                "discipline": "心理学/社会学",
                "definition": "Erikson 提出'认同危机'概念，描述青少年期的自我整合任务",
                "key_authors": ["Erik Erikson"],
                "seminal_works": ["Erikson, E. (1950). Childhood and Society.", "Erikson, E. (1968). Identity: Youth and Crisis."],
                "shift_description": "心理学起源，关注个体生命历程中的自我整合"
            },
            {
                "period": "1970s-1980s",
                "discipline": "社会学/政治学",
                "definition": "身份认同从个体层面扩展到集体层面，关注族群、性别、阶级等群体认同",
                "key_authors": ["Stuart Hall", "Charles Taylor"],
                "seminal_works": ["Hall, S. (1996). Questions of Cultural Identity.", "Taylor, C. (1992). Sources of the Self."],
                "shift_description": "社会学转向，身份认同成为理解社会运动和政治冲突的核心概念"
            },
            {
                "period": "1990s-至今",
                "discipline": "多学科",
                "definition": "身份认同理论进一步发展为交叉性、流动性、表演性等多元视角",
                "key_authors": ["Judith Butler", "Patricia Hill Collins"],
                "seminal_works": ["Butler, J. (1990). Gender Trouble.", "Collins, P. H. (2000). Black Feminist Thought."],
                "shift_description": "后现代转向，强调身份的流动性、多重性和建构性"
            }
        ],
        "current_debates": [
            "身份认同是本质主义的还是建构主义的？",
            "数字身份与线下身份的关系",
            "身份政治的积极效应与消极效应",
            "交叉性理论的方法论挑战"
        ],
        "cross_disciplinary_usage": {
            "心理学": "自我概念、发展心理学、人格",
            "社会学": "族群关系、性别研究、阶级认同",
            "政治学": "民族主义、公民身份、政治参与",
            "传播学": "媒介认同、粉丝文化、数字身份",
            "文化研究": "文化身份、后殖民认同、流散"
        }
    },
    "后真相": {
        "concept": "后真相",
        "evolution_stages": [
            {
                "period": "1990s-2000s",
                "discipline": "哲学/政治学",
                "definition": "后真相概念最早用于描述后现代主义对客观真理的解构",
                "key_authors": ["Jean Baudrillard", "Steve Tesich"],
                "seminal_works": ["Baudrillard, J. (1981). Simulacra and Simulation.", "Tesich, S. (1992). A Government of Lies."],
                "shift_description": "概念萌芽，描述政治话语与客观现实的脱节"
            },
            {
                "period": "2016-至今",
                "discipline": "传播学/政治学",
                "definition": "后真相成为年度词汇，描述情感和个人信念比客观事实更能影响舆论的时代",
                "key_authors": ["Oxford Dictionaries", "Lee McIntyre"],
                "seminal_works": ["McIntyre, L. (2018). Post-Truth.", "D'Ancona, M. (2017). Post-Truth: The New War on Truth and How to Fight Back."],
                "shift_description": "概念爆发，与英国脱欧、特朗普当选等事件关联，成为理解当代政治传播的关键词"
            }
        ],
        "current_debates": [
            "后真相是全新现象还是历史的延续？",
            "社交媒体算法是否加剧了后真相？",
            "事实核查能否有效对抗后真相？",
            "后真相与民粹主义的关系"
        ],
        "cross_disciplinary_usage": {
            "传播学": "假新闻、信息疫情、媒介素养",
            "政治学": "民粹主义、政治传播、公共舆论",
            "哲学": "真理理论、认识论、相对主义",
            "心理学": "确认偏误、认知偏差、情感决策",
            "教育学": "媒介素养教育、批判性思维"
        }
    }
}

# ---------------------------------------------------------------------------
# TheoryMapper
# ---------------------------------------------------------------------------

class TheoryMapper:
    """Map topics to theories, trace concept evolution, and compare schools."""

    def __init__(self, kb_path: Optional[str] = None, provider=None):
        self.kb_path = kb_path or _KB_PATH
        self.provider = provider
        self._kb: Dict[str, Any] = {}
        self._theories_by_id: Dict[str, Dict[str, Any]] = {}
        self._theories_by_discipline: Dict[str, List[Dict[str, Any]]] = {}
        self._load_kb()

    # -----------------------------------------------------------------------
    # Knowledge base
    # -----------------------------------------------------------------------

    def _load_kb(self) -> None:
        try:
            with open(self.kb_path, "r", encoding="utf-8") as f:
                self._kb = json.load(f)
        except FileNotFoundError:
            logger.warning("Theory KB not found at %s", self.kb_path)
            self._kb = {"theories": [], "disciplines": []}
        except json.JSONDecodeError as exc:
            logger.warning("Theory KB JSON error: %s", exc)
            self._kb = {"theories": [], "disciplines": []}

        for t in self._kb.get("theories", []):
            tid = t.get("theory_id")
            if tid:
                self._theories_by_id[tid] = t
                disc = t.get("discipline", "unknown")
                self._theories_by_discipline.setdefault(disc, []).append(t)

    def list_theories(self, discipline: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return list of theories, optionally filtered by discipline."""
        if discipline:
            return list(self._theories_by_discipline.get(discipline, []))
        return list(self._kb.get("theories", []))

    def get_theory(self, theory_id: str) -> Optional[Dict[str, Any]]:
        return self._theories_by_id.get(theory_id)

    def list_disciplines(self) -> List[str]:
        return list(self._kb.get("disciplines", []))

    # -----------------------------------------------------------------------
    # E-1: map_theories
    # -----------------------------------------------------------------------

    def map_theories(self, topic: str, discipline: Optional[str] = None) -> Dict[str, Any]:
        """Map a research topic to relevant theories.

        Uses rule-based keyword matching from the knowledge base, with optional
        LLM augmentation when a provider is available.
        """
        if not topic or not topic.strip():
            return {"topic": "", "theories": [], "relations": [], "recommended": [], "error": "topic is required"}

        topic_clean = topic.strip()

        # Rule path: keyword matching
        candidates = self._rule_match(topic_clean, discipline)

        # LLM path: if provider available, augment with LLM analysis
        llm_theories = []
        if self.provider is not None:
            try:
                llm_theories = self._llm_match(topic_clean, discipline)
            except Exception as exc:
                logger.warning("LLM theory mapping failed: %s", exc)

        # Merge results
        merged = self._merge_matches(candidates, llm_theories)

        # Build relations between matched theories
        relations = self._build_relations(merged)

        # Recommend top theories
        recommended = [t["theory_id"] for t in merged[:5]]

        return {
            "topic": topic_clean,
            "theories": merged,
            "relations": relations,
            "recommended": recommended,
        }

    def _rule_match(self, topic: str, discipline: Optional[str] = None) -> List[Dict[str, Any]]:
        """Keyword-based matching from the knowledge base."""
        topic_lower = topic.lower()
        topic_tokens = set(re.findall(r"[一-鿿]+|\w+", topic_lower))

        theories = self.list_theories(discipline)
        scored = []

        for t in theories:
            score = 0.0
            # Match name
            name_cn = t.get("name_cn", "")
            name_en = t.get("name_en", "")
            if name_cn and name_cn in topic:
                score += 10.0
            if name_en and name_en.lower() in topic_lower:
                score += 8.0

            # Match key concepts
            for concept in t.get("key_concepts", []):
                if concept in topic:
                    score += 5.0
                elif concept.lower() in topic_lower:
                    score += 4.0

            # Match core propositions
            for prop in t.get("core_propositions", []):
                prop_tokens = set(re.findall(r"[一-鿿]+|\w+", prop.lower()))
                if prop_tokens:
                    overlap = len(prop_tokens & topic_tokens) / len(prop_tokens)
                    score += overlap * 3.0

            # Match founders
            for founder in t.get("founders", []):
                if founder.lower() in topic_lower:
                    score += 3.0

            # Match discipline relevance
            disc = t.get("discipline", "")
            if discipline and disc == discipline:
                score += 2.0

            if score > 0:
                scored.append((score, t))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, t in scored:
            results.append({
                "theory_id": t["theory_id"],
                "name": t.get("name_cn", t.get("name_en", "")),
                "name_en": t.get("name_en", ""),
                "discipline": t.get("discipline", ""),
                "relevance_score": round(min(score / 20.0, 1.0), 3),
                "relation_to_topic": self._describe_relation(t, topic),
            })

        return results

    def _llm_match(self, topic: str, discipline: Optional[str] = None) -> List[Dict[str, Any]]:
        """Use LLM to suggest relevant theories."""
        disc_hint = f" within the discipline of {discipline}" if discipline else ""
        prompt = (
            f"You are a social science methodology expert. A researcher is studying: '{topic}'{disc_hint}.\n\n"
            f"From the following built-in theory knowledge base, identify the most relevant theories. "
            f"For each, provide: theory_id, relevance_score (0-1), and a brief explanation of how it relates to the topic.\n\n"
            f"Available theories:\n"
        )
        for t in self.list_theories(discipline):
            prompt += f"- {t['theory_id']}: {t.get('name_cn', '')} ({t.get('name_en', '')}) [{t.get('discipline', '')}]\n"

        prompt += (
            "\nReturn ONLY a JSON array like:\n"
            '[{"theory_id": "...", "relevance_score": 0.9, "relation_to_topic": "..."}]\n'
            "If none are relevant, return []."
        )

        response = self._call_llm(prompt)
        if not response:
            return []

        # Try to extract JSON
        try:
            # Find JSON array in response
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                if isinstance(data, list):
                    return data
        except json.JSONDecodeError:
            pass

        return []

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM provider if available."""
        if self.provider is None:
            return ""
        try:
            # Support different provider interfaces
            if hasattr(self.provider, "chat"):
                resp = self.provider.chat([{"role": "user", "content": prompt}], tools=None)
                if hasattr(resp, "content"):
                    return resp.content or ""
                if isinstance(resp, dict):
                    return resp.get("content", "") or resp.get("text", "")
                return str(resp)
            if hasattr(self.provider, "run"):
                result = self.provider.run(prompt=prompt)
                if isinstance(result, dict):
                    return result.get("text", "") or result.get("content", "")
                return str(result)
            return ""
        except Exception as exc:
            logger.warning("LLM call failed: %s", exc)
            return ""

    def _merge_matches(self, rule_results: List[Dict], llm_results: List[Dict]) -> List[Dict]:
        """Merge rule-based and LLM results, deduplicating by theory_id."""
        merged: Dict[str, Dict] = {}

        for r in rule_results:
            tid = r["theory_id"]
            merged[tid] = dict(r)

        for r in llm_results:
            tid = r.get("theory_id")
            if not tid:
                continue
            score = r.get("relevance_score", 0.0)
            if tid in merged:
                # Boost score if both paths agree
                merged[tid]["relevance_score"] = round(
                    min(merged[tid]["relevance_score"] * 0.6 + score * 0.4 + 0.1, 1.0), 3
                )
                if "relation_to_topic" in r and len(r["relation_to_topic"]) > len(merged[tid].get("relation_to_topic", "")):
                    merged[tid]["relation_to_topic"] = r["relation_to_topic"]
            else:
                t = self.get_theory(tid)
                merged[tid] = {
                    "theory_id": tid,
                    "name": t.get("name_cn", t.get("name_en", "")) if t else tid,
                    "name_en": t.get("name_en", "") if t else "",
                    "discipline": t.get("discipline", "") if t else "",
                    "relevance_score": round(score, 3),
                    "relation_to_topic": r.get("relation_to_topic", ""),
                }

        results = list(merged.values())
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results

    def _describe_relation(self, theory: Dict[str, Any], topic: str) -> str:
        """Generate a brief description of how a theory relates to a topic."""
        name = theory.get("name_cn", theory.get("name_en", ""))
        concepts = theory.get("key_concepts", [])[:3]
        propositions = theory.get("core_propositions", [])[:2]

        parts = [f"{name}通过其核心概念"]
        if concepts:
            parts.append("、".join(concepts))
        parts.append("为理解")
        parts.append(topic)
        parts.append("提供了理论框架")
        if propositions:
            parts.append(f"；其命题{'；'.join(propositions)}可直接应用于该主题")
        return "".join(parts)

    def _build_relations(self, theories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build relations between matched theories based on KB data."""
        relations = []
        tids = {t["theory_id"] for t in theories}

        for t in theories:
            tid = t["theory_id"]
            theory = self.get_theory(tid)
            if not theory:
                continue

            for related in theory.get("related_theories", []):
                if related in tids and related != tid:
                    relations.append({
                        "from": tid,
                        "to": related,
                        "type": "complements",
                        "strength": 0.7,
                    })

            for competing in theory.get("competing_theories", []):
                if competing in tids and competing != tid:
                    relations.append({
                        "from": tid,
                        "to": competing,
                        "type": "contradicts",
                        "strength": 0.8,
                    })

        # Deduplicate
        seen = set()
        unique = []
        for r in relations:
            key = (r["from"], r["to"], r["type"])
            if key not in seen:
                seen.add(key)
                unique.append(r)

        return unique

    # -----------------------------------------------------------------------
    # E-1: export_theory_map
    # -----------------------------------------------------------------------

    def export_theory_map(self, data: Dict[str, Any], format: str = "mermaid") -> str:
        """Export theory mapping data to diagram format.

        Supported formats: 'mermaid', 'tikz', 'dot'.
        """
        if format not in {"mermaid", "tikz", "dot"}:
            return f"# Error: unsupported format '{format}'. Use 'mermaid', 'tikz', or 'dot'."

        theories = data.get("theories", [])
        relations = data.get("relations", [])
        topic = data.get("topic", "Research Topic")

        if format == "mermaid":
            return self._to_mermaid(topic, theories, relations)
        if format == "tikz":
            return self._to_tikz(topic, theories, relations)
        return self._to_dot(topic, theories, relations)

    def _to_mermaid(self, topic: str, theories: List[Dict], relations: List[Dict]) -> str:
        lines = ["graph TD"]
        topic_id = "TOPIC"
        lines.append(f'    {topic_id}["{topic}"]')

        for t in theories:
            tid = t["theory_id"]
            name = t.get("name", tid)
            score = t.get("relevance_score", 0)
            lines.append(f'    {tid}["{name}<br/>relevance: {score}"]')
            lines.append(f'    {topic_id} --> {tid}')

        type_style = {
            "extends": "-.->",
            "contradicts": "-.-x",
            "complements": "==>",
            "influences": "-->",
        }

        for r in relations:
            arrow = type_style.get(r["type"], "-->")
            lines.append(f'    {r["from"]} {arrow}|{r["type"]}| {r["to"]}')

        return "\n".join(lines)

    def _to_tikz(self, topic: str, theories: List[Dict], relations: List[Dict]) -> str:
        lines = [
            "\\\\begin{tikzpicture}[node distance=2.5cm, auto]",
            "    % Topic node",
            f"    \\node[draw, rectangle, fill=blue!20] (topic) {{{topic}}};",
            "",
            "    % Theory nodes",
        ]

        positions = ["above right of=topic", "right of=topic", "below right of=topic",
                     "above left of=topic", "left of=topic", "below left of=topic"]

        for i, t in enumerate(theories):
            tid = t["theory_id"]
            name = t.get("name", tid).replace("&", "\\&")
            pos = positions[i % len(positions)]
            lines.append(f"    \\node[draw, rectangle, fill=green!10] ({tid}) [{pos} of=topic] {{{name}}};")

        lines.append("")
        lines.append("    % Relations")

        for r in relations:
            arrow_style = {
                "extends": "->, dashed",
                "contradicts": "->, red",
                "complements": "->, thick",
                "influences": "->",
            }.get(r["type"], "->")
            lines.append(f"    \\draw[{arrow_style}] ({r['from']}) -- node {{{r['type']}}} ({r['to']});")

        lines.append("\\\\end{tikzpicture}")
        return "\n".join(lines)

    def _to_dot(self, topic: str, theories: List[Dict], relations: List[Dict]) -> str:
        lines = [
            "digraph TheoryMap {",
            '    rankdir=TB;',
            '    node [shape=box, style=filled];',
            f'    topic [label="{topic}", fillcolor=lightblue];',
        ]

        for t in theories:
            tid = t["theory_id"]
            name = t.get("name", tid)
            score = t.get("relevance_score", 0)
            lines.append(f'    {tid} [label="{name}\\n({score})", fillcolor=lightgreen];')
            lines.append(f'    topic -> {tid};')

        type_color = {
            "extends": "blue",
            "contradicts": "red",
            "complements": "green",
            "influences": "gray",
        }

        for r in relations:
            color = type_color.get(r["type"], "black")
            lines.append(f'    {r["from"]} -> {r["to"]} [label="{r["type"]}", color={color}];')

        lines.append("}")
        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # E-2: trace_concept
    # -----------------------------------------------------------------------

    def trace_concept(self, concept: str, language: str = "zh") -> Dict[str, Any]:
        """Trace the historical evolution of a concept.

        Returns pre-computed results for known concepts, or attempts LLM generation
        for unknown concepts. Falls back to a message if LLM is unavailable.
        """
        if not concept or not concept.strip():
            return {"concept": "", "evolution_stages": [], "current_debates": [], "cross_disciplinary_usage": {}, "error": "concept is required"}

        concept_clean = concept.strip()

        # Check pre-computed cache
        if concept_clean in _PRECOMPUTED_CONCEPTS:
            return dict(_PRECOMPUTED_CONCEPTS[concept_clean])

        # Check runtime cache
        if concept_clean in _CONCEPT_CACHE:
            return dict(_CONCEPT_CACHE[concept_clean])

        # Try LLM
        if self.provider is not None:
            try:
                result = self._llm_trace_concept(concept_clean, language)
                if result:
                    _CONCEPT_CACHE[concept_clean] = result
                    return result
            except Exception as exc:
                logger.warning("LLM concept tracing failed: %s", exc)

        return {
            "concept": concept_clean,
            "evolution_stages": [],
            "current_debates": [],
            "cross_disciplinary_usage": {},
            "note": "该概念暂无预计算历史，且LLM不可用。请提供LLM支持以生成概念历史。",
        }

    def _llm_trace_concept(self, concept: str, language: str) -> Optional[Dict[str, Any]]:
        """Use LLM to generate concept history."""
        lang_hint = "in Chinese" if language == "zh" else "in English"
        prompt = (
            f"You are an expert in the history of social science concepts. "
            f"Trace the evolution of the concept '{concept}' {lang_hint}.\n\n"
            f"Return a JSON object with this exact structure:\n"
            f'{{\n'
            f'  "concept": "{concept}",\n'
            f'  "evolution_stages": [\n'
            f'    {{\n'
            f'      "period": "time period",\n'
            f'      "discipline": "primary discipline",\n'
            f'      "definition": "definition in that period",\n'
            f'      "key_authors": ["Author 1", "Author 2"],\n'
            f'      "seminal_works": ["Work 1", "Work 2"],\n'
            f'      "shift_description": "what changed in this period"\n'
            f'    }}\n'
            f'  ],\n'
            f'  "current_debates": ["debate 1", "debate 2"],\n'
            f'  "cross_disciplinary_usage": {{\n'
            f'    "discipline1": "usage description",\n'
            f'    "discipline2": "usage description"\n'
            f'  }}\n'
            f'}}\n\n'
            f"Ensure the JSON is valid and complete."
        )

        response = self._call_llm(prompt)
        if not response:
            return None

        try:
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                if "evolution_stages" in data:
                    return data
        except json.JSONDecodeError:
            pass

        return None

    # -----------------------------------------------------------------------
    # E-3: compare_schools
    # -----------------------------------------------------------------------

    def compare_schools(self, theory_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple theories/schools across standard dimensions.

        Returns a comparison table and markdown-formatted output.
        """
        if not theory_ids:
            return {"comparison_table": [], "dimensions": [], "markdown": "", "error": "theory_ids is required"}

        theories = []
        for tid in theory_ids:
            t = self.get_theory(tid)
            if t:
                theories.append(t)
            else:
                logger.warning("Theory '%s' not found in knowledge base", tid)

        if not theories:
            return {"comparison_table": [], "dimensions": [], "markdown": "", "error": "No valid theories found"}

        dimensions = [
            "本体论假设",
            "认识论立场",
            "方法论偏好",
            "核心概念",
            "代表学者",
            "经典文献",
            "主要批评",
            "适用场景",
            "局限性",
        ]

        # Build comparison table
        comparison_table = []

        # Dimension: 核心概念
        comparison_table.append({
            "dimension": "核心概念",
            **{t["theory_id"]: "、".join(t.get("key_concepts", [])) for t in theories}
        })

        # Dimension: 代表学者
        comparison_table.append({
            "dimension": "代表学者",
            **{t["theory_id"]: "、".join(t.get("founders", [])) for t in theories}
        })

        # Dimension: 经典文献
        comparison_table.append({
            "dimension": "经典文献",
            **{t["theory_id"]: "；".join(t.get("classic_works", [])) for t in theories}
        })

        # Dimension: 方法论偏好
        comparison_table.append({
            "dimension": "方法论偏好",
            **{t["theory_id"]: t.get("methodological_implications", "") for t in theories}
        })

        # For other dimensions, try LLM or use heuristics
        if self.provider is not None:
            try:
                llm_comparison = self._llm_compare_schools(theories, dimensions)
                if llm_comparison:
                    comparison_table = llm_comparison
            except Exception as exc:
                logger.warning("LLM school comparison failed: %s", exc)

        # If LLM didn't fill all dimensions, add heuristic rows for missing ones
        existing_dims = {row["dimension"] for row in comparison_table}
        for dim in dimensions:
            if dim not in existing_dims:
                row: Dict[str, Any] = {"dimension": dim}
                for t in theories:
                    row[t["theory_id"]] = self._heuristic_dimension(t, dim)
                comparison_table.append(row)

        # Generate markdown table
        markdown = self._comparison_to_markdown(comparison_table, theories)

        return {
            "comparison_table": comparison_table,
            "dimensions": dimensions,
            "markdown": markdown,
        }

    def _heuristic_dimension(self, theory: Dict[str, Any], dimension: str) -> str:
        """Provide heuristic values for dimensions not in the KB."""
        discipline = theory.get("discipline", "")
        name = theory.get("name_en", "")

        if dimension == "本体论假设":
            if discipline == "sociology":
                return "社会现实是建构的，由关系和结构构成"
            if discipline == "psychology":
                return "个体心理过程是理解行为的基础"
            if discipline == "politics":
                return "政治行为受制度和结构约束"
            if discipline == "education":
                return "知识是建构的，学习是主动过程"
            if discipline == "communication":
                return "传播过程塑造社会现实"
            return "社会现实具有多层次结构"

        if dimension == "认识论立场":
            if discipline in ("sociology", "communication"):
                return "解释主义/建构主义"
            if discipline == "psychology":
                return "实证主义与解释主义并存"
            if discipline == "politics":
                return "现实主义/制度主义"
            if discipline == "education":
                return "建构主义认识论"
            return "多元认识论"

        if dimension == "主要批评":
            competing = theory.get("competing_theories", [])
            if competing:
                return f"被{competing[0]}等理论批评为过度简化或忽视其他因素"
            return "缺乏跨文化适用性验证"

        if dimension == "适用场景":
            return theory.get("methodological_implications", "适用于相关领域的实证研究")

        if dimension == "局限性":
            return "可能存在文化偏见和时代局限性"

        return "待补充"

    def _llm_compare_schools(self, theories: List[Dict[str, Any]], dimensions: List[str]) -> Optional[List[Dict[str, Any]]]:
        """Use LLM to generate a rich comparison table."""
        theory_desc = "\n".join(
            f"- {t['theory_id']}: {t.get('name_cn', '')} ({t.get('name_en', '')})\n"
            f"  Discipline: {t.get('discipline', '')}\n"
            f"  Concepts: {', '.join(t.get('key_concepts', []))}\n"
            f"  Founders: {', '.join(t.get('founders', []))}"
            for t in theories
        )

        prompt = (
            f"Compare the following theories across these dimensions:\n"
            f"Dimensions: {', '.join(dimensions)}\n\n"
            f"Theories:\n{theory_desc}\n\n"
            f"Return a JSON array where each element is a row with 'dimension' and one key per theory_id:\n"
            f'[{{"dimension": "本体论假设", "theory_id_1": "...", "theory_id_2": "..."}}]\n\n'
            f"Use the exact theory_ids as keys. Keep each cell concise (under 100 characters)."
        )

        response = self._call_llm(prompt)
        if not response:
            return None

        try:
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                if isinstance(data, list) and data:
                    return data
        except json.JSONDecodeError:
            pass

        return None

    def _comparison_to_markdown(self, comparison_table: List[Dict[str, Any]], theories: List[Dict[str, Any]]) -> str:
        """Convert comparison table to markdown format."""
        if not comparison_table:
            return ""

        headers = ["维度"] + [t.get("name_cn", t["theory_id"]) for t in theories]
        lines = ["| " + " | ".join(headers) + " |"]
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        for row in comparison_table:
            cells = [row.get("dimension", "")]
            for t in theories:
                tid = t["theory_id"]
                val = row.get(tid, "")
                # Escape pipe characters
                val = str(val).replace("|", "\\|")
                cells.append(val)
            lines.append("| " + " | ".join(cells) + " |")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level helpers for tool registration
# ---------------------------------------------------------------------------

def _theory_map(args: Dict[str, Any], mapper: TheoryMapper) -> str:
    topic = args.get("topic", "")
    discipline = args.get("discipline")
    result = mapper.map_theories(topic, discipline)
    return json.dumps(result, ensure_ascii=False)


def _concept_trace(args: Dict[str, Any], mapper: TheoryMapper) -> str:
    concept = args.get("concept", "")
    language = args.get("language", "zh")
    result = mapper.trace_concept(concept, language)
    return json.dumps(result, ensure_ascii=False)


def _compare_schools(args: Dict[str, Any], mapper: TheoryMapper) -> str:
    theory_ids = args.get("theory_ids", [])
    if isinstance(theory_ids, str):
        theory_ids = [t.strip() for t in theory_ids.split(",") if t.strip()]
    result = mapper.compare_schools(theory_ids)
    return json.dumps(result, ensure_ascii=False)
