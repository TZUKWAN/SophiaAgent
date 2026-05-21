"""Research ethics and IRB support (Phase I).

I-1: Ethics review checklist
I-2: Informed consent generator
I-3: Research risk level assessment
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# I-1: Ethics Review Checklist
# ---------------------------------------------------------------------------

_ETHICS_DIMENSIONS = [
    {
        "id": "informed_consent",
        "title": "知情同意",
        "description": "参与者是否在充分知情的情况下自愿同意参与研究。",
        "items": [
            {"id": "ic_1", "text": "是否向参与者说明了研究目的和流程？", "weight": 1},
            {"id": "ic_2", "text": "是否告知参与者参与是自愿的，可随时退出？", "weight": 1},
            {"id": "ic_3", "text": "是否告知参与者拒绝参与不会带来不利后果？", "weight": 1},
            {"id": "ic_4", "text": "是否说明了数据的使用范围和保密措施？", "weight": 1},
            {"id": "ic_5", "text": "是否获得了参与者（或监护人）的书面或口头同意？", "weight": 1},
        ],
    },
    {
        "id": "privacy_confidentiality",
        "title": "隐私与保密",
        "description": "参与者个人信息和数据的保护。",
        "items": [
            {"id": "pc_1", "text": "是否对参与者的身份信息进行匿名化或去标识化处理？", "weight": 1},
            {"id": "pc_2", "text": "数据存储是否采取了安全措施（加密、访问控制）？", "weight": 1},
            {"id": "pc_3", "text": "是否限制了数据的访问权限（仅研究团队可见）？", "weight": 1},
            {"id": "pc_4", "text": "是否说明了数据保存期限和销毁方式？", "weight": 1},
        ],
    },
    {
        "id": "risk_benefit",
        "title": "风险与受益",
        "description": "研究对参与者的潜在风险与受益的权衡。",
        "items": [
            {"id": "rb_1", "text": "是否识别并评估了所有潜在的身体/心理/社会风险？", "weight": 1},
            {"id": "rb_2", "text": "风险是否已最小化至合理且必要的程度？", "weight": 1},
            {"id": "rb_3", "text": "研究的预期受益是否大于潜在风险？", "weight": 1},
            {"id": "rb_4", "text": "是否为参与者提供了必要的支持或补救措施？", "weight": 1},
        ],
    },
    {
        "id": "vulnerable_populations",
        "title": "弱势群体保护",
        "description": "对未成年人、老年人、残障人士、囚犯等特殊群体的额外保护。",
        "items": [
            {"id": "vp_1", "text": "研究是否涉及弱势群体？", "weight": 1},
            {"id": "vp_2", "text": "如涉及弱势群体，是否获得了监护人或法定代表的同意？", "weight": 1},
            {"id": "vp_3", "text": "是否采取了额外的保护措施（如简化语言、辅助沟通）？", "weight": 1},
            {"id": "vp_4", "text": "是否评估了该群体参与研究的特殊风险？", "weight": 1},
        ],
    },
    {
        "id": "data_integrity",
        "title": "数据完整性与透明度",
        "description": "研究数据的真实性和结果报告的透明度。",
        "items": [
            {"id": "di_1", "text": "是否有数据造假、篡改或选择性报告的风险？", "weight": 1},
            {"id": "di_2", "text": "是否计划公开研究方案或预注册假设？", "weight": 0.5},
            {"id": "di_3", "text": "是否计划共享去标识化数据（如适用）？", "weight": 0.5},
        ],
    },
    {
        "id": "researcher_conduct",
        "title": "研究者行为",
        "description": "研究者在研究过程中的职业操守。",
        "items": [
            {"id": "rc_1", "text": "研究者是否存在利益冲突（如与研究资助方有关系）？", "weight": 1},
            {"id": "rc_2", "text": "是否声明了利益冲突并制定了管理计划？", "weight": 1},
            {"id": "rc_3", "text": "研究者是否具备开展该研究的资质和能力？", "weight": 0.5},
        ],
    },
]


# ---------------------------------------------------------------------------
# I-2: Consent Templates
# ---------------------------------------------------------------------------

_CONSENT_TEMPLATES = {
    "adult_survey": {
        "title": "调查研究知情同意书",
        "sections": [
            {
                "heading": "研究目的",
                "content": "本研究旨在[研究目的]。您的参与将帮助我们[研究意义]。",
            },
            {
                "heading": "参与方式",
                "content": "您需要完成一份约[预计时间]分钟的问卷/访谈。",
            },
            {
                "heading": "风险与不适",
                "content": "本研究无已知风险。部分问题可能涉及个人感受，您有权选择不回答。",
            },
            {
                "heading": "受益",
                "content": "您不会直接从本研究获得个人受益，但您的贡献将推动[学科领域]的发展。",
            },
            {
                "heading": "保密性",
                "content": "您的回答将严格保密。所有数据将以匿名或编码形式存储和分析。研究报告不会包含任何可识别您身份的信息。",
            },
            {
                "heading": "自愿参与与退出",
                "content": "您的参与完全自愿。您有权在任何时间、无需说明理由地退出研究，且不会因此遭受任何不利后果。",
            },
            {
                "heading": "联系方式",
                "content": "如您对研究有任何疑问，请联系：\n研究者：[姓名] [邮箱]\n伦理委员会：[联系方式]",
            },
            {
                "heading": "同意声明",
                "content": "我已阅读并理解上述信息，同意自愿参与本研究。",
            },
        ],
    },
    "adult_interview": {
        "title": "访谈研究知情同意书",
        "sections": [
            {
                "heading": "研究目的",
                "content": "本研究通过深度访谈了解[研究主题]。",
            },
            {
                "heading": "参与方式",
                "content": "您将参加一次约[预计时间]分钟的半结构化访谈，访谈将被录音以便后续转录分析。",
            },
            {
                "heading": "录音同意",
                "content": "访谈过程将被录音。如您不同意录音，研究者将以笔记方式记录。",
            },
            {
                "heading": "风险与不适",
                "content": "访谈可能涉及个人经历或敏感话题。如感到不适，您可随时要求暂停或终止。",
            },
            {
                "heading": "保密性",
                "content": "您的身份将以化名形式出现在研究报告中。录音和转录稿将安全保存，研究结束后[保存期限]予以销毁。",
            },
            {
                "heading": "自愿参与与退出",
                "content": "参与完全自愿，可随时退出，无需说明理由。",
            },
            {
                "heading": "同意声明",
                "content": "我已阅读并理解上述信息，同意参与访谈并[同意/不同意]录音。",
            },
        ],
    },
    "minor_survey": {
        "title": "未成年人研究知情同意书（监护人版）",
        "sections": [
            {
                "heading": "研究目的",
                "content": "本研究旨在[研究目的]，研究对象为[年龄范围]的未成年人。",
            },
            {
                "heading": "参与方式",
                "content": "您的孩子将参与[活动描述]，预计耗时[预计时间]。",
            },
            {
                "heading": "风险与受益",
                "content": "风险：[具体风险或'无明显风险']。\n受益：[具体受益或'无直接受益']。",
            },
            {
                "heading": "保密性",
                "content": "您孩子的信息将严格保密，以编码形式处理，不会披露可识别身份的信息。",
            },
            {
                "heading": "监护人同意",
                "content": "作为法定监护人，我同意我的孩子参与本研究。",
            },
            {
                "heading": "未成年人知情",
                "content": "同时，研究者将向孩子口头说明研究内容，并征求其口头同意（如孩子表示不愿意，将尊重其意愿）。",
            },
        ],
    },
    "experiment": {
        "title": "实验研究知情同意书",
        "sections": [
            {
                "heading": "研究目的",
                "content": "本研究旨在通过实验方法检验[研究假设]。",
            },
            {
                "heading": "实验流程",
                "content": "您将被随机分配到[实验组/对照组]，完成[具体任务]。整个过程约[预计时间]分钟。",
            },
            {
                "heading": "潜在风险",
                "content": "可能的风险包括：[具体风险或'无明显风险']。如出现不适，请立即告知研究者。",
            },
            {
                "heading": "补偿",
                "content": "为感谢您的参与，您将获得[补偿方式，如课程学分/小额报酬]。",
            },
            {
                "heading": "保密性",
                "content": "您的数据将匿名化处理，仅用于学术研究。",
            },
            {
                "heading": "自愿参与与退出",
                "content": "参与完全自愿，可随时退出且不影响已获得的补偿。",
            },
            {
                "heading": "同意声明",
                "content": "我已阅读并理解上述信息，同意自愿参与本实验。",
            },
        ],
    },
}


class EthicsSupportEngine:
    """Research ethics review, consent generation, and risk assessment."""

    # ------------------------------------------------------------------
    # I-1: Ethics Review Checklist
    # ------------------------------------------------------------------

    def ethics_checklist(self, args: dict) -> Dict[str, Any]:
        """Generate an ethics review checklist with scoring.

        Args:
            study_type: str — "survey" | "interview" | "experiment" | "observation" | "secondary_data"
            involves_vulnerable: bool
            involves_deception: bool
            involves_sensitive_topics: bool
            data_linkable: bool
            cross_border: bool
            has_funding_conflict: bool

        Returns:
            Full checklist with dimension scores, overall risk, and recommendations.
        """
        study_type = args.get("study_type", "survey")
        involves_vulnerable = args.get("involves_vulnerable", False)
        involves_deception = args.get("involves_deception", False)
        involves_sensitive = args.get("involves_sensitive_topics", False)
        data_linkable = args.get("data_linkable", False)
        cross_border = args.get("cross_border", False)
        has_conflict = args.get("has_funding_conflict", False)

        results = []
        total_score = 0
        total_max = 0
        flagged_items = []

        for dim in _ETHICS_DIMENSIONS:
            dim_score = 0
            dim_max = 0
            dim_items = []

            for item in dim["items"]:
                # Auto-flag certain items based on study characteristics
                auto_flag = False
                if dim["id"] == "vulnerable_populations":
                    if item["id"] == "vp_1" and not involves_vulnerable:
                        auto_flag = True  # No vulnerable, so this is N/A but we skip it
                        continue
                    if not involves_vulnerable:
                        continue

                weight = item["weight"]
                dim_max += weight
                total_max += weight

                # For checklist, items are "pending" by default
                status = "pending"
                note = ""

                if dim["id"] == "risk_benefit" and involves_deception and item["id"] == "rb_1":
                    note = "研究涉及欺骗，须额外评估欺骗的必要性和事后解释(debriefing)计划"
                    status = "attention"

                if dim["id"] == "privacy_confidentiality" and data_linkable:
                    if item["id"] in ("pc_1", "pc_2"):
                        note = "数据可关联到个人身份，需加强去标识化和安全措施"
                        status = "attention"

                if dim["id"] == "researcher_conduct" and has_conflict:
                    if item["id"] == "rc_2":
                        status = "attention"
                        note = "存在利益冲突，必须声明并制定管理计划"

                dim_items.append({
                    "id": item["id"],
                    "text": item["text"],
                    "weight": weight,
                    "status": status,
                    "note": note,
                })

            results.append({
                "dimension_id": dim["id"],
                "title": dim["title"],
                "description": dim["description"],
                "score": dim_score,
                "max_score": dim_max,
                "items": dim_items,
            })

        # Risk level assessment
        risk_level = self._assess_risk_level(
            study_type, involves_vulnerable, involves_deception,
            involves_sensitive, data_linkable, cross_border
        )

        # Recommendations
        recommendations = self._ethics_recommendations(
            study_type, involves_vulnerable, involves_deception,
            involves_sensitive, data_linkable, cross_border, has_conflict
        )

        return {
            "study_type": study_type,
            "dimensions": results,
            "risk_level": risk_level,
            "recommendations": recommendations,
            "irb_required": risk_level in ("moderate", "high"),
            "note": "本清单为自评工具，正式伦理审查须提交机构伦理委员会(IRB/Ethics Committee)。",
        }

    # ------------------------------------------------------------------
    # I-2: Informed Consent Generator
    # ------------------------------------------------------------------

    def generate_consent(self, args: dict) -> Dict[str, Any]:
        """Generate an informed consent form.

        Args:
            template_type: str — "adult_survey" | "adult_interview" | "minor_survey" | "experiment"
            study_title: str
            researcher_name: str
            researcher_contact: str
            estimated_duration: str
            study_purpose: str
            compensation: str (optional)
            risks: str (optional)
            custom_sections: list of dict (optional)
        """
        template_type = args.get("template_type", "adult_survey")
        template = _CONSENT_TEMPLATES.get(template_type)
        if not template:
            return {
                "error": f"Unknown template type '{template_type}'",
                "available": list(_CONSENT_TEMPLATES.keys()),
            }

        sections = []
        for sec in template["sections"]:
            content = sec["content"]
            # Fill in placeholders
            content = content.replace("[研究目的]", args.get("study_purpose", "（请填写研究目的）"))
            content = content.replace("[研究意义]", args.get("study_significance", "（请填写研究意义）"))
            content = content.replace("[预计时间]", str(args.get("estimated_duration", "（请填写）")))
            content = content.replace("[学科领域]", args.get("discipline", "相关领域"))
            content = content.replace("[年龄范围]", args.get("age_range", "（请填写）"))
            content = content.replace("[活动描述]", args.get("activity_description", "（请填写）"))
            content = content.replace("[具体风险或'无明显风险']", args.get("risks", "无明显风险"))
            content = content.replace("[具体受益或'无直接受益']", args.get("benefits", "无直接受益"))
            content = content.replace("[保存期限]", args.get("retention_period", "按照机构规定"))
            content = content.replace("[姓名]", args.get("researcher_name", "（研究者姓名）"))
            content = content.replace("[邮箱]", args.get("researcher_contact", "（联系方式）"))
            content = content.replace("[补偿方式，如课程学分/小额报酬]", args.get("compensation", "（请填写补偿方式）"))
            content = content.replace("[同意/不同意]", args.get("recording_consent", "同意") if "recording" in template_type else "")
            sections.append({"heading": sec["heading"], "content": content})

        # Add custom sections
        for custom in args.get("custom_sections", []):
            sections.append({
                "heading": custom.get("heading", ""),
                "content": custom.get("content", ""),
            })

        return {
            "title": args.get("study_title", template["title"]),
            "template_type": template_type,
            "sections": sections,
            "signature_required": True,
            "date_required": True,
        }

    def list_consent_templates(self) -> List[Dict[str, str]]:
        return [
            {"type": k, "title": v["title"]}
            for k, v in _CONSENT_TEMPLATES.items()
        ]

    # ------------------------------------------------------------------
    # I-3: Risk Level Assessment
    # ------------------------------------------------------------------

    def assess_risk(self, args: dict) -> Dict[str, Any]:
        """Assess research risk level.

        Args:
            study_type: str
            involves_vulnerable: bool
            involves_deception: bool
            involves_sensitive_topics: bool
            physical_intervention: bool
            data_linkable: bool
            cross_border: bool
            data_sharing: bool
        """
        study_type = args.get("study_type", "survey")
        involves_vulnerable = args.get("involves_vulnerable", False)
        involves_deception = args.get("involves_deception", False)
        involves_sensitive = args.get("involves_sensitive_topics", False)
        physical = args.get("physical_intervention", False)
        data_linkable = args.get("data_linkable", False)
        cross_border = args.get("cross_border", False)
        data_sharing = args.get("data_sharing", False)

        risk_factors = []
        risk_score = 0

        if involves_vulnerable:
            risk_score += 3
            risk_factors.append("涉及弱势群体")
        if involves_deception:
            risk_score += 2
            risk_factors.append("涉及欺骗")
        if involves_sensitive:
            risk_score += 2
            risk_factors.append("涉及敏感话题")
        if physical:
            risk_score += 3
            risk_factors.append("涉及身体干预")
        if data_linkable:
            risk_score += 1
            risk_factors.append("数据可关联个人身份")
        if cross_border:
            risk_score += 1
            risk_factors.append("跨境数据传输")
        if data_sharing:
            risk_score += 1
            risk_factors.append("数据共享/公开")

        # Study type base risk
        base_risk = {
            "secondary_data": 0,
            "survey": 1,
            "interview": 1,
            "observation": 2,
            "experiment": 2,
        }
        risk_score += base_risk.get(study_type, 1)

        if risk_score <= 2:
            level = "minimal"
            description = "最低风险：风险不超过日常生活或常规检查中的风险。"
            irb_track = "豁免审查(Exempt Review)"
        elif risk_score <= 5:
            level = "low"
            description = "低风险：存在轻微不适或不便，但无持久影响。"
            irb_track = "快速审查(Expedited Review)"
        elif risk_score <= 8:
            level = "moderate"
            description = "中等风险：可能产生心理压力或社会风险，需要额外保护措施。"
            irb_track = "全面审查(Full Review)"
        else:
            level = "high"
            description = "高风险：可能造成显著伤害，须严格论证必要性和保护方案。"
            irb_track = "全面审查(Full Review)"

        return {
            "risk_score": risk_score,
            "risk_level": level,
            "description": description,
            "irb_review_track": irb_track,
            "risk_factors": risk_factors,
            "mitigation_suggestions": self._risk_mitigations(level, risk_factors),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _assess_risk_level(
        study_type: str, vulnerable: bool, deception: bool,
        sensitive: bool, linkable: bool, cross_border: bool
    ) -> str:
        score = 0
        if vulnerable:
            score += 3
        if deception:
            score += 2
        if sensitive:
            score += 1
        if linkable:
            score += 1
        if cross_border:
            score += 1
        base = {"secondary_data": 0, "survey": 1, "interview": 1, "observation": 2, "experiment": 2}
        score += base.get(study_type, 1)

        if score <= 2:
            return "minimal"
        elif score <= 5:
            return "low"
        elif score <= 8:
            return "moderate"
        else:
            return "high"

    @staticmethod
    def _ethics_recommendations(
        study_type: str, vulnerable: bool, deception: bool,
        sensitive: bool, linkable: bool, cross_border: bool, conflict: bool
    ) -> List[str]:
        recs = []
        if vulnerable:
            recs.append("涉及弱势群体：须获得监护人书面同意，并确保孩子的口头同意被尊重")
        if deception:
            recs.append("涉及欺骗：须在研究结束后及时进行debriefing（事后解释），说明真实目的")
        if sensitive:
            recs.append("涉及敏感话题：提供心理支持资源联系方式，访谈中注意创伤知情(truma-informed)原则")
        if linkable:
            recs.append("数据可识别：采用强去标识化措施，限制数据访问权限")
        if cross_border:
            recs.append("跨境数据：遵守数据出境相关法律法规（如中国《数据安全法》《个人信息保护法》）")
        if conflict:
            recs.append("利益冲突：在研究报告中声明利益冲突")
        if study_type in ("experiment", "observation"):
            recs.append("实验/观察研究：制定应急预案，确保参与者可随时退出")
        if not recs:
            recs.append("本研究伦理风险较低，但仍须遵守基本的知情同意和保密原则")
        return recs

    @staticmethod
    def _risk_mitigations(level: str, factors: List[str]) -> List[str]:
        mitigations = []
        if "涉及弱势群体" in factors:
            mitigations.append("为弱势群体参与者提供额外保护，简化知情同意语言")
        if "涉及欺骗" in factors:
            mitigations.append("研究结束后48小时内完成debriefing，提供退出数据删除选项")
        if "涉及敏感话题" in factors:
            mitigations.append("提供心理咨询热线，访谈前评估参与者心理状态")
        if "涉及身体干预" in factors:
            mitigations.append("配备急救措施，由专业人员执行干预")
        if "数据可关联个人身份" in factors:
            mitigations.append("采用k-匿名化或差分隐私技术")
        if "跨境数据传输" in factors:
            mitigations.append("签署标准合同条款(SCC)，确保接收方数据保护水平相当")
        if level in ("moderate", "high"):
            mitigations.append("建议进行正式伦理审查，提交完整的IRB申请材料")
        return mitigations if mitigations else ["遵循标准研究伦理规范即可"]
