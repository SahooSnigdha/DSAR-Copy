"""
CodedTool implementation for deterministic PII/PHI detection against the pinned Master Data Classifiers ruleset
plus local Multi-Entity Recognition for unmentioned PII categories.
"""

from logging import Logger, getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import re

import pypdf
from neuro_san.interfaces.coded_tool import CodedTool


class DetectPii(CodedTool):
    """
    CodedTool that executes deterministic PII/PHI detection on documents and filenames
    using the authoritative pinned ruleset (rev-003, version 1.2.0) AND an offline
    local Multi-Entity Recognition engine to catch unmentioned PII categories.
    Runs 100% locally with zero OpenAI token cost.
    """

    EXTENDED_RECOGNIZERS: List[Dict[str, Any]] = [
        {
            "classifier_id": "CLS-EXT-EMPLOYEE-ID-001",
            "category": "PII.IDENTIFIER.EMPLOYEE_ID",
            "pattern": r"(?i:\b(?:Employee|Staff|Worker|Badge)[- ]*(?:ID|No|Number|#|Code)\s*[:=]\s*)([A-Za-z0-9-]+)\b|\bEMP-[A-Za-z0-9-]+\b",
            "confidence": 0.90,
            "priority": 6,
            "description": "Employee ID, worker badge number, or staff identifier."
        },
        {
            "classifier_id": "CLS-EXT-VEHICLE-VIN-001",
            "category": "PII.IDENTIFIER.VEHICLE_VIN",
            "pattern": r"(?i:\b(?:VIN|Vehicle ID|Chassis (?:No|Number))\s*[:=]\s*)([A-HJ-NPR-Z0-9]{17})\b",
            "confidence": 0.92,
            "priority": 6,
            "description": "Vehicle Identification Number (17-character standard)."
        },
        {
            "classifier_id": "CLS-EXT-MAC-ADDRESS-001",
            "category": "PII.LOCATION.MAC_ADDRESS",
            "pattern": r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b",
            "confidence": 0.95,
            "priority": 6,
            "description": "Network hardware MAC physical address."
        },
        {
            "classifier_id": "CLS-EXT-ROUTING-NUM-001",
            "category": "PII.FINANCIAL.ROUTING_NUMBER",
            "pattern": r"(?i:\b(?:Routing|ABA|RTN|Routing Transit Number)\s*[:=]\s*)([0-9]{9})\b",
            "confidence": 0.92,
            "priority": 6,
            "description": "US ABA 9-digit bank routing transit number."
        },
        {
            "classifier_id": "CLS-EXT-BANK-ACCT-001",
            "category": "PII.FINANCIAL.BANK_ACCOUNT_NUMBER",
            "pattern": r"(?i:\b(?:Account[\s\n_-]*(?:Number|No|#)|Acct[\s\n_-]*(?:Number|No|#)|Bank[\s\n_-]*Acct|Direct Deposit Account)\b(?:\s+(?:is|was|:|=|#|-))?\s*[:#=]?\s*)([0-9]{8,17})\b",
            "confidence": 0.90,
            "priority": 6,
            "description": "Financial bank account or direct deposit account number with conversational tolerance."
        },
        {
            "classifier_id": "CLS-EXT-SWIFT-BIC-001",
            "category": "PII.FINANCIAL.SWIFT_BIC",
            "pattern": r"(?i:\b(?:SWIFT|BIC|SWIFT-BIC|BIC Code|SWIFT Code)\s*[:=]\s*)([A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?)\b",
            "confidence": 0.92,
            "priority": 6,
            "description": "International SWIFT / BIC banking identifier."
        },
        {
            "classifier_id": "CLS-EXT-SECRET-KEY-001",
            "category": "PII.SECURITY.API_KEY_SECRET",
            "pattern": r"(?i:\b(?:api[_-]?key|secret|password|auth[_-]?token|bearer|private[_-]?key)\s*[:=]\s*)([A-Za-z0-9_\-\.]{12,64})\b",
            "confidence": 0.95,
            "priority": 6,
            "description": "API keys, bearer authentication tokens, private secrets, and credentials."
        },
        {
            "classifier_id": "CLS-EXT-TAX-ID-001",
            "category": "PII.IDENTIFIER.TAX_ID_ITIN",
            "pattern": r"\b9\d{2}-[78]\d-\d{4}\b",
            "confidence": 0.95,
            "priority": 6,
            "description": "US Individual Taxpayer Identification Number (ITIN)."
        },
        {
            "classifier_id": "CLS-EXT-MRN-001",
            "category": "PHI.MEDICAL_IDENTIFIER.MEDICAL_RECORD_NUMBER",
            "pattern": r"(?i:\b(?:MRN|Medical Record (?:Number|ID))\s*[:=]\s*)([A-Za-z0-9-]{4,15})\b",
            "confidence": 0.92,
            "priority": 6,
            "description": "Hospital Medical Record Number (MRN) or clinical chart ID."
        },
        {
            "classifier_id": "CLS-EXT-NPI-001",
            "category": "PHI.MEDICAL_IDENTIFIER.NPI",
            "pattern": r"(?i:\b(?:NPI|National Provider Identifier)\s*[:=]\s*)([12]\d{9})\b",
            "confidence": 0.92,
            "priority": 6,
            "description": "National Provider Identifier (NPI) 10-digit medical practitioner code."
        },
        {
            "classifier_id": "CLS-EXT-EMERGENCY-CONTACT-001",
            "category": "PII.CONTACT.EMERGENCY_CONTACT",
            "pattern": r"(?i:\b(?:Emergency Contact|Next of Kin|Guardian)\s*[:=]\s*)([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+){1,3})\b",
            "confidence": 0.90,
            "priority": 6,
            "description": "Designated emergency contact, guardian, or next of kin."
        },
        {
            "classifier_id": "CLS-EXT-MAIDEN-NAME-001",
            "category": "PII.DEMOGRAPHIC.MAIDEN_NAME",
            "pattern": r"(?i:\b(?:Mother'?s Maiden Name|Maiden Name)\s*[:=]\s*)([A-Z][a-zA-Z]+)\b",
            "confidence": 0.90,
            "priority": 6,
            "description": "Mother's maiden name or family surname security question."
        },
        {
            "classifier_id": "CLS-EXT-AGENT-NAME-001",
            "category": "PII.PERSON_NAME",
            "pattern": r"(?i:\b(?:My name is|I am|I'm|This is)\s+)([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)?)\b",
            "confidence": 0.90,
            "priority": 5,
            "description": "Customer service representative or speaker introduction name."
        },
        {
            "classifier_id": "CLS-EXT-TITLED-NAME-001",
            "category": "PII.PERSON_NAME",
            "pattern": r"\b(?:Mr\.?|Mrs\.?|Ms\.?|Miss|Dr\.?|Prof\.?)\s+([A-Z][a-zA-Z]+)\b",
            "confidence": 0.90,
            "priority": 5,
            "description": "Honorific title with surname (e.g. Mr. Pendelton)."
        },
        # --- UK Specific Recognizers ---
        {
            "classifier_id": "CLS-EXT-UK-SPOKEN-DOB-001",
            "category": "PII.DEMOGRAPHIC.DATE_OF_BIRTH",
            "pattern": r"(?i:\b(?:born(?:\s+on)?(?:\s+the)?|DOB|Date of Birth)\s*[:=]?\s*)(?:the\s+)?(\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December)[,\s]+\d{4})\b|\b(?i:(?:born(?:\s+on)?|DOB|Date of Birth)\s*[:=]?\s*)(\d{1,2}[/-]\d{1,2}[/-]\d{4})\b",
            "confidence": 0.95,
            "priority": 3,
            "description": "UK and conversational spoken dates of birth (e.g. 14th of July, 1974 or 14/07/1974)."
        },
        {
            "classifier_id": "CLS-UK-NINO-001",
            "category": "PII.IDENTIFIER.UK_NINO",
            "pattern": r"(?i:\b(?:NINO|National Insurance(?:\s*(?:Number|No))?|NI\s*No|NI\s*Number)\s*[:#=]?\s*)([A-Za-z]{2}\s*\d{2}\s*\d{2}\s*\d{2}\s*[A-Da-d]?)\b|\b(?:[A-CEGHJ-PR-TW-Z][A-CEGHJ-NPR-TW-Z]|QQ)\s*\d{2}\s*\d{2}\s*\d{2}\s*[A-D]?\b",
            "confidence": 0.95,
            "priority": 6,
            "description": "UK National Insurance Number (NINO)."
        },
        {
            "classifier_id": "CLS-UK-NHS-NUM-001",
            "category": "PHI.MEDICAL_IDENTIFIER.UK_NHS_NUMBER",
            "pattern": r"(?i:\b(?:NHS|NHS Number|NHS No|National Health Service No)\s*[:#=]?\s*)(\d{3}[\s-]?\d{3}[\s-]?\d{4})\b|\b\d{3}[\s-]\d{3}[\s-]\d{4}\b",
            "confidence": 0.95,
            "priority": 6,
            "description": "UK National Health Service (NHS) 10-digit identifier."
        },
        {
            "classifier_id": "CLS-UK-PHONE-001",
            "category": "PII.CONTACT.PHONE_NUMBER",
            "pattern": r"(?:(?:\+44\s*(?:\(0\)\s*)?|0044\s*)|\b0)(?:1\d{1,4}|2\d|3\d{2}|7\d{3}|8\d{2,3}|9\d{2,3})\s*\d{3,4}\s*\d{3,4}\b",
            "confidence": 0.90,
            "priority": 6,
            "description": "UK landline (+44 1/2/3) and mobile (+44 7) telephone numbers with line-wrap tolerance."
        },
        {
            "classifier_id": "CLS-UK-POSTCODE-001",
            "category": "PII.LOCATION.UK_POSTCODE",
            "pattern": r"\b(?:GIR\s*0AA|[A-PR-UWYZ][A-HK-Y]?\d[A-Z\d]?\s*\d[A-DEFGHJLNPQRSTUWXYZ]{2})\b",
            "confidence": 0.92,
            "priority": 6,
            "description": "UK Royal Mail postcode."
        },
        {
            "classifier_id": "CLS-UK-SORT-CODE-001",
            "category": "PII.FINANCIAL.UK_SORT_CODE",
            "pattern": r"(?i:\b(?:Sort[\s\n_-]*Code|Sort-Code|SC|Sortcode)\b(?:\s+(?:is|was|:|=|#|-))?\s*[:#=]?\s*)([0-9]{2}[-\s]?[0-9]{2}[-\s]?[0-9]{2})\b",
            "confidence": 0.92,
            "priority": 6,
            "description": "UK 6-digit bank sort code with line-wrap and conversational tolerance."
        },
        {
            "classifier_id": "CLS-UK-DRIVING-LICENCE-001",
            "category": "PII.IDENTIFIER.UK_DRIVING_LICENCE",
            "pattern": r"(?i:\b(?:Driving Licen[cs]e|Driver Licen[cs]e|DL)\s*[:#=]?\s*)([A-Z]{5}\s*\d{6}\s*[A-Z]{2}\d[A-Z]{2})\b",
            "confidence": 0.92,
            "priority": 6,
            "description": "UK DVLA 16-character driving licence number."
        },
        {
            "classifier_id": "CLS-UK-VEHICLE-PLATE-001",
            "category": "PII.IDENTIFIER.UK_VEHICLE_PLATE",
            "pattern": r"(?i:\b(?:Registration|Reg Plate|Vehicle Reg|VRM|Number Plate|Reg No|Reg Number)\s*[:#=]?\s*)([A-Z]{2}\d{2}\s?[A-Z]{3})\b",
            "confidence": 0.92,
            "priority": 6,
            "description": "UK standard vehicle registration mark / plate."
        },
        {
            "classifier_id": "CLS-UK-UTR-001",
            "category": "PII.IDENTIFIER.UK_UTR",
            "pattern": r"(?i:\b(?:UTR|Unique Taxpayer Reference|Tax Reference)\s*[:#=]?\s*)(\d{5}\s?\d{5}|\d{10})\b",
            "confidence": 0.92,
            "priority": 6,
            "description": "UK HMRC 10-digit Unique Taxpayer Reference."
        }
    ]

    @staticmethod
    def luhn_checksum(digits: str) -> bool:
        """Validates a sequence of digits using Luhn formula."""
        d_list = [int(d) for d in digits if d.isdigit()]
        if len(d_list) < 13:
            return False
        rev = d_list[::-1]
        total = sum((x * 2 - 9 if x * 2 > 9 else x * 2) if i % 2 == 1 else x for i, x in enumerate(rev))
        return total % 10 == 0

    @classmethod
    def compute_mask(cls, category: str, raw_text: str) -> str:
        """
        Computes category-appropriate mask using solid black squares (■)
        or partial masking policies.
        """
        cat = category.upper()

        if "EMAIL" in cat:
            if "@" in raw_text:
                user, domain = raw_text.split("@", 1)
                return "■" * len(user) + "@" + domain
            return "■" * len(raw_text)
        elif "PHONE" in cat:
            result = []
            prefix_len = 0
            if raw_text.startswith("+1"):
                prefix_len = 2
                result.append("+1")
            elif raw_text.startswith("+44"):
                prefix_len = 3
                result.append("+44")
            elif raw_text.startswith("0044"):
                prefix_len = 4
                result.append("0044")
            elif raw_text.startswith("+"):
                match = re.match(r"^\+\d{1,3}", raw_text)
                if match:
                    prefix_len = len(match.group(0))
                    result.append(match.group(0))
            for char in raw_text[prefix_len:]:
                if char.isdigit():
                    result.append("■")
                else:
                    result.append(char)
            return "".join(result)
        elif "CREDIT_CARD" in cat or "CC" in cat:
            digits_indices = [i for i, c in enumerate(raw_text) if c.isdigit()]
            last_4_indices = set(digits_indices[-4:]) if len(digits_indices) >= 4 else set()
            result = []
            for i, char in enumerate(raw_text):
                if char.isdigit():
                    if i in last_4_indices:
                        result.append(char)
                    else:
                        result.append("■")
                else:
                    result.append(char)
            return "".join(result)
        elif "SSN" in cat:
            digits_seen = 0
            result = []
            for char in raw_text:
                if char.isdigit():
                    digits_seen += 1
                    if digits_seen <= 5:
                        result.append("■")
                    else:
                        result.append(char)
                else:
                    result.append(char)
            return "".join(result)
        elif "ROUTING_NUMBER" in cat or "BANK_ACCOUNT" in cat:
            if len(raw_text) > 4:
                return ("■" * (len(raw_text) - 4)) + raw_text[-4:]
            return "■" * len(raw_text)
        elif "TAX_ID" in cat or "ITIN" in cat:
            if len(raw_text) >= 11:
                return "■■■-■■-" + raw_text[-4:]
            return "".join("■" if c.isdigit() else c for c in raw_text)
        elif "IBAN" in cat:
            total_alnum = sum(1 for c in raw_text if c.isalnum())
            alnum_seen = 0
            result = []
            for char in raw_text:
                if char.isalnum():
                    alnum_seen += 1
                    if alnum_seen <= 4 or alnum_seen > (total_alnum - 2):
                        result.append(char)
                    else:
                        result.append("■")
                else:
                    result.append(char)
            return "".join(result)
        elif "NINO" in cat:
            digit_indices = [i for i, c in enumerate(raw_text) if c.isdigit()]
            last_2_digits = set(digit_indices[-2:]) if len(digit_indices) >= 2 else set()
            last_alpha_idx = None
            for i in range(len(raw_text) - 1, -1, -1):
                if raw_text[i].isalpha():
                    last_alpha_idx = i
                    break
            result = []
            for i, char in enumerate(raw_text):
                if char.isdigit():
                    if i in last_2_digits:
                        result.append(char)
                    else:
                        result.append("■")
                elif char.isalpha():
                    if i == last_alpha_idx and len(digit_indices) >= 2 and i > max(last_2_digits):
                        result.append(char)
                    else:
                        result.append("■")
                else:
                    result.append(char)
            return "".join(result)
        elif "NHS" in cat:
            digit_indices = [i for i, c in enumerate(raw_text) if c.isdigit()]
            last_4_digits = set(digit_indices[-4:]) if len(digit_indices) >= 4 else set()
            result = []
            for i, char in enumerate(raw_text):
                if char.isdigit():
                    if i in last_4_digits:
                        result.append(char)
                    else:
                        result.append("■")
                else:
                    result.append(char)
            return "".join(result)
        elif "POSTCODE" in cat:
            stripped = raw_text.strip()
            if " " in stripped:
                parts = stripped.split()
                return parts[0] + " " + ("■" * len(parts[1]))
            elif len(stripped) > 3:
                return stripped[:-3] + "■■■"
            return "■" * len(raw_text)
        elif "SORT_CODE" in cat:
            digit_indices = [i for i, c in enumerate(raw_text) if c.isdigit()]
            last_2_digits = set(digit_indices[-2:]) if len(digit_indices) >= 2 else set()
            result = []
            for i, char in enumerate(raw_text):
                if char.isdigit():
                    if i in last_2_digits:
                        result.append(char)
                    else:
                        result.append("■")
                else:
                    result.append(char)
            return "".join(result)
        elif "UTR" in cat:
            digit_indices = [i for i, c in enumerate(raw_text) if c.isdigit()]
            last_4_digits = set(digit_indices[-4:]) if len(digit_indices) >= 4 else set()
            result = []
            for i, char in enumerate(raw_text):
                if char.isdigit():
                    if i in last_4_digits:
                        result.append(char)
                    else:
                        result.append("■")
                else:
                    result.append(char)
            return "".join(result)
        elif "DRIVING_LICENCE" in cat:
            alnum_indices = [i for i, c in enumerate(raw_text) if c.isalnum()]
            last_2_alnum = set(alnum_indices[-2:]) if len(alnum_indices) >= 2 else set()
            result = []
            for i, char in enumerate(raw_text):
                if char.isalnum():
                    if i in last_2_alnum:
                        result.append(char)
                    else:
                        result.append("■")
                else:
                    result.append(char)
            return "".join(result)
        elif "VEHICLE_PLATE" in cat:
            stripped = raw_text.strip()
            if " " in stripped:
                parts = stripped.split()
                return parts[0] + " " + ("■" * len(parts[1]))
            elif len(stripped) > 3:
                return stripped[:-3] + "■■■"
            return "■" * len(raw_text)
        elif "STREET_ADDRESS" in cat or "ADDRESS" in cat:
            m = re.match(r"^(.*?)(,\s*[A-Za-z\s]+,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?)$", raw_text)
            if m:
                street_part, city_part = m.group(1), m.group(2)
                masked_street = "".join("■" if c.isalnum() else c for c in street_part)
                return masked_street + city_part
            else:
                return "".join("■" if c.isalnum() else c for c in raw_text)
        elif "INSURANCE" in cat:
            m = re.match(r"^([A-Za-z]+-)([A-Za-z0-9]+)(-[A-Za-z0-9]+)$", raw_text)
            if m:
                prefix, mid, suffix = m.group(1), m.group(2), m.group(3)
                return prefix + ("■" * len(mid)) + suffix
            m2 = re.match(r"^([A-Za-z]+-)([A-Za-z0-9]+)$", raw_text)
            if m2:
                prefix, mid = m2.group(1), m2.group(2)
                return prefix + ("■" * len(mid))
            return "".join("■" if c.isalnum() else c for c in raw_text)
        elif "IP" in cat or "MAC" in cat:
            return "".join("■" if c.isalnum() else c for c in raw_text)
        else:
            return "".join("■" if not c.isspace() else c for c in raw_text)

    async def async_invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> Dict[str, Any]:
        logger: Logger = getLogger(self.__class__.__name__)
        file_path_str: Optional[str] = args.get("file_path")
        content: Optional[str] = args.get("content")
        file_name: Optional[str] = args.get("file_name")
        ruleset_path_str: Optional[str] = args.get("ruleset_path", "master_data_classifiers/ruleset.json")
        enable_extended_detection: bool = args.get("enable_extended_detection", True)

        if file_path_str:
            p = Path(file_path_str).resolve()
            if not file_name:
                file_name = p.name
            if not content:
                if not p.exists():
                    raise ValueError(f"path_not_found: File '{p}' does not exist.")
                if p.suffix.lower() == ".pdf":
                    reader = pypdf.PdfReader(str(p))
                    content = "\n\n".join(page.extract_text() or "" for page in reader.pages)
                else:
                    content = p.read_text(encoding="utf-8", errors="replace")
        elif not content:
            raise ValueError("invalid_input: Either 'file_path' or 'content' must be provided.")

        if not file_name:
            file_name = "document.txt"

        r_path = Path(ruleset_path_str)
        if not r_path.exists():
            candidates = [
                Path("master_data_classifiers/ruleset.json"),
                Path("master_data_classifiers/ruleset.yaml"),
                Path("ruleset.json"),
            ]
            for c in candidates:
                if c.exists():
                    r_path = c
                    break
            if not r_path.exists():
                raise ValueError(f"ruleset_not_found: Authoritative ruleset not found at {ruleset_path_str}")

        with open(r_path, "r", encoding="utf-8") as f:
            if r_path.suffix == ".json":
                ruleset = json.load(f)
            else:
                import yaml
                ruleset = yaml.safe_load(f)

        active_rev = ruleset.get("active_revision_id", "rev-003")
        ruleset_ver = ruleset.get("ruleset_version", "1.2.0")
        classifiers = ruleset.get("classifiers", [])

        candidates: List[Dict[str, Any]] = []

        for cls in classifiers:
            cls_id = cls.get("classifier_id", "")
            cat = cls.get("category", "")
            priority = cls.get("priority", 3)
            conf_thresh = cls.get("confidence_threshold", 0.85)
            cls_type = cls.get("type", "regex")

            if cls_type in ("regex", "regex_context"):
                pattern = cls["pattern"]
                for m in re.finditer(pattern, content, flags=re.MULTILINE):
                    if m.lastindex and m.group(1):
                        matched = m.group(1)
                        start = m.start(1)
                        end = m.end(1)
                    else:
                        matched = m.group(0)
                        start = m.start(0)
                        end = m.end(0)

                    conf = 0.95
                    if cls.get("validator") == "luhn_soft_fail":
                        digits = [c for c in matched if c.isdigit()]
                        conf = 0.99 if self.luhn_checksum("".join(digits)) else 0.86
                    elif cls_id == "CLS-PII-DOB-001":
                        prefix = content[max(0, start - 30):start].lower()
                        if "birth" not in prefix and "dob" not in prefix:
                            continue
                    elif cls_id == "CLS-PII-PHONE-001":
                        prefix = content[max(0, start - 40):start].lower()
                        if any(term in prefix for term in ("npi", "nhs", "health service", "utr", "taxpayer", "tax reference", "sort code", "sc:")):
                            continue

                    if conf >= conf_thresh:
                        candidates.append({
                            "classifier_id": cls_id,
                            "category": cat,
                            "start_offset": start,
                            "end_offset": end,
                            "matched_text": matched,
                            "confidence": conf,
                            "priority": priority,
                            "source": "pinned_ruleset",
                            "location": "content"
                        })

            elif cls_type == "dictionary_and_heuristic":
                terms = cls.get("terms", [])
                for term in terms:
                    pattern = r"\b" + re.escape(term) + r"\b"
                    for m in re.finditer(pattern, content, flags=re.IGNORECASE):
                        candidates.append({
                            "classifier_id": cls_id,
                            "category": cat,
                            "start_offset": m.start(),
                            "end_offset": m.end(),
                            "matched_text": m.group(0),
                            "confidence": 0.9,
                            "priority": priority,
                            "source": "pinned_ruleset",
                            "location": "content"
                        })

            elif cls_type == "ner":
                patterns = [
                    r"(?i:\b(?:Full Name|Name|Patient Name|Individual):[ \t]*)([A-Z][a-zA-Z]+(?:[ \t]+[A-Z][a-zA-Z]+){1,3})"
                ]
                for pat in patterns:
                    for m in re.finditer(pat, content):
                        matched = m.group(1)
                        candidates.append({
                            "classifier_id": cls_id,
                            "category": cat,
                            "start_offset": m.start(1),
                            "end_offset": m.end(1),
                            "matched_text": matched,
                            "confidence": 0.95,
                            "priority": priority,
                            "source": "pinned_ruleset",
                            "location": "content"
                        })

        if enable_extended_detection:
            for rec in self.EXTENDED_RECOGNIZERS:
                pattern = rec["pattern"]
                for m in re.finditer(pattern, content, flags=re.MULTILINE):
                    if m.lastindex and m.group(1):
                        val = m.group(1)
                        st = m.start(1)
                        ed = m.end(1)
                    else:
                        val = m.group(0)
                        st = m.start(0)
                        ed = m.end(0)

                    candidates.append({
                        "classifier_id": rec["classifier_id"],
                        "category": rec["category"],
                        "start_offset": st,
                        "end_offset": ed,
                        "matched_text": val,
                        "confidence": rec["confidence"],
                        "priority": rec["priority"],
                        "source": "extended_multi_entity",
                        "location": "content"
                    })

        discovered_names = {
            c["matched_text"] for c in candidates
            if c["category"] in ("PII.PERSON_NAME", "PII.CONTACT.EMERGENCY_CONTACT", "PII.DEMOGRAPHIC.MAIDEN_NAME")
        }
        tokens_to_propagate = set()
        for dname in discovered_names:
            tokens_to_propagate.add(dname)
            parts = dname.split()
            if len(parts) >= 2:
                for title in ("Mr", "Mr.", "Mrs", "Mrs.", "Ms", "Ms.", "Miss", "Dr", "Dr."):
                    tokens_to_propagate.add(f"{title} {parts[-1]}")
                if len(parts[0]) >= 3 and parts[0][0].isupper():
                    tokens_to_propagate.add(parts[0])
            elif len(parts) == 1 and len(parts[0]) >= 3 and parts[0][0].isupper():
                tokens_to_propagate.add(parts[0])

        for token in tokens_to_propagate:
            dpat = r"\b" + re.escape(token) + r"\b"
            for m in re.finditer(dpat, content):
                st, ed = m.start(), m.end()
                if not any(c["start_offset"] == st and c["end_offset"] == ed for c in candidates):
                    candidates.append({
                        "classifier_id": "CLS-PII-NAME-PROPAGATED",
                        "category": "PII.PERSON_NAME",
                        "start_offset": st,
                        "end_offset": ed,
                        "matched_text": m.group(0),
                        "confidence": 0.95,
                        "priority": 5,
                        "source": "name_propagation",
                        "location": "content"
                    })

        candidates.sort(key=lambda x: (
            x["start_offset"],
            -(x["end_offset"] - x["start_offset"]),
            x["priority"],
            -x["confidence"]
        ))
        resolved_spans: List[Dict[str, Any]] = []
        last_end = -1
        for c in candidates:
            if c["start_offset"] >= last_end:
                c["proposed_mask"] = self.compute_mask(c["category"], c["matched_text"])
                snippet_start = max(0, c["start_offset"] - 25)
                snippet_end = min(len(content), c["end_offset"] + 25)
                c["context"] = content[snippet_start:snippet_end].replace("\n", " ")
                resolved_spans.append(c)
                last_end = c["end_offset"]

        for idx, span in enumerate(resolved_spans, 1):
            span["field_id"] = idx

        filename_detections: List[Dict[str, Any]] = []
        sensitive_name_tokens = list(dict.fromkeys(
            s["matched_text"] for s in resolved_spans
            if s["category"] in ("PII.PERSON_NAME", "PII.IDENTIFIER.USERNAME", "PII.CONTACT.EMERGENCY_CONTACT", "PII.DEMOGRAPHIC.MAIDEN_NAME")
        ))
        for name in sensitive_name_tokens:
            parts = name.split()
            variants = [name]
            if len(parts) >= 2:
                variants.extend([
                    f"{parts[0]}_{parts[-1]}",
                    f"{parts[0]}_{parts[-1][0]}",
                    f"{parts[0]} {parts[-1][0]}",
                    f"{parts[0][0]}_{parts[-1]}",
                    f"{parts[0][0]} {parts[-1]}",
                ])
                if len(parts[0]) >= 3:
                    variants.append(parts[0])
                if len(parts[-1]) >= 3:
                    variants.append(parts[-1])
            elif len(parts) == 1 and len(parts[0]) >= 3:
                variants.append(parts[0])

            variants.sort(key=len, reverse=True)
            for var in variants:
                if var in file_name and not any(fd["matched_text"] == var or var in fd["matched_text"] for fd in filename_detections):
                    start_pos = file_name.find(var)
                    end_pos = start_pos + len(var)
                    x_mask = "".join("X" if c.isalnum() else c for c in var)
                    proposed_fn = file_name[:start_pos] + x_mask + file_name[end_pos:]
                    fn_stem = Path(proposed_fn).stem
                    fn_ext = Path(proposed_fn).suffix
                    proposed_redacted_fn = f"{fn_stem}_redacted{fn_ext}"
                    filename_detections.append({
                        "field_id": len(resolved_spans) + len(filename_detections) + 1,
                        "classifier_id": "CLS-PII-FILENAME-001",
                        "category": "PII.PERSON_NAME (File Name)",
                        "matched_text": var,
                        "start_offset": start_pos,
                        "end_offset": end_pos,
                        "location": "file_name",
                        "confidence": 0.95,
                        "source": "filename_scanner",
                        "replacement": x_mask,
                        "proposed_mask": x_mask,
                        "proposed_filename": proposed_redacted_fn,
                        "context": f"File Name: {file_name}"
                    })
                    break

        all_detections = resolved_spans + filename_detections
        return {
            "status": "SUCCESS",
            "active_revision_id": active_rev,
            "ruleset_version": ruleset_ver,
            "file_name": file_name,
            "total_detections": len(all_detections),
            "content_detections_count": len(resolved_spans),
            "filename_detections_count": len(filename_detections),
            "pinned_detections_count": sum(1 for s in resolved_spans if s.get("source") == "pinned_ruleset"),
            "extended_detections_count": sum(1 for s in resolved_spans if s.get("source") == "extended_multi_entity"),
            "detections": all_detections,
            "approved_spans_template": all_detections
        }

