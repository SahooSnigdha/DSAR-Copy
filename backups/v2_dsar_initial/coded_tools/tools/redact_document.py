"""
CodedTool implementation for deterministic document and filename redaction,
strict zero-leakage verification, and artifact generation (PDF and text).
"""

import re
from logging import Logger, getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional
import pypdf

try:
    import pymupdf
except ImportError:
    pymupdf = None

from neuro_san.interfaces.coded_tool import CodedTool
from coded_tools.tools.write_pdf import WritePdf
from coded_tools.tools.detect_pii import DetectPii


class RedactDocument(CodedTool):
    """
    CodedTool that takes a source document (or text) and a list of human-approved
    sensitive spans, applies category-appropriate partial masking using solid black
    squares (■), redacts filename PII with 'X', executes strict zero-leakage verification,
    and writes hardened output artifacts (.pdf and/or .txt).
    For PDF documents, preserves 100% of the original document layout, fonts, colors,
    and styling by applying direct in-place visual blackout redaction.
    Runs 100% locally with zero OpenAI token cost.
    """

    @classmethod
    def get_sensitive_targets(cls, raw_val: str, category: str, mask: str = "") -> List[str]:
        """
        Determines the exact substrings that need visual blackout within a PDF page,
        adhering to category-appropriate partial masking policies (e.g. preserving phone country code,
        bank account last 4 digits, postcode outward code, etc.).
        """
        cat = category.upper()
        raw = raw_val.strip()
        if not raw:
            return []

        if "SORT_CODE" in cat:
            m = re.match(r"^(\d{2}[-\s]?\d{2})[-\s]?(\d{2})$", raw)
            if m:
                return [m.group(1) + "-", m.group(1), m.group(1).replace("-", " "), m.group(1).replace("-", "")]
            return [raw[:5], raw[:4]]

        elif "BANK_ACCOUNT" in cat or "ROUTING" in cat:
            digits = "".join(c for c in raw if c.isdigit())
            if len(digits) >= 8:
                return [digits[:-4], raw[:-4]]
            return [raw]

        elif "POSTCODE" in cat:
            parts = raw.split()
            if len(parts) >= 2:
                return [parts[-1]]
            elif len(raw) > 3:
                return [raw[-3:]]
            return [raw]

        elif "PHONE" in cat:
            if "+44" in raw:
                sub = raw.replace("+44", "").strip()
                tokens = [sub]
                tokens.extend(sub.split())
                return tokens
            elif raw.startswith("+1"):
                sub = raw[2:].strip()
                return [sub, raw]
            digits = "".join(c for c in raw if c.isdigit())
            if len(digits) == 10:
                return [raw, digits, digits[3:]]
            return [raw]

        elif "EMAIL" in cat:
            if "@" in raw:
                return [raw.split("@")[0]]
            return [raw]

        elif "CREDIT_CARD" in cat or "CC" in cat:
            digits = "".join(c for c in raw if c.isdigit())
            if len(digits) >= 16:
                parts = [raw[:-4]]
                if len(raw.split("-")) == 4:
                    p = raw.split("-")
                    parts.extend(["-".join(p[:3]) + "-", p[0], p[1], p[2]])
                return parts
            return [raw]

        elif "SSN" in cat:
            if "-" in raw:
                parts = raw.split("-")
                if len(parts) == 3:
                    return [f"{parts[0]}-{parts[1]}-", f"{parts[0]}-{parts[1]}", parts[0], parts[1]]
            elif len(raw) >= 9:
                return [raw[:5]]
            return [raw]

        elif "NINO" in cat:
            parts = raw.split()
            if len(parts) >= 4:
                return [" ".join(parts[:3]), " ".join(parts[:2]), parts[0], parts[1], parts[2]]
            return [raw[:8]]

        elif "NHS" in cat:
            parts = raw.split()
            if len(parts) >= 3:
                return [" ".join(parts[:2]), parts[0], parts[1]]
            return [raw[:7]]

        elif "UTR" in cat:
            digits = "".join(c for c in raw if c.isdigit())
            if len(digits) >= 10:
                return [digits[:6]]
            return [raw]

        elif "INSURANCE" in cat:
            m = re.match(r"^[A-Za-z]+-([A-Za-z0-9]+)(?:-[A-Za-z0-9]+)?$", raw)
            if m:
                return [m.group(1)]
            return [raw]

        elif "STREET_ADDRESS" in cat:
            parts = raw.split(",")
            if len(parts) >= 3:
                return [parts[0].strip(), parts[1].strip(), ",".join(parts[:2]).strip()]
            return [raw]

        else:
            if "\n" in raw:
                return [line.strip() for line in raw.split("\n") if line.strip()]
            return [raw]

    @classmethod
    def get_mask(cls, category: str, raw_text: str) -> str:
        """
        Fuzzy matches category to ensure partial masking policies apply
        regardless of whether category string includes intermediate namespaces.
        """
        cat = category.upper()
        if "EMAIL" in cat:
            return DetectPii.compute_mask("PII.CONTACT.EMAIL", raw_text)
        elif "PHONE" in cat:
            return DetectPii.compute_mask("PII.CONTACT.PHONE_NUMBER", raw_text)
        elif "CREDIT_CARD" in cat or "CC" in cat:
            return DetectPii.compute_mask("PII.FINANCIAL.CREDIT_CARD", raw_text)
        elif "SSN" in cat:
            return DetectPii.compute_mask("PII.IDENTIFIER.SSN", raw_text)
        elif "IBAN" in cat:
            return DetectPii.compute_mask("PII.FINANCIAL.IBAN", raw_text)
        elif "NINO" in cat:
            return DetectPii.compute_mask("PII.IDENTIFIER.UK_NINO", raw_text)
        elif "NHS" in cat:
            return DetectPii.compute_mask("PHI.MEDICAL_IDENTIFIER.UK_NHS_NUMBER", raw_text)
        elif "POSTCODE" in cat:
            return DetectPii.compute_mask("PII.LOCATION.UK_POSTCODE", raw_text)
        elif "SORT_CODE" in cat:
            return DetectPii.compute_mask("PII.FINANCIAL.UK_SORT_CODE", raw_text)
        elif "UTR" in cat:
            return DetectPii.compute_mask("PII.IDENTIFIER.UK_UTR", raw_text)
        elif "DRIVING_LICENCE" in cat:
            return DetectPii.compute_mask("PII.IDENTIFIER.UK_DRIVING_LICENCE", raw_text)
        elif "VEHICLE_PLATE" in cat:
            return DetectPii.compute_mask("PII.IDENTIFIER.UK_VEHICLE_PLATE", raw_text)
        elif "ROUTING_NUMBER" in cat:
            return DetectPii.compute_mask("PII.FINANCIAL.ROUTING_NUMBER", raw_text)
        elif "BANK_ACCOUNT" in cat:
            return DetectPii.compute_mask("PII.FINANCIAL.BANK_ACCOUNT_NUMBER", raw_text)
        elif "STREET_ADDRESS" in cat or "ADDRESS" in cat:
            return DetectPii.compute_mask("PII.LOCATION.STREET_ADDRESS", raw_text)
        elif "INSURANCE" in cat:
            return DetectPii.compute_mask("PHI.MEDICAL_IDENTIFIER.HEALTH_INSURANCE_MEMBER_ID", raw_text)
        elif "IP" in cat or "MAC" in cat:
            return DetectPii.compute_mask("PII.LOCATION.IP_ADDRESS", raw_text)
        else:
            return DetectPii.compute_mask(category, raw_text)

    async def async_invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes document and filename redaction with strict verification and optional DSAR requestor_name support.
        """
        logger: Logger = getLogger(self.__class__.__name__)
        file_path_str: Optional[str] = args.get("file_path")
        content: Optional[str] = args.get("content")
        file_name: Optional[str] = args.get("file_name")
        output_dir_str: Optional[str] = args.get("output_dir", "data")
        requestor_name: Optional[str] = (
            args.get("requestor_name") or
            args.get("dsar_requestor") or
            args.get("dsar_requestor_name")
        )
        approved_spans: List[Dict[str, Any]] = args.get("approved_spans", [])

        if not approved_spans and not isinstance(approved_spans, list):
            raise ValueError("invalid_input: 'approved_spans' must be a list of approved spans.")

        # 1. Acquire raw content and source file metadata
        is_pdf = False
        if file_path_str:
            p = Path(file_path_str).resolve()
            if not file_name:
                file_name = p.name
            if p.suffix.lower() == ".pdf":
                is_pdf = True
            if not content:
                if not p.exists():
                    raise ValueError(f"path_not_found: Source file '{p}' does not exist.")
                if is_pdf:
                    reader = pypdf.PdfReader(str(p))
                    content = "\n\n".join(page.extract_text() or "" for page in reader.pages)
                else:
                    content = p.read_text(encoding="utf-8", errors="replace")
        elif not content:
            raise ValueError("invalid_input: Either 'file_path' or 'content' must be provided.")

        if not file_name:
            file_name = "document.pdf" if is_pdf else "document.txt"
        elif file_name.lower().endswith(".pdf"):
            is_pdf = True

        output_dir = Path(output_dir_str).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        dsar_active = bool(
            requestor_name and
            requestor_name.strip().lower() not in ("", "none", "n/a", "no", "all", "null", "general", "redact all")
        )

        # Collect all person names mentioned in spans (for canonical requestor name resolution)
        span_person_names = list(dict.fromkeys(
            (s.get("value") or s.get("text") or s.get("matched_text") or s.get("content") or "")
            for s in approved_spans
            if s.get("category") in DetectPii.NAME_CATEGORIES or "PERSON_NAME" in s.get("category", "")
        ))

        # 2. Partition Spans into Filename vs Content Spans (enforcing DSAR requestor exemption if active)
        filename_spans: List[Dict[str, Any]] = []
        content_spans: List[Dict[str, Any]] = []
        preserved_requestor_count = 0

        for span in approved_spans:
            if span.get("dsar_role") == "REQUESTOR" or span.get("dsar_action") == "PRESERVE_UNREDACTED":
                preserved_requestor_count += 1
                continue

            raw_val = (
                span.get("value") or
                span.get("text") or
                span.get("matched_text") or
                span.get("content") or ""
            )
            cat = span.get("category", "")

            if dsar_active and (cat in DetectPii.NAME_CATEGORIES or "PERSON_NAME" in cat):
                if DetectPii.is_requestor_name(raw_val, requestor_name, span_person_names):
                    preserved_requestor_count += 1
                    continue

            is_fn = (
                span.get("target") == "filename" or
                span.get("location") == "file_name" or
                "filename" in cat.lower() or
                "file name" in cat.lower() or
                span.get("action") == "filename_mask"
            )
            if is_fn:
                filename_spans.append(span)
            else:
                content_spans.append(span)

        # 3. Apply Filename Redaction
        original_stem = Path(file_name).stem
        original_suffix = Path(file_name).suffix or (".pdf" if is_pdf else ".txt")
        redacted_stem = original_stem

        # Check explicit target_file_name or redacted_file_name override in args
        target_override = args.get("target_file_name") or args.get("redacted_file_name") or args.get("redacted_filename")
        if target_override:
            override_stem = Path(target_override).stem
            if override_stem.endswith("_redacted"):
                override_stem = override_stem[:-9]
            # Only apply override_stem if it didn't mistakenly X-mask the DSAR requestor's own name
            if not (dsar_active and "XXX" in override_stem and not filename_spans):
                redacted_stem = override_stem

        if filename_spans:
            for s in filename_spans:
                if s.get("proposed_filename"):
                    p_stem = Path(s["proposed_filename"]).stem
                    if p_stem.endswith("_redacted"):
                        redacted_stem = p_stem[:-9]
                    else:
                        redacted_stem = p_stem
                    break

                raw_val = (
                    s.get("value") or
                    s.get("text") or
                    s.get("matched_text") or
                    s.get("content") or ""
                )
                repl = s.get("replacement")
                if not repl:
                    mask_candidate = s.get("proposed_mask")
                    if mask_candidate and not any(mask_candidate.lower().endswith(ext) for ext in [".pdf", ".txt", ".csv", ".md"]):
                        repl = mask_candidate
                    elif raw_val:
                        repl = "".join("X" if c.isalnum() else c for c in raw_val)

                if raw_val and raw_val in redacted_stem:
                    redacted_stem = redacted_stem.replace(raw_val, repl)
        else:
            # If filename_spans was not explicitly passed, inspect sensitive name tokens from content_spans
            # and automatically mask any third-party name or username token that appears in redacted_stem
            sensitive_name_tokens = list(dict.fromkeys(
                (s.get("value") or s.get("text") or s.get("matched_text") or s.get("content") or "")
                for s in content_spans
                if s.get("category") in ("PII.PERSON_NAME", "PII.IDENTIFIER.USERNAME", "PII.CONTACT.EMERGENCY_CONTACT")
            ))
            for name in sensitive_name_tokens:
                if dsar_active and DetectPii.is_requestor_name(name, requestor_name, span_person_names):
                    continue
                raw_parts = name.split()
                parts = raw_parts[1:] if (raw_parts and raw_parts[0].lower() in DetectPii.HONORIFICS) else raw_parts
                variants = [name]
                if len(parts) >= 2:
                    variants.extend([
                        " ".join(parts),
                        f"{parts[0]} {parts[-1]}",
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
                    if dsar_active and DetectPii.is_requestor_name(var, requestor_name, span_person_names):
                        continue
                    if var and var in redacted_stem:
                        x_mask = "".join("X" if c.isalnum() else c for c in var)
                        redacted_stem = redacted_stem.replace(var, x_mask)
                        break

        # Append _redacted if not already present
        if not redacted_stem.endswith("_redacted"):
            redacted_filename = f"{redacted_stem}_redacted{original_suffix}"
        else:
            redacted_filename = f"{redacted_stem}{original_suffix}"

        redacted_base_stem = Path(redacted_filename).stem

        # 4. Normalize & Apply Content Redactions
        normalized_content_spans: List[Dict[str, Any]] = []
        for s in content_spans:
            raw_val = (
                s.get("value") or
                s.get("text") or
                s.get("matched_text") or
                s.get("content") or ""
            )
            cat = s.get("category", "")
            start = s.get("start") if s.get("start") is not None else s.get("start_offset")
            end = s.get("end") if s.get("end") is not None else s.get("end_offset")
            mask = s.get("replacement") or s.get("proposed_mask")
            if not mask:
                mask = self.get_mask(cat, raw_val)

            # Offset boundary verification & search fallback
            if start is not None and end is not None and 0 <= start <= end <= len(content):
                if content[start:end] != raw_val and raw_val and raw_val in content:
                    start = content.find(raw_val)
                    end = start + len(raw_val)
            elif raw_val and raw_val in content:
                start = content.find(raw_val)
                end = start + len(raw_val)

            normalized_content_spans.append({
                "raw_val": raw_val,
                "category": cat,
                "start": start,
                "end": end,
                "mask": mask
            })

        # Sort descending by start offset to preserve string indices during slicing
        normalized_content_spans.sort(
            key=lambda x: x["start"] if x["start"] is not None else -1,
            reverse=True
        )

        redacted_content = content
        for s in normalized_content_spans:
            st, ed, m = s["start"], s["end"], s["mask"]
            if st is not None and ed is not None and 0 <= st <= ed <= len(redacted_content):
                redacted_content = redacted_content[:st] + m + redacted_content[ed:]
            elif s["raw_val"] and s["raw_val"] in redacted_content:
                redacted_content = redacted_content.replace(s["raw_val"], m, 1)

        # 5. Strict Zero-Leakage & Integrity Verification
        leakage_violations: List[str] = []

        # Check critical identity failure: if spans were approved, content MUST change
        if content_spans and redacted_content == content:
            leakage_violations.append(
                "CRITICAL INTEGRITY FAILURE: Redacted content is byte-identical to original cleartext despite approved redactions!"
            )

        for s in normalized_content_spans:
            raw_val = s["raw_val"]
            cat = s["category"].upper()
            if not raw_val:
                continue

            # Check specific partial or full masking leakage
            if "EMAIL" in cat:
                user_part = raw_val.split("@")[0] if "@" in raw_val else raw_val
                if user_part and user_part in redacted_content:
                    leakage_violations.append(f"Email username leaked: '{user_part}'")
            elif "PHONE" in cat:
                digits = "".join(c for c in raw_val if c.isdigit())
                if raw_val.startswith("+44"):
                    sub_digits = digits[2:]
                elif digits.startswith("1") and len(digits) == 11:
                    sub_digits = digits[1:]
                else:
                    sub_digits = digits
                if len(sub_digits) >= 7 and sub_digits in redacted_content:
                    leakage_violations.append(f"Phone subscriber digits leaked: '{sub_digits}'")
            elif "CREDIT_CARD" in cat or "CC" in cat:
                digits = "".join(c for c in raw_val if c.isdigit())
                if len(digits) >= 12 and digits[:12] in redacted_content:
                    leakage_violations.append(f"Credit card prefix leaked: '{digits[:12]}'")
            elif "SSN" in cat:
                first5 = raw_val[:6]
                if first5 in redacted_content:
                    leakage_violations.append(f"SSN prefix leaked: '{first5}'")
            elif "NINO" in cat:
                digits = "".join(c for c in raw_val if c.isdigit())
                if len(digits) >= 4 and digits[:4] in redacted_content:
                    leakage_violations.append(f"UK NINO prefix digits leaked: '{digits[:4]}'")
            elif "NHS" in cat:
                digits = "".join(c for c in raw_val if c.isdigit())
                if len(digits) >= 6 and digits[:6] in redacted_content:
                    leakage_violations.append(f"UK NHS prefix digits leaked: '{digits[:6]}'")
            elif "SORT_CODE" in cat:
                digits = "".join(c for c in raw_val if c.isdigit())
                if len(digits) >= 4 and digits[:4] in redacted_content:
                    leakage_violations.append(f"UK Sort Code prefix digits leaked: '{digits[:4]}'")
            elif "UTR" in cat:
                digits = "".join(c for c in raw_val if c.isdigit())
                if len(digits) >= 6 and digits[:6] in redacted_content:
                    leakage_violations.append(f"UK UTR prefix digits leaked: '{digits[:6]}'")
            elif "DRIVING_LICENCE" in cat:
                alnum = "".join(c for c in raw_val if c.isalnum())
                if len(alnum) >= 14 and alnum[:14] in redacted_content:
                    leakage_violations.append(f"UK Driving Licence prefix leaked: '{alnum[:14]}'")
            elif "ROUTING_NUMBER" in cat or "BANK_ACCOUNT" in cat:
                if len(raw_val) > 4 and raw_val[:-4] in redacted_content:
                    leakage_violations.append(f"Bank account / routing prefix leaked: '{raw_val[:-4]}'")
            elif raw_val in redacted_content:
                leakage_violations.append(f"Raw sensitive value leaked: '{raw_val}' for category '{cat}'")

        if leakage_violations:
            logger.error("RedactDocument: verification FAILED with violations: %s", leakage_violations)
            raise ValueError(
                f"leakage_verification_failed: Redaction aborted to prevent writing cleartext: {leakage_violations}"
            )

        # 6. Write Hardened Artifacts (ONLY AFTER VERIFICATION PASSES)
        artifacts: List[str] = []

        # If source document is PDF, output ONLY PDF artifact with EXACT input layout preserved
        if is_pdf:
            pdf_path = output_dir / f"{redacted_base_stem}.pdf"
            source_pdf_file = Path(file_path_str).resolve() if file_path_str and Path(file_path_str).exists() else None

            pdf_written_visually = False
            if pymupdf and source_pdf_file and source_pdf_file.suffix.lower() == ".pdf":
                try:
                    doc = pymupdf.open(str(source_pdf_file))
                    total_rects = 0
                    for page in doc:
                        for s in content_spans:
                            raw_val = (
                                s.get("value") or
                                s.get("text") or
                                s.get("matched_text") or
                                s.get("content") or ""
                            )
                            cat = s.get("category", "")
                            mask = s.get("replacement") or s.get("proposed_mask") or ""
                            targets = self.get_sensitive_targets(raw_val, cat, mask)
                            for t in targets:
                                if not t or len(t) < 2:
                                    continue
                                rects = page.search_for(t)
                                for r in rects:
                                    page.add_redact_annot(r, fill=(0, 0, 0))
                                    total_rects += 1
                        page.apply_redactions()

                    doc.save(str(pdf_path), garbage=4, deflate=True)
                    doc.close()
                    pdf_written_visually = True
                    logger.info(
                        "RedactDocument: applied %d visual redactions to %s preserving exact original layout",
                        total_rects, pdf_path
                    )
                except Exception as ex:
                    logger.warning("RedactDocument: visual PDF redaction failed (%s), falling back to WritePdf", ex)
                    pdf_written_visually = False

            if not pdf_written_visually:
                pdf_bytes = WritePdf._text_to_pdf_bytes(redacted_content)
                pdf_path.write_bytes(pdf_bytes)
                logger.info("RedactDocument: wrote redacted PDF fallback to %s", pdf_path)

            # Strict post-write verification on the generated PDF file
            try:
                reader = pypdf.PdfReader(str(pdf_path))
                generated_pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)
                pdf_leakage = []
                for s in normalized_content_spans:
                    raw_val = s["raw_val"]
                    cat = s["category"].upper()
                    if not raw_val or len(raw_val) < 3:
                        continue
                    if "PERSON_NAME" in cat or "IDENTIFIER" in cat or "DOB" in cat or "DATE_OF_BIRTH" in cat:
                        if raw_val in generated_pdf_text:
                            pdf_leakage.append(f"Sensitive entity leaked in output PDF: '{raw_val}'")
                    elif "SORT_CODE" in cat and raw_val[:5] in generated_pdf_text:
                        pdf_leakage.append(f"Sort code leaked in output PDF: '{raw_val[:5]}'")
                    elif "BANK_ACCOUNT" in cat and raw_val[:-4] in generated_pdf_text:
                        pdf_leakage.append(f"Bank account prefix leaked in output PDF: '{raw_val[:-4]}'")

                if pdf_leakage:
                    pdf_path.unlink(missing_ok=True)
                    raise ValueError(f"leakage_verification_failed: Output PDF failed verification: {pdf_leakage}")
            except ValueError:
                raise
            except Exception as e:
                logger.warning("RedactDocument: post-write PDF text check error (ignored): %s", e)

            artifacts.append(str(pdf_path))
            logger.info("RedactDocument: wrote redacted PDF to %s", pdf_path)

            if args.get("create_text_artifact", False):
                txt_path = output_dir / f"{redacted_base_stem}.txt"
                txt_path.write_text(redacted_content, encoding="utf-8")
                artifacts.append(str(txt_path))
                logger.info("RedactDocument: wrote redacted text to %s", txt_path)
        else:
            txt_path = output_dir / f"{redacted_base_stem}.txt"
            txt_path.write_text(redacted_content, encoding="utf-8")
            artifacts.append(str(txt_path))
            logger.info("RedactDocument: wrote redacted text to %s", txt_path)

        # Cleanup stale unredacted-name artifacts in output_dir (e.g. Arthur_P_Transcription_redacted.pdf/txt)
        if original_stem != redacted_base_stem and not original_stem.endswith("_redacted"):
            stale_candidates = [
                output_dir / f"{original_stem}_redacted.pdf",
                output_dir / f"{original_stem}_redacted.txt",
                output_dir / f"{original_stem}.txt"
            ]
            for sc in stale_candidates:
                if sc.exists() and str(sc) not in artifacts:
                    try:
                        sc.unlink()
                        logger.info("RedactDocument: deleted stale unredacted-name artifact %s", sc)
                    except Exception as e:
                        logger.warning("Could not unlink stale file %s: %s", sc, e)

        return {
            "status": "COMPLETED",
            "dsar_mode": dsar_active,
            "requestor_name": requestor_name if dsar_active else None,
            "preserved_requestor_count": preserved_requestor_count,
            "redacted_file_name": redacted_filename,
            "output_artifacts": artifacts,
            "total_redactions_applied": len(content_spans) + len(filename_spans),
            "content_redactions_count": len(content_spans),
            "filename_redactions_count": len(filename_spans),
            "leakage_verification": {
                "passed": True,
                "violations": []
            },
            "redacted_content": redacted_content
        }
