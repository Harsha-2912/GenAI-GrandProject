import json
from typing import List, Dict, Any

def get_file_icon(file_type: str) -> str:
    """Returns an appropriate emoji icon based on the file type."""
    ft = file_type.lower().replace(".", "")
    if ft == "pdf":
        return "📕"
    elif ft in ["xlsx", "xls", "csv", "excel"]:
        return "📊"
    elif ft in ["pptx", "ppt"]:
        return "📙"
    elif ft in ["docx", "doc"]:
        return "📘"
    elif ft in ["txt", "md"]:
        return "📄"
    return "📁"

def human_readable_size(num_bytes: int) -> str:
    """Formats raw byte counts into human-readable strings."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"

def export_conversation_to_markdown(history: List[Dict[str, Any]]) -> str:
    """Exports chat / Q&A conversation history to formatted Markdown."""
    lines = [
        "# DocuSage AI - Document Intelligence Q&A Session Report",
        f"Generated automatically by DocuSage AI.\n",
        "---",
        ""
    ]
    for idx, item in enumerate(history, start=1):
        q = item.get("question", "")
        a = item.get("answer", "")
        citations = item.get("citations", [])
        model = item.get("model", "DocuSage AI")
        
        lines.append(f"### Q{idx}: {q}")
        lines.append(f"**Answer ({model}):**\n")
        lines.append(a)
        lines.append("")

        if citations:
            lines.append("**Citations & Document References:**")
            for c in citations:
                tag = c.get("tag", "") if isinstance(c, dict) else getattr(c, "tag", "")
                sim = c.get("similarity_score", 0.0) if isinstance(c, dict) else getattr(c, "similarity_score", 0.0)
                lines.append(f"- `{tag}` (Similarity: {sim:.1%})")
            lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)

def export_conversation_to_json(history: List[Dict[str, Any]]) -> str:
    """Exports conversation history to serialized JSON string."""
    serializable = []
    for item in history:
        item_copy = dict(item)
        if "citations" in item_copy and isinstance(item_copy["citations"], list):
            item_copy["citations"] = [
                c if isinstance(c, dict) else {
                    "tag": getattr(c, "tag", ""),
                    "filename": getattr(c, "filename", ""),
                    "section_label": getattr(c, "section_label", ""),
                    "similarity_score": getattr(c, "similarity_score", 0.0),
                    "snippet": getattr(c, "snippet", "")
                }
                for c in item_copy["citations"]
            ]
        serializable.append(item_copy)
    return json.dumps(serializable, ensure_ascii=False, indent=2)
