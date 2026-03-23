from typing import List, Dict, Any
from collections import defaultdict
from pathlib import Path
import json

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

class DrainParser:
  """
  Simple wrapper around Drain3's TemplateMiner for:
    - training on a log file
    - exporting learned templates
    - validating template quality heuristically
  """
  def __init__(self, config_path: str | None = None):
    cfg = TemplateMinerConfig()
    if config_path:
      cfg.load(config_path)
    self.miner = TemplateMiner(config=cfg)
    
    # For validation
    self.template_to_lines: Dict[str, List[str]] = defaultdict(list)
    self.line_template_ids: List[str] = []  


  def fit_file(self, log_path: str, max_lines: int | None = None) -> None:
    log_path = Path(log_path)
    print(f"[INFO] Training Drain3 on: {log_path}")

    with log_path.open("r", errors="replace") as f:
      for i, raw in enumerate(f, start=1):
        line = raw.rstrip("\n")
        if not line:
          continue

        result = self.miner.add_log_message(line) # Process the log line through Drain3
        cluster_id = result.get("cluster_id")
        template   = result.get("template_mined")

        if cluster_id is None or template is None:
          print(f"[WARN] No cluster for line {i}: {line[:120]}...")
          continue
        
        # Store for validation
        self.line_template_ids.append(cluster_id)
        self.template_to_lines[template].append(line)

        # Early stopping
        if max_lines is not None and i >= max_lines:
          print(f"[INFO] Stopped early at {max_lines} lines")
          break

        if i % 100000 == 0:
          print(f"[INFO] Processed {i} lines...")
      
      print(f"[INFO] Parsed {len(self.line_template_ids)} log lines total.")
      print(f"[INFO] Learned {len(self.template_to_lines)} distinct templates.")
  

  def export_templates(self, out_path: str) -> None:
    """Export learned templates to a file."""
    records = []
    for cluster in self.miner.drain.id_to_cluster.values():
        tmpl = " ".join(cluster.log_template_tokens)
        lines = self.template_to_lines.get(tmpl, [])
        records.append({
            "template": tmpl,             # The template string learned by Drain3
            "count": cluster.size,        # authoritative count from Drain3
            "examples": lines[:5]         # Include up to 5 example lines for context
        })

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
      json.dump(records, f, indent=2)

    print(f"[INFO] Exported {len(records)} templates → {out_path}")


  def _print_template_support_distribution(self) -> List[int]:
    support_counts = [len(v) for v in self.template_to_lines.values()]
    support_counts.sort()
    min_sup = support_counts[0]
    max_sup = support_counts[-1]
    avg_sup = sum(support_counts) / len(support_counts)
    print(
      f"[INFO] Template support (lines per template): "
      f"min={min_sup}, max={max_sup}, avg={avg_sup:.1f}"
    )
    return support_counts


  def _validate_singleton_templates(self, support_counts: List[int], n_templates: int) -> None:
    singletons = sum(1 for s in support_counts if s == 1)
    if singletons / n_templates > 0.3:
      print(
        f"[WARN] High fraction of singleton templates "
        f"({singletons}/{n_templates} ≈ {singletons/n_templates:.1%}). "
        f"Drain might be overfitting (simply memorizing lines)."
      )
    else:
      print("[OK] Singleton template fraction looks reasonable.")


  def _validate_overly_generic_templates(self) -> None:
    too_generic = []
    for tmpl, lines in self.template_to_lines.items():
      num_tokens = len(tmpl.split())
      num_wild = tmpl.count("<*>")
      if num_tokens > 0 and num_wild / num_tokens > 0.7:
        too_generic.append((tmpl, len(lines)))

    if too_generic:
      print(
        f"[WARN] Found {len(too_generic)} templates that look very generic "
        f"(>70% wildcard tokens). Examples:"
      )
      for tmpl, sup in too_generic[:5]:
        print(f"      support={sup:4d} | template='{tmpl}'")
      if len(too_generic) > 5:
        print("      ...")
    else:
      print("[OK] No overly generic templates detected by wildcard heuristic.")


  def _validate_overly_specific_templates(self) -> None:
    too_specific = []
    for tmpl, lines in self.template_to_lines.items():
      if len(lines) == 1 and "<*>" not in tmpl:
        too_specific.append(tmpl)

    if too_specific:
      print(
        f"[WARN] Found {len(too_specific)} templates that are "
        f"single-use with no wildcards (likely overfitting)."
      )
      for tmpl in too_specific[:5]:
        print(f"      '{tmpl}'")
      if len(too_specific) > 5:
        print("      ...")
    else:
      print("[OK] No obviously over-specific single-use templates found.")


  def validate(self) -> None:
    """Heuristic validation of learned templates."""
    print("\n[INFO] Running template validation...")

    total_lines = len(self.line_template_ids)
    n_templates = len(self.template_to_lines)

    if total_lines == 0:
        print("[ERROR] No lines were parsed. Check your log path / format.")
        return
    
    print(f"[INFO] Total lines parsed   : {total_lines}")
    print(f"[INFO] Distinct templates   : {n_templates}")

    assigned = len(self.line_template_ids)
    if assigned != total_lines:
        print(f"[WARN] Coverage mismatch: {assigned}/{total_lines} lines have a cluster_id.")
    else:
        print("[OK] 100% of lines received a template assignment.")

    support_counts = self._print_template_support_distribution()
    self._validate_singleton_templates(support_counts, n_templates)
    self._validate_overly_generic_templates()
    self._validate_overly_specific_templates()