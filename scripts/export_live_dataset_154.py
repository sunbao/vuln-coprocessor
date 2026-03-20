#!/usr/bin/env python3

import argparse
import csv
import datetime as dt
import errno
import hashlib
import json
import os
import pty
import re
import select
import shlex
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


LANGUAGE_MAP = {
    "1": "Java",
    "2": "JavaScript",
    "3": "Php",
    "4": "Ruby",
    "5": "Golang",
    "6": "Rust",
    "7": "Erlang",
    "8": "Python",
}

RISK_MAP = {
    "4": "超危",
    "3": "高危",
    "2": "中危",
    "1": "低危",
    "0": "未评级",
    "-1": "安全",
}

DEFAULT_SOURCE_TABLES = [
    "easyw-sca.project_assembly",
    "easyw-sca.assembly",
    "easyw-sca.assembly_version",
    "easyw-sca.assembly_new",
    "easyw-sca.loophole",
    "easyw-crawling.cpematch",
    "easyw-crawling.cveitem",
    "easyw-crawling.cnnvd",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a V2 dataset using the live online matching source."
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prefix", default="v2-live-154-online")
    parser.add_argument("--ssh_host", default=os.environ.get("LASUN_SSH_HOST", "127.0.0.1"))
    parser.add_argument("--ssh_user", default=os.environ.get("LASUN_SSH_USER", "root"))
    parser.add_argument("--ssh_password", default=os.environ.get("LASUN_SSH_PASSWORD"))
    parser.add_argument("--db_host", default=os.environ.get("LASUN_DB_HOST", "127.0.0.1"))
    parser.add_argument("--db_user", default=os.environ.get("LASUN_DB_USER", "root"))
    parser.add_argument("--db_password", default=os.environ.get("LASUN_DB_PASSWORD"))
    parser.add_argument("--split_scope", choices=["project_assembly_id", "project_id"], default="project_assembly_id")
    parser.add_argument("--candidate_limit", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--page_size", type=int, default=100)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--start_loophole_id", default="0")
    parser.add_argument("--sleep_seconds", type=float, default=1.0)
    parser.add_argument("--project_assembly_ids", default="")
    parser.add_argument("--count_only", action="store_true")
    parser.add_argument("--allow_full_export", action="store_true")
    return parser.parse_args()


def require_secret(value: Optional[str], name: str) -> str:
    if value:
        return value
    raise SystemExit(f"Missing required secret: {name}")


def run_ssh_command(host: str, user: str, password: str, remote_command: str, timeout: int = 600) -> str:
    pid, master_fd = pty.fork()
    if pid == 0:
        os.execvp(
            "ssh",
            ["ssh", "-tt", "-o", "StrictHostKeyChecking=no", f"{user}@{host}", remote_command],
        )

    chunks: List[bytes] = []
    sent_password = False
    start = time.time()
    child_returncode = 0

    try:
        while True:
            if time.time() - start > timeout:
                try:
                    os.kill(pid, 9)
                except ProcessLookupError:
                    pass
                raise TimeoutError(f"SSH command timed out after {timeout}s")

            ready, _, _ = select.select([master_fd], [], [], 0.2)
            if master_fd in ready:
                try:
                    data = os.read(master_fd, 65536)
                except OSError as exc:
                    if exc.errno == errno.EIO:
                        data = b""
                    else:
                        raise
                if not data:
                    break
                chunks.append(data)
                text = data.decode("utf-8", errors="replace").lower()
                if "are you sure you want to continue connecting" in text:
                    os.write(master_fd, b"yes\n")
                elif (not sent_password) and "password:" in text:
                    os.write(master_fd, (password + "\n").encode("utf-8"))
                    sent_password = True

            try:
                waited_pid, status = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                waited_pid, status = pid, 0
            if waited_pid == pid:
                if os.WIFEXITED(status):
                    child_returncode = os.WEXITSTATUS(status)
                elif os.WIFSIGNALED(status):
                    child_returncode = 128 + os.WTERMSIG(status)
                ready, _, _ = select.select([master_fd], [], [], 0)
                if not ready:
                    break
    finally:
        os.close(master_fd)

    output = b"".join(chunks).decode("utf-8", errors="replace")
    cleaned = []
    for line in output.splitlines():
        lowered = line.lower()
        if "password:" in lowered:
            continue
        if "are you sure you want to continue connecting" in lowered:
            continue
        cleaned.append(line)
    result = "\n".join(cleaned)
    if child_returncode != 0:
        raise RuntimeError(result.strip() or f"ssh command failed with exit code {child_returncode}")
    return result


def mysql_query_tsv(
    ssh_host: str,
    ssh_user: str,
    ssh_password: str,
    db_host: str,
    db_user: str,
    db_password: str,
    sql: str,
    timeout: int = 600,
    expected_columns: Optional[int] = None,
) -> List[List[str]]:
    remote_command = (
        f"MYSQL_PWD={shlex.quote(db_password)} "
        f"mysql --batch --raw --skip-column-names -h{shlex.quote(db_host)} "
        f"-u{shlex.quote(db_user)} -e {shlex.quote(sql)}"
    )
    output = run_ssh_command(ssh_host, ssh_user, ssh_password, remote_command, timeout=timeout)
    rows: List[List[str]] = []
    reader = csv.reader(output.splitlines(), delimiter="\t")
    for row in reader:
        if not row:
            continue
        if row[0].startswith("mysql: [Warning]"):
            continue
        if row[0].startswith("Connection to "):
            continue
        if expected_columns is not None and len(row) != expected_columns:
            continue
        rows.append(row)
    return rows


def batched(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for idx in range(0, len(values), size):
        yield values[idx : idx + size]


def normalize_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def normalize_version(raw_version: Optional[str]) -> Tuple[str, str]:
    raw = (raw_version or "").strip()
    normalized = raw.replace("^", "").replace("~", "").replace("==", "").strip()
    suffix = ""
    if "-" in normalized:
        base, suffix = normalized.split("-", 1)
        normalized = base.strip()
        suffix = suffix.strip()
    return normalized, suffix


def ecosystem_from_source(source: Optional[str], language_name: str) -> str:
    if source:
        primary = source.split(",")[0].strip().lower()
        if primary:
            return primary
    fallback = {
        "Java": "maven",
        "JavaScript": "npm",
        "Php": "composer",
        "Ruby": "ruby",
        "Golang": "go",
        "Rust": "cargo",
        "Python": "pypi",
    }
    return fallback.get(language_name, "unknown")


def build_split(value: str) -> str:
    bucket = int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16) % 10
    if bucket == 0:
        return "validation"
    if bucket == 1:
        return "test"
    return "train"


def score_candidate(candidate: Dict[str, Optional[str]], component_name: str, vendor: str, language_name: str, normalized_version: str) -> int:
    score = 0
    candidate_product = normalize_text(candidate.get("product"))
    component_product = normalize_text(component_name)
    candidate_vendor = normalize_text(candidate.get("vendor"))
    component_vendor = normalize_text(vendor)
    candidate_lang = (candidate.get("p_lang") or "").strip().lower()
    component_lang = language_name.strip().lower()
    candidate_version = (candidate.get("version") or "").strip()

    if candidate_product and candidate_product == component_product:
        score += 100
    elif candidate_product and component_product and (candidate_product in component_product or component_product in candidate_product):
        score += 60

    if candidate_vendor and component_vendor and candidate_vendor == component_vendor:
        score += 40
    elif candidate_vendor and component_vendor and (candidate_vendor in component_vendor or component_vendor in candidate_vendor):
        score += 20

    if candidate_lang and component_lang and candidate_lang == component_lang:
        score += 10

    if candidate_version == "*":
        score += 3
    elif normalized_version and candidate_version == normalized_version:
        score += 15

    if any(candidate.get(key) for key in ("versionStartIncluding", "versionStartExcluding", "versionEndIncluding", "versionEndExcluding")):
        score += 2

    return score


def choose_candidates(
    candidates_by_cve: Dict[str, List[Dict[str, Optional[str]]]],
    cve_id: str,
    component_name: str,
    vendor: str,
    language_name: str,
    normalized_version: str,
    limit: int,
) -> List[Dict[str, Optional[str]]]:
    candidates = candidates_by_cve.get(cve_id, [])
    ranked = sorted(
        candidates,
        key=lambda item: (
            score_candidate(item, component_name, vendor, language_name, normalized_version),
            item.get("vendor") or "",
            item.get("product") or "",
            item.get("version") or "",
        ),
        reverse=True,
    )
    return ranked[:limit]


def build_output(sample_input: Dict[str, object]) -> Dict[str, str]:
    component_name = str(sample_input["component_name"])
    component_version = str(sample_input["component_version"])
    vulnerability_id = str(sample_input["vulnerability_id"])
    vulnerability_name = str(sample_input.get("vulnerability_name") or "").strip()
    risk_level = str(sample_input.get("risk_level") or "")
    cvss_score = sample_input.get("cvss_score")
    recommended_version = str(sample_input.get("recommended_version") or "").strip()
    latest_version = str(sample_input.get("latest_version") or "").strip()
    vendor = str(sample_input.get("component_vendor") or "").strip()

    summary = (
        f"组件 `{component_name}` 当前版本 `{component_version}` 命中漏洞 "
        f"`{vulnerability_id}`（{vulnerability_name}），风险等级为 `{risk_level}`"
    )
    if cvss_score not in ("", None):
        summary += f"，CVSS 分数约为 {float(cvss_score):.2f}。"
    else:
        summary += "。"

    why_affected = (
        f"项目组件清单中存在 `{component_name}` 版本 `{component_version}`，并在漏洞结果表中关联到 "
        f"`{vulnerability_id}`。"
    )
    evidence = (sample_input.get("evidence") or {})
    cpematch_candidates = evidence.get("cpematch_candidates") or []
    if cpematch_candidates:
        first = cpematch_candidates[0]
        cand_vendor = str(first.get("vendor") or "").strip()
        cand_product = str(first.get("product") or "").strip()
        if cand_vendor or cand_product:
            why_affected += f" 参考在线匹配规则，供应商 `{cand_vendor or vendor or '未知'}`、产品 `{cand_product or component_name}` 与该组件能够对应。"

    if risk_level in {"超危", "高危"}:
        risk_assessment = f"从当前数据看，该问题属于 `{risk_level}`。如果该组件位于核心依赖链或外部输入处理路径，应优先处理。"
    elif risk_level in {"中危", "低危"}:
        risk_assessment = f"从当前数据看，该问题属于 `{risk_level}`。建议结合实际暴露面和调用路径安排修复优先级。"
    else:
        risk_assessment = "当前记录未体现明确风险排序，建议结合业务暴露面和组件调用位置做进一步评估。"

    if recommended_version and recommended_version != component_version:
        remediation = f"优先评估将 `{component_name}` 从 `{component_version}` 升级到 `{recommended_version}`，并完成回归验证。"
        upgrade_recommendation = f"建议优先采用推荐版本 `{recommended_version}`；如果业务约束允许，再评估是否进一步跟进到最新版本 `{latest_version}`。" if latest_version else f"建议优先采用推荐版本 `{recommended_version}`。"
    elif latest_version and latest_version != component_version:
        remediation = f"当前记录中的推荐版本未体现更高版本收益，建议至少评估升级到 `{latest_version}`，并完成回归验证。"
        upgrade_recommendation = f"建议结合兼容性验证后升级到 `{latest_version}`。"
    else:
        remediation = f"建议先确认 `{component_name}` 的安全可用版本，再制定升级、替换或缓解方案。"
        upgrade_recommendation = "当前样本缺少明确可执行升级目标，需要结合组件仓库和兼容性要求进一步确认。"

    return {
        "summary": summary,
        "why_affected": why_affected,
        "risk_assessment": risk_assessment,
        "remediation": remediation,
        "upgrade_recommendation": upgrade_recommendation,
    }


def parse_csv_ids(raw_value: str) -> List[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def build_where_clause(args: argparse.Namespace, last_loophole_id: Optional[str] = None) -> str:
    clauses = [
        "l.code IS NOT NULL",
        "l.code <> ''",
        "a.name IS NOT NULL",
        "a.name <> ''",
    ]
    if last_loophole_id is not None:
        clauses.append(f"l.id > {int(last_loophole_id)}")
    project_assembly_ids = parse_csv_ids(args.project_assembly_ids)
    if project_assembly_ids:
        in_clause = ",".join(str(int(value)) for value in project_assembly_ids)
        clauses.append(f"l.project_assembly_id IN ({in_clause})")
    return " AND ".join(clauses)


def build_base_page_sql(args: argparse.Namespace, last_loophole_id: str) -> str:
    where_clause = build_where_clause(args, last_loophole_id)
    return f"""
    USE easyw-sca;
    SELECT
      l.id,
      l.project_assembly_id,
      COALESCE(an.project_id, pa.project_id),
      l.assembly_id,
      a.name,
      COALESCE(av.org, an.org, ''),
      COALESCE(av.suffix, ''),
      COALESCE(NULLIF(a.vendor, ''), an.vendor, ''),
      a.language,
      COALESCE(NULLIF(an.source, ''), a.source, ''),
      COALESCE(NULLIF(a.recommended_version, ''), an.recommended_version, ''),
      COALESCE(NULLIF(a.latest_version, ''), an.latest_version, ''),
      l.code,
      COALESCE(l.risk_level, ''),
      COALESCE(l.cvss_score, ''),
      COALESCE(l.cwe_Id, ''),
      COALESCE(l.cnnvd_code, ''),
      COALESCE(l.name, '')
    FROM loophole l
    INNER JOIN assembly a ON a.id = l.assembly_id
    LEFT JOIN assembly_version av ON av.assembly_id = a.id
    LEFT JOIN assembly_new an ON an.id = a.id
    LEFT JOIN project_assembly pa ON pa.id = l.project_assembly_id
    WHERE {where_clause}
    ORDER BY l.id
    LIMIT {int(args.page_size)}
    """


def build_count_sql(args: argparse.Namespace) -> str:
    where_clause = build_where_clause(args, None)
    return f"""
    USE easyw-sca;
    SELECT COUNT(1)
    FROM loophole l
    INNER JOIN assembly a ON a.id = l.assembly_id
    WHERE {where_clause}
    """


def validate_args(args: argparse.Namespace) -> None:
    if args.page_size <= 0:
        raise SystemExit("--page_size must be > 0")
    if args.batch_size <= 0:
        raise SystemExit("--batch_size must be > 0")
    if args.sleep_seconds < 0:
        raise SystemExit("--sleep_seconds must be >= 0")
    if args.max_rows is not None and args.max_rows <= 0:
        raise SystemExit("--max_rows must be > 0 when provided")
    if int(args.start_loophole_id) < 0:
        raise SystemExit("--start_loophole_id must be >= 0")

    project_assembly_ids = parse_csv_ids(args.project_assembly_ids)
    staged_scope_selected = bool(project_assembly_ids) or args.max_rows is not None
    if not args.count_only and not staged_scope_selected and not args.allow_full_export:
        raise SystemExit(
            "Refusing unrestricted export from production. "
            "Provide --max_rows, --project_assembly_ids, or explicitly pass --allow_full_export."
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.ssh_password = require_secret(args.ssh_password, "LASUN_SSH_PASSWORD or --ssh_password")
    args.db_password = require_secret(args.db_password, "LASUN_DB_PASSWORD or --db_password")
    os.makedirs(args.output_dir, exist_ok=True)

    count_rows = mysql_query_tsv(
        args.ssh_host,
        args.ssh_user,
        args.ssh_password,
        args.db_host,
        args.db_user,
        args.db_password,
        build_count_sql(args),
        timeout=1800,
        expected_columns=1,
    )
    estimated_row_count = int(count_rows[0][0]) if count_rows else 0

    if args.count_only:
        print(
            json.dumps(
                {
                    "prefix": args.prefix,
                    "source_mode": "live-154-online",
                    "estimated_row_count": estimated_row_count,
                    "page_size": args.page_size,
                    "project_assembly_ids": parse_csv_ids(args.project_assembly_ids),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    created_at = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    combined_path = os.path.join(args.output_dir, f"{args.prefix}.jsonl")
    split_files = {
        "train": os.path.join(args.output_dir, f"{args.prefix}-train.jsonl"),
        "validation": os.path.join(args.output_dir, f"{args.prefix}-validation.jsonl"),
        "test": os.path.join(args.output_dir, f"{args.prefix}-test.jsonl"),
    }

    combined_handle = open(combined_path, "w", encoding="utf-8")
    split_handles = {name: open(path, "w", encoding="utf-8") for name, path in split_files.items()}

    summary_counter = Counter()
    ecosystem_counter = Counter()
    risk_counter = Counter()
    split_counter = Counter()
    source_counter = Counter()
    unique_projects = set()
    unique_project_assemblies = set()
    unique_cves_in_output = set()
    page_counter = 0
    last_loophole_id = str(int(args.start_loophole_id))

    try:
        while True:
            if args.max_rows is not None and summary_counter["sample_count"] >= args.max_rows:
                break

            base_rows = mysql_query_tsv(
                args.ssh_host,
                args.ssh_user,
                args.ssh_password,
                args.db_host,
                args.db_user,
                args.db_password,
                build_base_page_sql(args, last_loophole_id),
                timeout=1800,
                expected_columns=18,
            )
            if not base_rows:
                break

            page_counter += 1
            unique_cves = sorted({row[12] for row in base_rows if row[12]})
            candidates_by_cve: Dict[str, List[Dict[str, Optional[str]]]] = defaultdict(list)

            for batch in batched(unique_cves, args.batch_size):
                in_clause = ",".join("'" + value.replace("'", "''") + "'" for value in batch)
                candidate_sql = f"""
                USE easyw-crawling;
                SELECT
                  cveId,
                  COALESCE(vendor, ''),
                  COALESCE(product, ''),
                  COALESCE(version, ''),
                  COALESCE(updateV, ''),
                  COALESCE(targetSW, ''),
                  COALESCE(versionStartIncluding, ''),
                  COALESCE(versionStartExcluding, ''),
                  COALESCE(versionEndIncluding, ''),
                  COALESCE(versionEndExcluding, ''),
                  COALESCE(p_lang, '')
                FROM cpematch
                WHERE part = 'a'
                  AND cveId IN ({in_clause})
                ORDER BY cveId
                """
                candidate_rows = mysql_query_tsv(
                    args.ssh_host,
                    args.ssh_user,
                    args.ssh_password,
                    args.db_host,
                    args.db_user,
                    args.db_password,
                    candidate_sql,
                    timeout=1800,
                    expected_columns=11,
                )
                for row in candidate_rows:
                    candidates_by_cve[row[0]].append(
                        {
                            "cveId": row[0],
                            "vendor": row[1] or None,
                            "product": row[2] or None,
                            "version": row[3] or None,
                            "updateV": row[4] or None,
                            "targetSW": row[5] or None,
                            "versionStartIncluding": row[6] or None,
                            "versionStartExcluding": row[7] or None,
                            "versionEndIncluding": row[8] or None,
                            "versionEndExcluding": row[9] or None,
                            "p_lang": row[10] or None,
                        }
                    )

            for row in base_rows:
                if args.max_rows is not None and summary_counter["sample_count"] >= args.max_rows:
                    break

                (
                    loophole_id,
                    project_assembly_id,
                    project_id,
                    assembly_id,
                    component_name,
                    component_version_raw,
                    component_suffix_from_db,
                    component_vendor,
                    component_language_code,
                    component_source,
                    recommended_version,
                    latest_version,
                    vulnerability_id,
                    risk_level_code,
                    cvss_score_raw,
                    cwe_id,
                    cnnvd_code,
                    vulnerability_name,
                ) = row

                last_loophole_id = loophole_id
                language_name = LANGUAGE_MAP.get(component_language_code, component_language_code or "未知")
                ecosystem = ecosystem_from_source(component_source, language_name)
                normalized_version, derived_suffix = normalize_version(component_version_raw)
                component_suffix = component_suffix_from_db or derived_suffix
                split_seed = project_assembly_id if args.split_scope == "project_assembly_id" else project_id
                split = build_split(f"{args.split_scope}:{split_seed}")
                candidates = choose_candidates(
                    candidates_by_cve,
                    vulnerability_id,
                    component_name,
                    component_vendor,
                    language_name,
                    normalized_version,
                    args.candidate_limit,
                )

                sample_input = {
                    "ecosystem": ecosystem,
                    "component_name": component_name,
                    "component_version": component_version_raw,
                    "component_version_raw": component_version_raw,
                    "component_version_normalized": normalized_version,
                    "component_version_suffix": component_suffix or None,
                    "project_id": project_id,
                    "project_assembly_id": project_assembly_id,
                    "assembly_id": assembly_id,
                    "loophole_id": loophole_id,
                    "vulnerability_id": vulnerability_id,
                    "risk_level": RISK_MAP.get(risk_level_code, risk_level_code or "未知"),
                    "cvss_score": float(cvss_score_raw) if cvss_score_raw not in ("", None) else None,
                    "cwe_id": cwe_id or None,
                    "cnnvd_code": cnnvd_code or None,
                    "vulnerability_name": vulnerability_name.strip(),
                    "recommended_version": recommended_version or None,
                    "latest_version": latest_version or None,
                    "component_vendor": component_vendor or None,
                    "component_language": language_name,
                    "component_source": component_source,
                    "evidence": {
                        "source_mode": "live-154-online",
                        "source_tables": DEFAULT_SOURCE_TABLES,
                        "candidate_selection": "top-ranked by product/vendor/language/version heuristic over easyw-crawling.cpematch",
                        "cpematch_candidates": candidates,
                        "impact_statement": "component version is linked to a recorded vulnerability finding in the project result tables",
                    },
                }

                sample = {
                    "sample_id": f"{ecosystem}-{project_assembly_id}-{assembly_id}-{vulnerability_id}",
                    "task": "vulnerability_explanation",
                    "language": "zh-CN",
                    "input": sample_input,
                    "instruction": "请基于给定事实，说明漏洞风险、命中原因、处理建议，并给出是否建议升级的结论。不要编造未提供的事实。",
                    "output": build_output(sample_input),
                    "metadata": {
                        "source": args.prefix,
                        "source_mode": "live-154-online",
                        "split": split,
                        "split_scope": args.split_scope,
                        "created_at": created_at,
                    },
                }

                line = json.dumps(sample, ensure_ascii=False)
                combined_handle.write(line + "\n")
                split_handles[split].write(line + "\n")

                summary_counter["sample_count"] += 1
                ecosystem_counter[ecosystem] += 1
                risk_counter[sample_input["risk_level"]] += 1
                split_counter[split] += 1
                source_counter[component_source] += 1
                unique_projects.add(project_id)
                unique_project_assemblies.add(project_assembly_id)
                unique_cves_in_output.add(vulnerability_id)

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
    finally:
        combined_handle.close()
        for handle in split_handles.values():
            handle.close()

    summary = {
        "prefix": args.prefix,
        "source_mode": "live-154-online",
        "created_at": created_at,
        "split_scope": args.split_scope,
        "estimated_row_count": estimated_row_count,
        "combined_file": combined_path,
        "split_files": split_files,
        "sample_count": summary_counter["sample_count"],
        "unique_project_count": len(unique_projects),
        "unique_project_assembly_count": len(unique_project_assemblies),
        "unique_vulnerability_count": len(unique_cves_in_output),
        "counts_by_split": dict(split_counter),
        "counts_by_ecosystem": dict(ecosystem_counter),
        "counts_by_risk_level": dict(risk_counter),
        "counts_by_component_source": dict(source_counter),
        "candidate_limit": args.candidate_limit,
        "page_size": args.page_size,
        "page_count": page_counter,
        "sleep_seconds": args.sleep_seconds,
        "start_loophole_id": args.start_loophole_id,
        "last_loophole_id": last_loophole_id,
        "project_assembly_ids": parse_csv_ids(args.project_assembly_ids),
    }

    summary_path = os.path.join(args.output_dir, f"{args.prefix}-summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
