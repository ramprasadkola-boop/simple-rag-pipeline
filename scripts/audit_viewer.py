#!/usr/bin/env python3
"""
Simple CLI to view audit records from data/audit.db
Usage: python scripts/audit_viewer.py [--tail N] [--csv]
"""
import sqlite3
import os
import sys
import json
from pathlib import Path


def get_db_path() -> str:
    return os.environ.get("AUDIT_DB_PATH", "data/audit.db")


def view_audit(tail: int = None, csv: bool = False):
    db_path = get_db_path()
    if not os.path.exists(db_path):
        print(f"No audit DB found at {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, ts, system_message, user_message, response, served_from_fallback, metadata FROM ai_audit ORDER BY id")
    except Exception as e:
        print(f"Query failed: {e}")
        conn.close()
        return
    
    rows = cur.fetchall()
    if not rows:
        print("No audit records found")
        conn.close()
        return
    
    # Apply tail limit
    if tail:
        rows = rows[-tail:]
    
    if csv:
        # Simple CSV output
        print("id,timestamp,fallback,system_message_len,user_message_len,response_len")
        for r in rows:
            sys_len = len(r[2] or "")
            usr_len = len(r[3] or "")
            resp_len = len(r[4] or "")
            print(f"{r[0]},{r[1]},{r[5]},{sys_len},{usr_len},{resp_len}")
    else:
        # Human-readable format with side-by-side display (request/response "facing each other")
        for i, r in enumerate(rows, 1):
            record_id = r[0]
            timestamp = r[1]
            system_msg = r[2] or "[empty]"
            user_msg = r[3] or "[empty]"
            response = r[4] or "[empty]"
            is_fallback = r[5]
            metadata = r[6]

            print("\n" + "=" * 100)
            print(f"Record #{record_id} | {timestamp} | Fallback: {'YES ⚠️' if is_fallback else 'NO ✅'}")
            if metadata:
                try:
                    meta = json.loads(metadata)
                    print(f"Metadata: {meta}")
                except:
                    print(f"Metadata: {metadata}")

            # Side-by-side format: REQUEST | RESPONSE
            print("\n" + "-" * 100)
            
            # Truncate long messages for display
            max_width = 45
            
            def truncate_lines(text, max_width=max_width):
                lines = (text or "[empty]").split('\n')
                result = []
                for line in lines:
                    if len(line) > max_width:
                        result.append(line[:max_width-3] + "...")
                    else:
                        result.append(line)
                return result
            
            sys_lines = truncate_lines(system_msg)
            usr_lines = truncate_lines(user_msg)
            resp_lines = truncate_lines(response)
            
            # Combine system and user into "request"
            request_lines = ["[SYSTEM]"] + sys_lines + ["[QUERY]"] + usr_lines
            response_lines = ["[RESPONSE]"] + resp_lines
            
            # Print side by side
            max_request_lines = len(request_lines)
            max_response_lines = len(response_lines)
            max_lines = max(max_request_lines, max_response_lines)
            
            # Header
            print(f"{'REQUEST':^45} | {'RESPONSE':^45}")
            print("-" * 45 + "|" + "-" * 45)
            
            for idx in range(max_lines):
                req_line = request_lines[idx] if idx < len(request_lines) else ""
                resp_line = response_lines[idx] if idx < len(response_lines) else ""
                print(f"{req_line:<45} | {resp_line:<45}")
    
    conn.close()
    print("\n" + "="*80)
    print(f"Total records: {len(rows)}")


if __name__ == "__main__":
    tail = None
    csv = False
    
    for arg in sys.argv[1:]:
        if arg == "--csv":
            csv = True
        elif arg.startswith("--tail="):
            tail = int(arg.split("=")[1])
        elif arg.startswith("--tail"):
            # next arg is the number
            try:
                idx = sys.argv.index(arg)
                tail = int(sys.argv[idx + 1])
            except:
                pass
    
    view_audit(tail=tail, csv=csv)
