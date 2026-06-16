import sqlite3

DATABASE_PATH = r"" # Path to your database
TARGET_PROBLEM = 'token-classification'

conn = sqlite3.connect(DATABASE_PATH)
cur = conn.cursor()

# Get status counts
cur.execute("""
    SELECT COALESCE(health_status, 'UNTESTED') as status, COUNT(*) 
    FROM Models WHERE problem = ?
    GROUP BY health_status ORDER BY COUNT(*) DESC
""", (TARGET_PROBLEM,))
status_counts = cur.fetchall()

# Get all models with their status AND library
cur.execute("""
    SELECT model_name, health_status, health_error, downloads, library
    FROM Models WHERE problem = ?
    ORDER BY health_status, library, downloads DESC
""", (TARGET_PROBLEM,))
all_models = cur.fetchall()

# Get status counts by library
cur.execute("""
    SELECT library, COALESCE(health_status, 'UNTESTED') as status, COUNT(*)
    FROM Models WHERE problem = ?
    GROUP BY library, health_status
    ORDER BY library, COUNT(*) DESC
""", (TARGET_PROBLEM,))
library_status_counts = cur.fetchall()

conn.close()

# Organize library stats into a dict
library_stats = {}
for lib, status, count in library_status_counts:
    if lib not in library_stats:
        library_stats[lib] = {}
    library_stats[lib][status] = count

# Calculate totals
total = sum(c for _, c in status_counts)
ok_count = next((c for s, c in status_counts if s == 'OK'), 0)
tested = sum(c for s, c in status_counts if s != 'UNTESTED')

def generate_report():
    lines = []
    
    lines.append(f"\n{'='*70}")
    lines.append(f"HEALTH REPORT: {TARGET_PROBLEM.upper()}")
    lines.append(f"{'='*70}")
    lines.append(f"Total: {total:,} | Tested: {tested:,} | OK: {ok_count:,} | Failed: {tested - ok_count:,}")
    
    lines.append(f"\nSTATUS BREAKDOWN:")
    for status, count in status_counts:
        pct = count / total * 100
        lines.append(f"  {status:<15} {count:>5} ({pct:>5.1f}%)")
    
    # Library summary
    lines.append(f"\n{'='*70}")
    lines.append("SUMMARY BY LIBRARY")
    lines.append(f"{'='*70}")
    
    for lib in sorted(library_stats.keys()):
        stats = library_stats[lib]
        lib_total = sum(stats.values())
        lib_ok = stats.get('OK', 0)
        lib_fail = sum(v for k, v in stats.items() if k not in ('OK', 'UNTESTED'))
        lib_untested = stats.get('UNTESTED', 0)
        
        lines.append(f"\n  [{lib.upper()}] - {lib_total} models")
        lines.append(f"    OK: {lib_ok} | FAIL: {lib_fail} | UNTESTED: {lib_untested}")
        for status, count in sorted(stats.items(), key=lambda x: -x[1]):
            pct = count / lib_total * 100 if lib_total > 0 else 0
            lines.append(f"      {status:<15} {count:>5} ({pct:>5.1f}%)")
    
    # All models by status, grouped by library
    lines.append(f"\n{'='*70}")
    lines.append("ALL MODELS BY STATUS (GROUPED BY LIBRARY)")
    lines.append(f"{'='*70}")
    
    current_status = None
    current_library = None
    
    for model_name, status, error, downloads, library in all_models:
        status = status or 'UNTESTED'
        library = library or 'unknown'
        
        # Print header when status changes
        if status != current_status:
            current_status = status
            current_library = None  # Reset library when status changes
            count = next((c for s, c in status_counts if s == status), 0)
            lines.append(f"\n[{status}] - {count} models")
            lines.append("-" * 70)
        
        # Print library subheader when library changes within same status
        if library != current_library:
            current_library = library
            # Count models for this status+library combo
            lib_count = sum(1 for m in all_models 
                          if (m[1] or 'UNTESTED') == status and (m[4] or 'unknown') == library)
            lines.append(f"\n  --- {library} ({lib_count} models) ---")
        
        # Print model info
        if status == 'OK':
            lines.append(f"  {model_name} ({downloads:,} downloads)")
        elif status == 'UNTESTED':
            lines.append(f"  {model_name} ({downloads:,} downloads)")
        else:
            lines.append(f"  {model_name} ({downloads:,} downloads)")
            lines.append(f"      Error: {error or 'No error'}")
    
    return '\n'.join(lines)

# Generate and print report
report = generate_report()
print(report)

# Save to file
output_file = f'health_report_{TARGET_PROBLEM.replace("-", "_")}.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\n{'='*70}")
print(f"Report saved to: {output_file}")