import sqlite3
from collections import defaultdict

# Connect to database
DATABASE_PATH = r'' # Path to your database
conn = sqlite3.connect(DATABASE_PATH)
cursor = conn.cursor()
# Query to get total unique problem types
cursor.execute("SELECT COUNT(DISTINCT problem) FROM Models")
total_problem_types = cursor.fetchone()[0]

print(f"Total Problem Types in Database: {total_problem_types}")
print()
# Query to get health status by problem type
query = """
SELECT 
    problem,
    health_status,
    COUNT(*) as count
FROM Models
GROUP BY problem, health_status
ORDER BY problem, count DESC
"""

cursor.execute(query)
results = cursor.fetchall()

# Organize data by problem type
problem_stats = defaultdict(lambda: {
    'total': 0,
    'tested': 0,
    'ok': 0,
    'failed': 0,
    'untested': 0,
    'failure_types': defaultdict(int)
})

for problem, status, count in results:
    problem_stats[problem]['total'] += count
    
    if status is None:
        problem_stats[problem]['untested'] += count
    else:
        problem_stats[problem]['tested'] += count
        
        if status == 'OK':
            problem_stats[problem]['ok'] += count
        else:
            problem_stats[problem]['failed'] += count
            problem_stats[problem]['failure_types'][status] += count

# Generate Report
print("=" * 80)
print("MODEL HEALTH STATUS REPORT BY PROBLEM TYPE")
print("=" * 80)
print()

# Overall Summary
total_models = sum(stats['total'] for stats in problem_stats.values())
total_tested = sum(stats['tested'] for stats in problem_stats.values())
total_ok = sum(stats['ok'] for stats in problem_stats.values())
total_failed = sum(stats['failed'] for stats in problem_stats.values())
total_untested = sum(stats['untested'] for stats in problem_stats.values())

print(f"OVERALL SUMMARY:")
print(f"  Total Models in Database: {total_models:,}")
print(f"  Models Tested: {total_tested:,} ({total_tested/total_models*100:.1f}%)")
print(f"  Models Passed (OK): {total_ok:,} ({total_ok/total_tested*100:.1f}% of tested)" if total_tested > 0 else "  Models Passed (OK): 0")
print(f"  Models Failed: {total_failed:,} ({total_failed/total_tested*100:.1f}% of tested)" if total_tested > 0 else "  Models Failed: 0")
print(f"  Models Untested: {total_untested:,} ({total_untested/total_models*100:.1f}%)")
print()
print("=" * 80)
print()

# Detailed report by problem type
for problem in sorted(problem_stats.keys()):
    stats = problem_stats[problem]
    
    print(f"PROBLEM TYPE: {problem}")
    print("-" * 80)
    print(f"  Total Models: {stats['total']:,}")
    print(f"  Tested: {stats['tested']:,} ({stats['tested']/stats['total']*100:.1f}%)")
    print(f"  Untested: {stats['untested']:,} ({stats['untested']/stats['total']*100:.1f}%)")
    
    if stats['tested'] > 0:
        print(f"  Status OK: {stats['ok']:,} ({stats['ok']/stats['tested']*100:.1f}% of tested)")
        print(f"  Status FAILED: {stats['failed']:,} ({stats['failed']/stats['tested']*100:.1f}% of tested)")
        
        if stats['failure_types']:
            print(f"  Failure Types:")
            for failure_type, count in sorted(stats['failure_types'].items(), key=lambda x: x[1], reverse=True):
                print(f"    - {failure_type}: {count:,} models")
    
    print()

# Save to file
with open('health_report.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("MODEL HEALTH STATUS REPORT BY PROBLEM TYPE\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"OVERALL SUMMARY:\n")
    f.write(f"  Total Models in Database: {total_models:,}\n")
    f.write(f"  Models Tested: {total_tested:,} ({total_tested/total_models*100:.1f}%)\n")
    f.write(f"  Models Passed (OK): {total_ok:,} ({total_ok/total_tested*100:.1f}% of tested)\n" if total_tested > 0 else "  Models Passed (OK): 0\n")
    f.write(f"  Models Failed: {total_failed:,} ({total_failed/total_tested*100:.1f}% of tested)\n" if total_tested > 0 else "  Models Failed: 0\n")
    f.write(f"  Models Untested: {total_untested:,} ({total_untested/total_models*100:.1f}%)\n")
    f.write("\n" + "=" * 80 + "\n\n")
    
    for problem in sorted(problem_stats.keys()):
        stats = problem_stats[problem]
        
        f.write(f"PROBLEM TYPE: {problem}\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Total Models: {stats['total']:,}\n")
        f.write(f"  Tested: {stats['tested']:,} ({stats['tested']/stats['total']*100:.1f}%)\n")
        f.write(f"  Untested: {stats['untested']:,} ({stats['untested']/stats['total']*100:.1f}%)\n")
        
        if stats['tested'] > 0:
            f.write(f"  Status OK: {stats['ok']:,} ({stats['ok']/stats['tested']*100:.1f}% of tested)\n")
            f.write(f"  Status FAILED: {stats['failed']:,} ({stats['failed']/stats['tested']*100:.1f}% of tested)\n")
            
            if stats['failure_types']:
                f.write(f"  Failure Types:\n")
                for failure_type, count in sorted(stats['failure_types'].items(), key=lambda x: x[1], reverse=True):
                    f.write(f"    - {failure_type}: {count:,} models\n")
        
        f.write("\n")

print("=" * 80)
print("Report saved to 'health_report.txt'")
print("=" * 80)