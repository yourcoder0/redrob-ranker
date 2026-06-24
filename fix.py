import csv
rows = list(csv.DictReader(open('submission_v2.csv')))
rows.sort(key=lambda r: (-float(r['score']), r['candidate_id']))
with open('submission_v3.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
    for i, r in enumerate(rows, 1):
        writer.writerow([r['candidate_id'], i, r['score'], r['reasoning']])
print('Fixed!')