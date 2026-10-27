import json, math

with open('results/stress_test_manifest.json') as f:
    manifest = json.load(f)

# Compute AUC and Resilience for each channel
for ch_key, ch_data in manifest.items():
    steps = ch_data['steps']
    methods = ['global', 'otsu', 'adaptive', 'edge']
    ch_data['stats'] = {'auc': {}, 'resilience': {}}
    for m in methods:
        ious = [s['methods'][m]['iou'] for s in steps]
        # Trapezoidal AUC normalized to [0, 1] across the 7 segments
        auc = sum((ious[i] + ious[i+1]) / 2 for i in range(len(ious)-1)) / (len(ious)-1)
        ch_data['stats']['auc'][m] = round(float(auc), 3)
        resilience = (ious[-1] / max(ious[0], 0.001)) * 100.0
        ch_data['stats']['resilience'][m] = round(float(resilience), 1)

with open('results/stress_test_manifest_enriched.json', 'w') as f:
    json.dump(manifest, f, indent=2)

print('Enriched manifest created successfully!')
