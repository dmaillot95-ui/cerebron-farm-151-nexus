import json,pathlib
R=pathlib.Path(__file__).resolve().parent
req=[R/"architecture"/"NEXUS_ARCHITECTURE_V1.json",R/"memory"/"NEXUS_MEMORY_CONTRACT_V1.json",R/"benchmarks"/"NEXUS_BENCHMARK_V1.json",R/"config"/"model-candidates.json",R/"training"/"NEXUS_TRAINING_READINESS_GATES_V1.json"]
missing=[str(p.relative_to(R)) for p in req if not p.exists()]
if missing:print(json.dumps({"status":"FAIL","missing":missing}));raise SystemExit(1)
b=json.loads((R/"benchmarks"/"NEXUS_BENCHMARK_V1.json").read_text());m=json.loads((R/"memory"/"NEXUS_MEMORY_CONTRACT_V1.json").read_text());a=json.loads((R/"architecture"/"NEXUS_ARCHITECTURE_V1.json").read_text())
assert b["m6_rule"]=="M6_NEVER_IN_TRAIN";assert "M6_COLD_BENCHMARK" in m["training_forbidden"];assert a["runtime_changed"] is False
print(json.dumps({"status":"PASS","name":"NEXUS","m6_guard":True,"runtime_changed":False}))
