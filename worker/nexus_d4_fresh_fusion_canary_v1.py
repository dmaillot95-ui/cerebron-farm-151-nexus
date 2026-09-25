#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,pathlib,re,resource,time,urllib.request,gc
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM

PRECHECK_REPO="dmaillot95-ui/cerebron-omega-ai"
PRECHECK_COMMIT="32aa8415bc2ac9d7927dfa003767dd8478fea1f1"
PRECHECK_PATH="receipts/wave3a-d4/nexus/nexus-d4-complementarity.json"

MODELS=[
  ("qwen","Qwen/Qwen3-4B","1cfa9a7208912126459214e8b04321603b3df60c"),
  ("smol","HuggingFaceTB/SmolLM3-3B","a07cc9a04f16550a088caea529712d1d335b0ac1")
]

TASKS=[
 {"id":"N01","domain":"entity_linking","question":"Record A names farm F115 as 'vector-semantic-memory'. Record B names repository 'cerebron-farm-115-vector-semantic-memory'. What is the correct link?","options":{"A":"Same canonical farm entity","B":"Different farms","C":"One is a model weight","D":"Insufficient because names differ"},"expected":"A"},
 {"id":"N02","domain":"entity_linking","question":"ELYSIUM and ELYSION appear in the same registry. How should NEXUS link them?","options":{"A":"Merge as spelling variants","B":"Keep as distinct AI entities","C":"Delete ELYSIUM","D":"Treat both as F152"},"expected":"B"},
 {"id":"N03","domain":"entity_linking","question":"F152 is named CEREBRON RDX EXCHANGE and its repository is cerebron-rdx-exchange. What relation should be stored?","options":{"A":"Repository implements the same F152 product entity","B":"Unrelated entities","C":"Repository is F118","D":"Repository is an M6 benchmark"},"expected":"A"},

 {"id":"N04","domain":"dependency_tracking","question":"F118 M4 GOLD requires F72 PASS. Current global F72 is FAIL. What is the dependency result?","options":{"A":"Promote to GOLD anyway","B":"Block F118 GOLD promotion","C":"Train first then recheck F72","D":"Use M6 as substitute evidence"},"expected":"B"},
 {"id":"N05","domain":"dependency_tracking","question":"NEXUS D4 complementarity precheck has positive oracle potential, but no real fusion gain has yet been measured. What remains true?","options":{"A":"D5 is automatically PASS","B":"D4 is fully promoted","C":"A real fusion canary is still required before later gates","D":"Weights are now trained"},"expected":"C"},
 {"id":"N06","domain":"dependency_tracking","question":"A proposed training dataset contains rows copied from M6 cold benchmarks. How should the route resolve?","options":{"A":"Allow because M6 is accurate","B":"Deny training and repair the dataset","C":"Promote directly to M4 GOLD","D":"Count the rows twice for confidence"},"expected":"B"},

 {"id":"N07","domain":"tool_routing","question":"A query needs semantic retrieval over RDX memory. Which memory farm is the primary route?","options":{"A":"F114 canonical index","B":"F115 vector semantic memory","C":"F119 evidence archive","D":"F120 governor"},"expected":"B"},
 {"id":"N08","domain":"tool_routing","question":"A claim needs its immutable evidence and provenance receipt fetched. Which farm is the primary archive route?","options":{"A":"F116","B":"F117","C":"F119","D":"F120"},"expected":"C"},
 {"id":"N09","domain":"tool_routing","question":"Two RDX records may be duplicates and their source lineages need fusion. Which farm should handle that function?","options":{"A":"F114","B":"F115","C":"F116","D":"F118"},"expected":"C"},

 {"id":"N10","domain":"evidence_provenance","question":"Two workflow runs repeat the same underlying evidence and lineage. How many independent confirmations do they create?","options":{"A":"Two","B":"One independent confirmation each","C":"Zero new independent confirmations beyond the shared evidence","D":"Four because two models ran"},"expected":"C"},
 {"id":"N11","domain":"evidence_provenance","question":"Which evidence bundle is required before claiming neural learning from a new adapter?","options":{"A":"A prompt change only","B":"weights_or_adapter_changed plus artifact hash, dataset hash, training run and cold post-test","C":"A successful RAG lookup","D":"A workflow success badge"},"expected":"B"},
 {"id":"N12","domain":"evidence_provenance","question":"A synthetic retrieval canary scores 24/24. What may NEXUS conclude?","options":{"A":"Production RAG quality is proven","B":"The scoped synthetic canary passed, but production quality remains unproven","C":"The model is now trained","D":"F72 global automatically passes"},"expected":"B"},

 {"id":"N13","domain":"contradiction_routing","question":"Claim A says simulation success is physical validation. Claim B says simulation is not a physical test. Under CEREBRON rules, which route is admissible?","options":{"A":"Accept A","B":"Accept B and preserve the limitation","C":"Average both claims","D":"Discard both without record"},"expected":"B"},
 {"id":"N14","domain":"contradiction_routing","question":"Two sources make incompatible technical claims and both have provenance. What should knowledge fusion do first?","options":{"A":"Silently average them","B":"Preserve the contradiction and route it for audit","C":"Choose the newer source automatically","D":"Count both as agreement"},"expected":"B"},
 {"id":"N15","domain":"contradiction_routing","question":"A scoped audit is positive but the global F72 gate remains FAIL. What happens to M4 GOLD?","options":{"A":"Promote because the scoped audit passed","B":"Remain blocked by global F72","C":"Train then rollback if needed","D":"Use consensus instead of F72"},"expected":"B"},

 {"id":"N16","domain":"graph_retrieval","question":"For the configured immediate RDX memory route, what is the first memory farm after F152 ingress?","options":{"A":"F114","B":"F117","C":"F118","D":"F120"},"expected":"A"},
 {"id":"N17","domain":"graph_retrieval","question":"After NEXUS/router receives a question, which node is first in the configured RDX read path before semantic retrieval?","options":{"A":"F119 evidence archive","B":"F114 canonical index","C":"F118 GOLD","D":"F117 replay"},"expected":"B"},
 {"id":"N18","domain":"graph_retrieval","question":"A validated object is specifically a replayable simulation recipe with seed and versioned configuration. Which conditional memory farm should receive it?","options":{"A":"F114","B":"F116","C":"F117","D":"F118"},"expected":"C"}
]

CRITICAL_DOMAINS={"dependency_tracking","evidence_provenance","contradiction_routing"}

def fetch_precheck():
    url=f"https://raw.githubusercontent.com/{PRECHECK_REPO}/{PRECHECK_COMMIT}/{PRECHECK_PATH}"
    with urllib.request.urlopen(url,timeout=30) as r:
        raw=r.read()
    d=json.loads(raw)
    assert d["ai_id"]=="NEXUS"
    assert d["decision"]=="CONTINUE_D4_FUSION_CANARY"
    assert d["oracle_gain_over_best_single"]>0
    return d,hashlib.sha256(raw).hexdigest()

def build_prompt(task):
    opts="\n".join(f"{k}. {v}" for k,v in task["options"].items())
    return (
      "NEXUS fresh evaluation task. Choose exactly one option letter A, B, C, or D. "
      "Do not explain and do not use external tools. /no_think\n"
      f"Question: {task['question']}\n{opts}\nAnswer:"
    )

def build_judge_prompt(task,a,b):
    opts="\n".join(f"{k}. {v}" for k,v in task["options"].items())
    return (
      "NEXUS fusion judge. Two candidate answers were produced independently. "
      f"The candidates are {a} and {b}. Re-evaluate the task and reply with exactly one of those two letters. "
      "Do not explain. /no_think\n"
      f"Question: {task['question']}\n{opts}\nCandidate letters: {a}, {b}\nAnswer:"
    )

def parse_letter(s):
    m=re.search(r"\b([A-D])\b",s.upper())
    return m.group(1) if m else None

def run_model(model_id,revision,prompts,max_new_tokens=5):
    tok=AutoTokenizer.from_pretrained(model_id,revision=revision)
    if tok.pad_token_id is None: tok.pad_token_id=tok.eos_token_id
    tok.padding_side="left"
    rendered=[]
    for p in prompts:
        messages=[{"role":"system","content":"Return only the requested option letter."},{"role":"user","content":p}]
        try:
            s=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        except TypeError:
            s=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        rendered.append(s)
    x=tok(rendered,return_tensors="pt",padding=True)
    model=AutoModelForCausalLM.from_pretrained(
        model_id,revision=revision,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True
    )
    model.eval()
    t0=time.time()
    with torch.inference_mode():
        y=model.generate(
          **x,max_new_tokens=max_new_tokens,do_sample=False,
          pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id
        )
    cut=x["input_ids"].shape[1]
    outs=[tok.decode(row[cut:],skip_special_tokens=True).strip() for row in y]
    elapsed=time.time()-t0
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    del model,x,y
    gc.collect()
    return outs,elapsed,rss

def metrics(preds):
    correct=0; crit=0
    for t,p in zip(TASKS,preds):
        ok=p==t["expected"]
        correct+=int(ok)
        if t["domain"] in CRITICAL_DOMAINS and not ok: crit+=1
    return {"correct":correct,"total":len(TASKS),"accuracy":correct/len(TASKS),"critical_errors":crit}

def main():
    precheck,precheck_file_sha=fetch_precheck()
    prompts=[build_prompt(t) for t in TASKS]

    base={}; timing={}; rss={}
    for slug,mid,rev in MODELS:
        raw,elapsed,maxrss=run_model(mid,rev,prompts)
        base[slug]=[parse_letter(x) for x in raw]
        timing[slug+"_base_s"]=round(elapsed,3); rss[slug+"_base_max_rss_kb"]=maxrss

    disagreements=[i for i,(a,b) in enumerate(zip(base["qwen"],base["smol"])) if a!=b]
    judge_preds={"qwen":{},"smol":{}}
    if disagreements:
        jp=[build_judge_prompt(TASKS[i],base["qwen"][i],base["smol"][i]) for i in disagreements]
        for slug,mid,rev in MODELS:
            raw,elapsed,maxrss=run_model(mid,rev,jp)
            timing[slug+"_judge_s"]=round(elapsed,3); rss[slug+"_judge_max_rss_kb"]=maxrss
            for idx,out in zip(disagreements,raw):
                letter=parse_letter(out)
                if letter in {base["qwen"][idx],base["smol"][idx]}:
                    judge_preds[slug][idx]=letter
                else:
                    judge_preds[slug][idx]=None

    fused=[]; rows=[]; helpful=harmful=judge_consensus=fallbacks=0
    for i,t in enumerate(TASKS):
        q=base["qwen"][i]; s=base["smol"][i]
        if q==s and q is not None:
            f=q; route="AGREEMENT"
        else:
            jq=judge_preds["qwen"].get(i); js=judge_preds["smol"].get(i)
            if jq is not None and jq==js:
                f=jq; route="DUAL_JUDGE_CONSENSUS"; judge_consensus+=1
            else:
                f=q; route="QWEN_FALLBACK"; fallbacks+=1
        fused.append(f)
        qok=q==t["expected"]; fok=f==t["expected"]
        helpful+=int((not qok) and fok)
        harmful+=int(qok and (not fok))
        rows.append({
          "id":t["id"],"domain":t["domain"],"expected":t["expected"],
          "qwen":q,"smol":s,"qwen_judge":judge_preds["qwen"].get(i),
          "smol_judge":judge_preds["smol"].get(i),"fusion":f,"route":route,
          "qwen_correct":qok,"smol_correct":s==t["expected"],"fusion_correct":fok
        })

    qm=metrics(base["qwen"]); sm=metrics(base["smol"]); fm=metrics(fused)
    best=max(qm["correct"],sm["correct"])
    if qm["correct"]>=sm["correct"]:
        best_critical=qm["critical_errors"]; best_model="Qwen/Qwen3-4B"
    else:
        best_critical=sm["critical_errors"]; best_model="HuggingFaceTB/SmolLM3-3B"
    gain=fm["correct"]-best
    critical_delta=best_critical-fm["critical_errors"]
    critical_regression=fm["critical_errors"]>best_critical

    if gain>0 and not critical_regression:
        decision="D4_FRESH_FUSION_PASS_MEASURED_GAIN"
    elif gain==0 and critical_delta>0:
        decision="D4_FRESH_FUSION_PASS_RISK_REDUCTION"
    elif gain<0 or critical_regression:
        decision="D4_FRESH_FUSION_HOLD_REGRESSION"
    else:
        decision="D4_FRESH_FUSION_HOLD_NO_MEASURED_GAIN"

    out={
      "schema":"NEXUS_D4_FRESH_DUAL_CORE_FUSION_CANARY_V1",
      "run_id":int(__import__("os").environ["GITHUB_RUN_ID"]),
      "farm_id":151,"ai_id":"NEXUS",
      "source_precheck":{
        "repo":PRECHECK_REPO,"commit":PRECHECK_COMMIT,"path":PRECHECK_PATH,
        "source_run_id":precheck["run_id"],
        "source_receipt_sha256":precheck["receipt_sha256"],
        "source_file_sha256":precheck_file_sha,
        "oracle_gain_over_best_single":precheck["oracle_gain_over_best_single"]
      },
      "holdout":{
        "kind":"FRESH_SYNTHETIC_INDEPENDENTLY_AUTHORED_NOT_M6_DERIVED",
        "task_count":len(TASKS),
        "taskset_sha256":hashlib.sha256(json.dumps(TASKS,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
        "domains":sorted(set(t["domain"] for t in TASKS)),
        "m6_content_used":False
      },
      "models":[
        {"model_id":MODELS[0][1],"revision":MODELS[0][2]},
        {"model_id":MODELS[1][1],"revision":MODELS[1][2]}
      ],
      "fusion_policy":{
        "frozen_before_execution":True,
        "agreement":"use_agreed_answer",
        "disagreement":"two_models_independently_judge_only_the_two_candidate_letters",
        "judge_consensus":"use_consensus_candidate",
        "otherwise":"fallback_to_qwen_d3_best_single"
      },
      "metrics":{
        "qwen":qm,"smol":sm,"fusion":fm,
        "best_single_model":best_model,
        "best_single_correct":best,
        "fusion_gain_over_best_single":gain,
        "best_single_critical_errors":best_critical,
        "fusion_critical_errors":fm["critical_errors"],
        "critical_error_reduction":critical_delta,
        "critical_domain_regression":critical_regression,
        "prediction_disagreement_count":len(disagreements),
        "judge_consensus_count":judge_consensus,
        "fallback_count":fallbacks,
        "helpful_switches_vs_qwen":helpful,
        "harmful_switches_vs_qwen":harmful
      },
      "timing":timing,"resource":rss,
      "rows":rows,
      "decision":decision,
      "independent_evidence_count":0,
      "two_models_not_two_independent_proofs":True,
      "training_executed":False,
      "weights_changed":False,
      "claim_ceiling":"FRESH_SYNTHETIC_NEXUS_D4_FUSION_CANARY_ONLY_NO_PRODUCTION_ACTIVATION_NO_NEURAL_TRAINING"
    }
    out["receipt_sha256"]=hashlib.sha256(json.dumps(out,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    pathlib.Path("artifacts").mkdir(exist_ok=True)
    pathlib.Path("artifacts/nexus_d4_fresh_fusion_canary_v1.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({"metrics":out["metrics"],"decision":decision,"receipt_sha256":out["receipt_sha256"]},sort_keys=True))

if __name__=="__main__":
    main()
