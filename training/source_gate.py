ALLOWED_CLASSES={"M4_GOLD","M2_REPLAY_PROMOTED","M7_EVIDENCE_METADATA"}
FORBIDDEN_CLASSES={"M6_COLD_BENCHMARK","RAW_AGORA","UNVERIFIED_EXTERNAL","SECRETS"}
def admit(record):
 cls=str(record.get("memory_class","")).upper();path=str(record.get("object_path","")).upper();validated=bool(record.get("validated",False));promoted=bool(record.get("promoted",False))
 if cls in FORBIDDEN_CLASSES:return False,"FORBIDDEN_CLASS"
 if "/M6/" in path:return False,"M6_PATH_FORBIDDEN"
 if cls=="M4_GOLD":return (validated,"GOLD_VALIDATED" if validated else "GOLD_NOT_VALIDATED")
 if cls=="M2_REPLAY_PROMOTED":return (validated and promoted,"REPLAY_PROMOTED" if validated and promoted else "REPLAY_NOT_PROMOTED")
 if cls=="M7_EVIDENCE_METADATA":return (validated,"EVIDENCE_METADATA" if validated else "EVIDENCE_NOT_VALIDATED")
 return False,"CLASS_NOT_ALLOWLISTED"
