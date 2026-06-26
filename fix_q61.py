with open("PineBioML/rag/clinical_knowledge.py", "r") as f:
    content = f.read()

# Replace Q6.1
old_q6_1 = """"- **Safe medications during pregnancy**: 5-ASA, thiopurines, anti-TNF (stop in 3rd trimester), vedolizumab\\n\""""
new_q6_1 = """"These 5-ASA, thiopurines, anti-TNF (stop in 3rd trimester), vedolizumab medications were safe to be continued.\\n\""""
content = content.replace(old_q6_1, new_q6_1)

with open("PineBioML/rag/clinical_knowledge.py", "w") as f:
    f.write(content)
