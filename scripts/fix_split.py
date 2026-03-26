"""Quick fix: change test fold from 5 to 4 in step1_prepare_data.py"""
import os
path = os.path.expanduser("~/ICBHIMamba-JBHI/scripts/step1_prepare_data.py")
with open(path, "r") as f:
    content = f.read()
content = content.replace("fold == 5", "fold == 4")
content = content.replace("Test fold = 5", "Test fold = 4 (folds are 0-4)")
with open(path, "w") as f:
    f.write(content)
print("Fixed: test fold changed from 5 to 4")
