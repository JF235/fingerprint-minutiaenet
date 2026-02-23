import json
import os

with open('/home/joaocontreras/work/minutiaenet/pytorch/examples/example_extraction.ipynb', 'r') as f:
    nb = json.load(f)

code = []
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = [line for line in cell['source'] if not line.startswith('%')]
        code.extend(source)
        code.append('\n\n')

src = "".join(code)

with open('/home/joaocontreras/work/minutiaenet/pytorch/examples/run_extraction.py', 'w') as f:
    f.write(src)
    
print("Successfully generated run_extraction.py")
