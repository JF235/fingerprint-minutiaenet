import json
import os

with open('/home/joaocontreras/work/minutiaenet/Demo_notebooks/demo_FineNet.ipynb', 'r') as f:
    nb = json.load(f)

code = []
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        # Remove magic commands
        source = [line for line in cell['source'] if not line.startswith('%')]
        code.extend(source)
        code.append('\n\n')

src = "".join(code)

with open('/home/joaocontreras/work/minutiaenet/Demo_notebooks/run_demo_FineNet.py', 'w') as f:
    f.write(src)
    
print("Successfully generated run_demo_FineNet.py")
