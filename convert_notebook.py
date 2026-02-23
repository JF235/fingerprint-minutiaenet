import json
import os

with open('/home/joaocontreras/work/minutiaenet/Demo_notebooks/demo_CoarseNet.ipynb', 'r') as f:
    nb = json.load(f)

code = []
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        # Remove magic commands
        source = [line for line in cell['source'] if not line.startswith('%')]
        code.extend(source)
        code.append('\n\n')

src = "".join(code)

# Make output_dir = '../output_old'
src = src.replace("output_dir = '../output_CoarseNet/'+datetime.now().strftime('%Y%m%d-%H%M%S')", 
                  "output_dir = '../output_old'")

# Ensure output_old exists
src = "import os\nif not os.path.exists('../output_old'): os.makedirs('../output_old')\n" + src

with open('/home/joaocontreras/work/minutiaenet/Demo_notebooks/run_demo_CoarseNet.py', 'w') as f:
    f.write(src)
    
print("Successfully generated run_demo_CoarseNet.py")
