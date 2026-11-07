import os

# 1. Crear carpetas principales
folders = [
    'configs', 'data/raw', 'data/processed', 'models',
    'training', 'stream_ingestion', 'inference',
    'triage_alerts', 'scripts', 'tests'
]
for f in folders:
    os.makedirs(f, exist_ok=True)

# 2. Convertir carpetas en módulos de Python
packages = ['data', 'models', 'training', 'stream_ingestion', 'inference', 'triage_alerts']
for p in packages:
    with open(f'{p}/__init__.py', 'w') as f: pass

# 3. Retener carpetas vacías en Git
for k in ['configs', 'data/raw', 'data/processed', 'scripts', 'tests']:
    with open(f'{k}/.gitkeep', 'w') as f: pass

# 4. Crear los archivos en blanco para el Vision Transformer
files = [
    'models/patch_embedding.py', 'models/attention.py',
    'models/vit_backbone.py', 'models/temporal_aggregator.py',
    'models/anomaly_detector.py', 'requirements.txt', 'Dockerfile'
]
for file in files:
    with open(file, 'w') as f: pass

# 5. Proteger archivos pesados (.gitignore) y documentar (README)
with open('.gitignore', 'w') as f:
    f.write("venv/\n.venv/\n__pycache__/\n*.py[cod]\n.idea/\n.vscode/\ndata/raw/*\n!data/raw/.gitkeep\ndata/processed/*\n!data/processed/.gitkeep\n*.mp4\n*.pt\n*.pth\n*.onnx\nmlruns/\nlogs/\n")

with open('README.md', 'w', encoding='utf-8') as f:
    f.write("# Detección de Patrones Delictivos mediante Vision Transformers (ViT)\n")

print('Estructura creada correctamente.')