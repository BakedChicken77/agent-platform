import os
import re

repo_file = 'agent_platform.md'
with open(repo_file, 'r', encoding='utf-8') as f:
    lines = f.read().splitlines()

prefix = 'C:\\Users\\Steve.Long\\Downloads\\agent-service-toolkit-main\\agent-service-toolkit-main\\'

files = {}
inside = False
path = None
content = []
nested = 0

i = 0
while i < len(lines):
    line = lines[i]
    if not inside:
        if line.startswith('```'):
            if i + 1 < len(lines):
                next_line = lines[i+1]
                m1 = re.match(r'#\s*(.*)', next_line)
                m2 = re.match(r'File:\s*\*\s*(.*?)\s*\*', next_line)
                possible_path = None
                if m1:
                    possible_path = m1.group(1).strip()
                elif m2:
                    possible_path = m2.group(1).strip()
                if possible_path:
                    if possible_path.startswith(prefix):
                        possible_path = possible_path[len(prefix):]
                    possible_path = possible_path.replace('\\', '/')
                    path = possible_path
                    inside = True
                    nested = 0
                    content = []
                    i += 1  # skip path line
    else:
        if line.startswith('```'):
            if line.strip() == '```':
                if nested == 0:
                    files.setdefault(path, []).append('\n'.join(content))
                    inside = False
                    path = None
                    content = []
                else:
                    nested -= 1
                    content.append(line)
            else:
                nested += 1
                content.append(line)
        else:
            content.append(line)
    i += 1

for path, contents in files.items():
    full = '\n'.join(contents)
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(full + '\n')

print(f'Created {len(files)} files')
