import os

files_to_include = ['main.py', 'database.py', 'train_model.py', 'mock_portal.html', 'widget.js']
output_file = 'complete_project_code.md'

with open(output_file, 'w', encoding='utf-8') as outfile:
    outfile.write('# AWIS Project Source Code\n\n')
    for fname in files_to_include:
        if os.path.exists(fname):
            outfile.write(f'## {fname}\n')
            ext = fname.split('.')[-1]
            lang = 'python' if ext == 'py' else 'html' if ext == 'html' else 'javascript'
            outfile.write(f'```{lang}\n')
            with open(fname, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
            outfile.write('\n```\n\n')

print(f'Successfully created {output_file}')
