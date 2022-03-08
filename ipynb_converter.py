import nbformat
from nbconvert import PythonExporter

def convertNotebook(notebookPath, modulePath):

  with open(notebookPath) as fh:
    nb = nbformat.reads(fh.read(), nbformat.NO_CONVERT)

  exporter = PythonExporter()
  source, meta = exporter.from_notebook_node(nb)

  with open(modulePath, 'w+') as fh:
    fh.writelines(source)

file_in = input("Enter input file's name:")
file_out = input("Enter output file's name:")

import os

cwd = os.getcwd()

in_path = cwd+"/"+file_in
out_path = cwd+"/"+file_out

convertNotebook(in_path, out_path)
