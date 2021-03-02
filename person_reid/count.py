import os

files = folders = 0
path = '.'
for _, dirnames, filenames in os.walk(path):
  # ^ this idiom means "we won't be using this value"
    files += len(filenames)
    folders += len(dirnames)

print(files)
print(folders)