
import os
import sys
import argparse
from phonemizer.phonemize import phonemize

njobs = 4

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--in_file", type=str)
  parser.add_argument("--out_file", type=str)

  args = parser.parse_args()

  fout = open(args.out_file, 'w')

  with open(args.in_file, 'r', encoding="utf-8") as fin:
    for line in fin:
      lines = line.split('\t')

      try:
        out = phonemize(lines, language='en-us', backend='espeak', strip=True, njobs=njobs)
        if len(out) == 2:
          fout.write(lines[0] + '\t' + out[0] + '\t' + lines[1] + '\t' + out[1] + '\n')
      except AttributeError:
        continue

  fout.close()

if __name__ == '__main__':
  main()
