import sys
import csv

for line in sys.stdin:
    line = line.strip()
    try:
        name, aadhaar, claim, subsidy = line.split(",")
        print(f"{aadhaar}\t1")
    except:
        continue
