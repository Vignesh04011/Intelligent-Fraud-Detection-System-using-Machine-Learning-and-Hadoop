import sys

current_aadhaar = None
count = 0

for line in sys.stdin:
    aadhaar, value = line.strip().split("\t")

    if current_aadhaar == aadhaar:
        count += 1
    else:
        if current_aadhaar and count > 1:
            print(f"{current_aadhaar}\tDUPLICATE")
        current_aadhaar = aadhaar
        count = 1

if current_aadhaar and count > 1:
    print(f"{current_aadhaar}\tDUPLICATE")
