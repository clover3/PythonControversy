import re

line = open("..\\data\\expandedBigram.txt","r").readline()
new_line = ""
lastChar = 'c'
tab_count = 0
for idx, char in enumerate(line):
    if tab_count == 100:
        if lastChar.isnumeric() and not char.isnumeric() and char != '.' and char != '\t':
            new_line += "\n"
            print("")
            tab_count = 0
    if char=='\t':
        tab_count = tab_count + 1
    new_line += char
    lastChar = char
    print(char, end="")

open("..\\data\\expandedBigram_new.txt", "w").write(new_line)