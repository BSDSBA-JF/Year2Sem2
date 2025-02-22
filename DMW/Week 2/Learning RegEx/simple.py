import re

text_to_search = '''
abcdefghijklmnopqurtuvwxyz
ABCDEFGHIJKLMNOPQRSTUVWXYZ
1234567890

Ha HaHa

MetaCharacters (Need to be escaped):
. ^ $ * + ? { } [ ] \ | ( )

coreyms.com

321-555-4321
123.555.1234
123*555*1234
800-555-1234
900-555-1234

Mr. Schafer
Mr Smith
Ms Davis
Mrs. Robinson
Mr. T

cat
mat
pat
bat
'''


sentence = 'Start a sentence and then bring it to an end'

'''
pattern = re.compile(r'start', re.I)

matches = pattern.search(sentence)


print(matches)


pattern = re.compile(r'\BHa')

matches = pattern.finditer(text_to_search)

for match in matches:
    print(match)
'''

pattern = re.compile(r'a')

matches = pattern.search(sentence)

print(matches)

"""
# opening the file
with open('data.txt', 'r', encoding = 'utf-8') as f:
    contents = f.read()
    matches = pattern.finditer(contents)

    for match in matches:
        print(match)
"""