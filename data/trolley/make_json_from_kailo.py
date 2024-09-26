'''
           adapted from script by Edoardo Guido (under MIT licsence)
               edoardo.guido.93@gmail.com
                https://edoardoguido.com
'''

import sys, time, json, re
import pandas as pd

input_file = 'the-trolley-problem-whats-the-right-solution-921.txt'
#input_file = 'should-we-adopt-a-universal-language-31699.txt'

input_file = 'god-exists-3491.txt'

with open(input_file, 'r') as fi:
    lines = []
    for line in fi:
        lines.append(line)

# datframe containing each parsed comment
result= pd.DataFrame()

# we remove the first two lines of the text
# as we don't need the header
for line in range(0, 6):
    line=lines.pop(0)
    if line.startswith('1.'):
            tree =  re.search(r"(\d{1,}\.){1,}", line)
            level=0
            
            line=lines.pop(0)
            
            result = pd.DataFrame({
                'assistant': tree.group(),
                'Level': level,
                'category': 'Thesis',
                'user': line,
                'References': [''],
                'Link': ['']
            })
            break




foundNumber=False

line=lines.pop(0)
while not(line.startswith('Sources')):
    # find the tree position the comment is in
    # has format like 1.2.1.3.
    # We use regular expression to identify that format
        
    if not foundNumber:
                
        tree =  re.search(r"(\d{1,}\.){1,}", line)
        
        if tree:
            # find if the comment is Pro or Con
            stance = re.search(r"(Con|Pro)(?::)", line)
            
            # define the hierarchy of the current comment
            # which is based on the tree structure
            parsed = re.findall(r"(\d{1,}(?=\.))+", tree.group())
            level = len(parsed)-1
            print("hej")
            foundNumber=True
        else:
            
            
            pass
    else:
        
        # find the text of the comment
        foundNumber=False
        
        
        
        references = re.findall(r"\[(\d+)\]", line)
        # make a dictionary with the single entry
        # and put it at the end of the list
        if references:
            entry = pd.DataFrame({
                'assistant': tree.group(),
                'Level': level,
                'category': stance.group(1),
                'user': line,
                'References': references,
                'Link': ''
            })
        else:
            entry = pd.DataFrame({
                'assistant': tree.group(),
                'Level': level,
                'category': stance.group(1),
                'user': line,
                'References': [''],
                'Link': ['']
            })

        result = pd.concat([result, entry], ignore_index=True)
        
    #Get the next line
    line=lines.pop(0)
    print('Read in argument: ' + line)

    # Now get the refences
referencedict={}
while lines:
    print('Read in reference: ' +line)
    line=lines.pop(0)
    reference = re.findall(r"\[(\d+)\]", line)
    if reference:
        text = re.split(r"\[\d+\]", line)
        text =''.join(text)
        referencedict[reference[0]]=text
        # Deals with case where pmore text in reference
        prev_reference=reference[0]
    else:
        referencedict[prev_reference]=referencedict[prev_reference]+text 
        

# Fill in column 'Link' with the content of the references dict
for index, row in result.iterrows():
    if row['References']:
        result.at[index, 'Link'] = referencedict[row['References'][0]]

# Save the result to a csv file

result.to_csv('GodTree.csv', index=False)

