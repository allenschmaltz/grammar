"""
This script takes an input file containing <ins> and <del> diff tags and removes
the tags (and text within del tags) to generate the (corrected) natural language sentences.


Orphaned start or end tags are simply deleted. However, it is recommended
to provide this script with well-formed diff tags. Use post_hoc_fix_standard_tags.py to fix the output prior to running
this script.

"""

import sys
import argparse

import string
import codecs
from os import path
import random
from collections import defaultdict
import numpy as np

random.seed(1776)

EOS_SYM = "<eos>"
NUMERIC_SYM = "N"
LOW_COUNT_SYM = "unk"
LOW_COUNT_SYM_UPPER = "UNK"

INS_START_SYM = "<ins>"
INS_END_SYM = "</ins>"
DEL_START_SYM = "<del>"
DEL_END_SYM = "</del>"


def get_lines(filepath_with_name): 
    """
    Returns lines as lists
    """
    
    lines = []
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            lines.append(line.strip().split())
    return lines

def save_lines(filename_with_path, list_of_lists):   
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        f.writelines(list_of_lists)
        

def find_next_del_end_sym(start_idx, line):
    """
    line: string
    Assumes no insert tags remain in the line
    
    Returns: the index of the next DEL_END_SYM, or -1
    """
    next_del_start_sym = line[start_idx+1:].find(DEL_START_SYM)
    next_del_end_sym = line[start_idx+1:].find(DEL_END_SYM)
    
    # for clarity, the cases are separated below:
    if next_del_end_sym == -1: # no end symbol exists
        return -1
    if next_del_start_sym == -1: # no start symbol exists (and an end symbol does exist)
        return next_del_end_sym + start_idx+1
    else: 
        # both a start symbol and an end symbol exist:
        # need to check that the end sym occurs before the next start sym
        if next_del_end_sym >= next_del_start_sym:
            return -1
            
    return next_del_end_sym + start_idx+1
       
    
def remove_syms(line_id, line):

    
    corrected_nosyms_line = line
    if (INS_START_SYM in corrected_nosyms_line) or (INS_END_SYM in corrected_nosyms_line):
        #if not ((INS_START_SYM in corrected_nosyms_line) and (INS_END_SYM not in corrected_nosyms_line)):
        #    print "Warning: orphaned insert tag in line %d" % line_id
        corrected_nosyms_line = string.replace(corrected_nosyms_line, INS_START_SYM, "")
        corrected_nosyms_line = string.replace(corrected_nosyms_line, INS_END_SYM, "")      
    assert INS_START_SYM not in corrected_nosyms_line and INS_END_SYM not in corrected_nosyms_line
    if DEL_START_SYM in line:
        while corrected_nosyms_line.find(DEL_START_SYM) != -1:
            del_start_idx = corrected_nosyms_line.find(DEL_START_SYM)
            next_del_end_idx = find_next_del_end_sym(del_start_idx, corrected_nosyms_line)
            #del_end_idx = corrected_nosyms_line.find(DEL_END_SYM)
            if next_del_end_idx == -1: # if the delete start symbol is orphaned, then just remove (this occurrence only)
                corrected_nosyms_line = string.replace(corrected_nosyms_line, DEL_START_SYM, "", 1)
            else:
                corrected_nosyms_line = string.replace(corrected_nosyms_line, corrected_nosyms_line[del_start_idx:next_del_end_idx+len(DEL_END_SYM)], "")
    # delete any remaining orphaned end delete symbols:
    corrected_nosyms_line = string.replace(corrected_nosyms_line, DEL_END_SYM, "")
    assert DEL_START_SYM not in corrected_nosyms_line and DEL_END_SYM not in corrected_nosyms_line 
    return corrected_nosyms_line


def remove_syms_from_lines(lines):
    """
    lines: A list (for each sentence) of token lists 
    
    returns: A list of strings 
    """
    nosyms_lines = []
    for i, line in enumerate(lines):
        line = remove_syms(i, " ".join(line))
        nosyms_lines.append(line)
    return nosyms_lines
    
        
def remove_whitespace(lines):
    tok_lines = []
    for line in lines:
        line = line.strip().split()
        tok_line = []
        for token in line:
            if token not in string.whitespace:
                tok_line.append(token)
        tok_lines.append(" ".join(tok_line) + "\n")
    return tok_lines
               
                 
def main(arguments):

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--input_tag_file', help="The input file (with well-formed diff tags).")
    parser.add_argument('-o', '--output_file', help="The file in which to save the sentences with diff tags removed.")
    
    args = parser.parse_args(arguments)
    
    input_tag_file = args.input_tag_file
    output_file = args.output_file
    
    tagged_tokenized_lines = get_lines(input_tag_file)
    nosyms_lines = remove_syms_from_lines(tagged_tokenized_lines)
    output_lines = remove_whitespace(nosyms_lines)
    
    save_lines(output_file, output_lines)
    
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

