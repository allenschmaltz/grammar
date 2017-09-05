"""

This script reads in the original source and the generated target (with diff tags, which may be malformed), performs
a post-hoc correction to fix changes that occur outside of insertion tags,
and then saves the result.

"""

import sys
import argparse

import string
import codecs
from os import path
import random
from collections import defaultdict
import numpy as np


INS_START_SYM = "<ins>"
INS_END_SYM = "</ins>"
DEL_START_SYM = "<del>"
DEL_END_SYM = "</del>"




def get_lines(filepath_with_name): 
    lines = []

    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            lines.append(line.strip().split())
    return lines

    
def save_lines(filename_with_path, list_of_lists):   
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        f.writelines(list_of_lists)
            

def remove_whitespace_from_line(line_string):

    line = line_string.strip().split()
    tok_line = []
    for token in line:
        if token not in string.whitespace:
            tok_line.append(token)
    return " ".join(tok_line)
            
def main(arguments):

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-g', '--generated_file', help="The generated file (with diffs).")
    parser.add_argument('-s', '--source', help="The source file.")    
    parser.add_argument('-o', '--output_file', help="The file in which to save the generated lines \
        that have been fixed (with diffs).")
    
    args = parser.parse_args(arguments)
    
    generated_target_file = args.generated_file
    source_file = args.source
    output_file = args.output_file
    
    generated_lines = get_lines(generated_target_file)
    source_lines = get_lines(source_file)
    
    if len(generated_lines) != len(source_lines):
        print "WARNING: len(generated_lines) != len(source_lines)"
    
    output = []
    

    for source_line, generated_line in zip(source_lines, generated_lines):
    
        s = list(source_line)[::-1]
        g = list(generated_line)[::-1]
        gen_fixed = []
        

        if len(s) == 0:
            output.append(" ".join(generated_line) + "\n")
            continue
        
        if len(g) == 0:
            output.append(" ".join(source_line) + "\n")
            continue            
            
        while True:
            if len(s) == 0 and len(g) == 0:
                break
    
            try:
                g_token = g.pop()
            except:
                g_token = None
                
            if g_token == INS_START_SYM:
                # add tokens until the closing symbol is reached
                gen_fixed.append(g_token)
                while True:
                    if len(g) > 0:
                        next_token = g[-1]
                        if next_token not in [INS_START_SYM, INS_END_SYM, DEL_START_SYM, DEL_END_SYM]:
                            gen_fixed.append(next_token)
                            g.pop()
                        else:
                            # non-end tags are replaced with the proper end tag
                            gen_fixed.append(INS_END_SYM)
                            g.pop()
                            break
                    else:
                        # add the end tag
                        gen_fixed.append(INS_END_SYM)
                        break
            elif g_token == INS_END_SYM:
                # an errant tag, so ignore
                pass
            elif g_token == DEL_START_SYM:
                gen_fixed.append(g_token)
                                    
                # add tokens from source until the closing symbol is reached
                while True: # need to advance source and target
                    if len(g) > 0:
                        next_token = g[-1]
                        
                        if next_token not in [INS_START_SYM, INS_END_SYM, DEL_START_SYM, DEL_END_SYM]:
                            # advance source
                            try:  
                                s_token = s.pop()
                            except:
                                s_token = None
                                gen_fixed.append(DEL_END_SYM)
                                g.pop()
                                break                                  
        
                            gen_fixed.append(s_token)
                            g.pop()
                        else:
                            # non-end tags are replaced with the proper end tag
                            gen_fixed.append(DEL_END_SYM)
                            g.pop()
                            break                            
                    else:
                        # add the end tag
                        gen_fixed.append(DEL_END_SYM)
                        break          
            elif g_token == DEL_END_SYM:
                # an errant tag, so ignore
                pass
            else:                    
                # advance source
                try:  
                    s_token = s.pop()
                    gen_fixed.append(s_token)
                except:
                    s_token = None
                    g = []
                    
            
        output.append(" ".join(gen_fixed) + "\n")


    save_lines(output_file, output)
    
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

