#!/usr/bin/env python3
import wget
import os
import pandas as pd
import re

'''
Download the AAIndex from the official website, https://www.genome.jp/ftp/db/community/aaindex/aaindex1
And format it into CSV:
    $ cat AAIndex.csv
    A,R,N,D,C,Q,E,G,H,I,L,K,M,F,P,S,T,W,Y,V,AAIndex
    4.35,4.38,4.75,4.76,4.65,4.37,4.29,3.97,4.63,3.95,4.17,4.36,4.52,4.66,4.44,4.50,4.35,4.70,4.60,3.95,alpha-CH chemical shifts (Andersen et al. 1992)
    0.61,0.60,0.06,0.46,1.07,0.,0.47,0.07,0.61,2.22,1.53,1.15,1.18,2.02,1.95,0.05,0.05,2.65,1.88,1.32,Hydrophobicity index (Argos et al. 1982)
    1.18,0.20,0.23,0.05,1.89,0.72,0.11,0.49,0.31,1.45,3.23,0.06,2.67,1.96,0.76,0.97,0.84,0.77,0.39,1.08,Signal sequence helical potential (Argos et al. 1982)
'''

def download_aaindex():
    if os.path.isfile('aaindex1'):
        print ('Skipped `aaindex1`, already found.')
        return 'aaindex1'
    url = 'https://www.genome.jp/ftp/db/community/aaindex/aaindex1'
    filename = wget.download(url)
    print (f'{filename} downloaded from {url=}')
    return filename

def parse_aaindex(raw_filename: str):
    '''
    Format:
    Data Format of AAindex1 (https://www.genome.jp/aaindex/aaindex_help.html)
    * H Accession number                                                   *
    * D Data description                                                   *
    * R PMID                                                               *
    * A Author(s)                                                          *
    * T Title of the article                                               *
    * J Journal reference                                                  *
    * I Amino acid index data in the following order                       *
    *   Ala    Arg    Asn    Asp    Cys    Gln    Glu    Gly    His    Ile *
    *   Leu    Lys    Met    Phe    Pro    Ser    Thr    Trp    Tyr    Val *
    '''
    with open(raw_filename) as f:
        lines = f.readlines()        
    lines = [line.rstrip('\n') for line in lines]
    
    d = {}
    for i,line in enumerate(lines):
        m = re.match(r'^([A-Z])\s+(.*)', line)
        if not m:
            continue
        if m.group(1) == 'D':
            description = m.group(2)
        if m.group(1) == 'I':
            values  = (lines[i+1] + lines[i+2]).split()
            assert len(values) == 20, f'Found {len(values)} values for amino acids in {description}, expected 20.'
            assert m.group(2).split() == ['A/L', 'R/K', 'N/M', 'D/F', 'C/P', 'Q/S', 'E/T', 'G/W', 'H/Y', 'I/V'], f'Amino acid ordering is not the expected pattern.'
            d[description] = values

    df = pd.DataFrame.from_dict(d, orient='index', columns=['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V'])
    print (f'Finished parsing AAIndex with {len(d.keys())} properties.') 
    return df

def main():
    raw_filename = download_aaindex()
    df = parse_aaindex(raw_filename)
    print (df.head())
    df.to_csv('AAIndex.csv')

if __name__ == '__main__':
    main()
