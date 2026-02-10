# -*- coding: utf-8 -*-
import os  , sys 
from sys import byteorder 
import glob  
import ctypes
import re

__all__=[ "StringParser" ]

from   .exceptions               import *
from   .odb_glossary             import OdbLexic 


od=OdbLexic ()  

class StringParser:
      def __init__(self):
          return None

      def CheckPattern ( self,string ,  pattern , **kwargs):
          found=None 
          self.string = string 
          self.pattern= pattern 
          self.find  = kwargs["find"]
          self.match = kwargs["match"]
          self.path  = kwargs["path"]
          self.dbname= kwargs["dbname"]
          if self.find ==re.findall ( self.pattern , self.string    ):
              return True

      
      # FOR THE MOMENT DONE WITH SIMPLE regex (works pretty well !)
#      FROM   include/info.h  
#  int ncols;              /* total number of columns to be dealt with when executing SELECT */
#  int ncols_true;         /* True number of columns i.e. ncols_pure + ncols_formula + ncols_aggr_formula */
#  int ncols_pure;         /* no formula, pure column (kind=1) */
#  int ncols_formula;      /* simple formula (kind=2) */
#  int ncols_aggr_formula; /* aggregate formula (kind=4) */
#  int ncols_aux;          /* auxiliary cols needed due to aggregate functions (kind=8) */
#  int ncols_nonassoc;     /* column variables needed to be prefetched due to formula(s) (kind=0) */
#  int ncolaux;            /* Should be == ncols_true + ncols_aux */



      def ParseTokens(self, raw):
          fcnt=[]
          od  =OdbLexic ()
          kws =od.odbSqlwords()                     # ODB/SQL WORDS
          func=od.odbFuncs   ()                     # ODB FUNCTIONS 
          
          
          rr=raw.lower().split()                    # BE SURE THAT THE QUERY IS IN LOWER CASE 
          rj="".join(rr).partition('from')          # JOIN EVERYTHING AND USE "from" KEYWORD AS DELIMITER
          tokens = rj[0:1]                          # THE SELECT STETEMENT IS AT INDEX 0 
          # GET built-in functions OCCURENCES WITH REGEX 
          # TEMPORARY LIST
          # FUNCTIONS NAMES ARE NOT CASE SENSITIVE IN THE ODB/SQL SYNTAX PARSER 
          # Tokens are already in lower case , the functions names should also !
          # Counting the number of functions is not for fun !  
          # The ODB C code allocates N columns for pure select in query + N cols for the used functions ( it should be sustracted ! to avoid <NULL> in returned arrays )

          cnt=0
          for t in tokens:
              for f in func: 
                  fn=f.lower()
                  regex  =fn+r"\("
                  found =re.findall ( regex , t  ) 
                  n_occur=len(found )  
                  if n_occur  != 0:
                    cnt = cnt + n_occur 
                
          return cnt  


      def CleanString  (self , string ):
          # C characters not accepted in strings arguments 
          c_char='\0\a\b\f\n\r\r\t\v\\'
          cleaned_st= string.translate({ord(i): None for i in c_char})
          return cleaned_st
                 
