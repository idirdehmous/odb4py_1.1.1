import sys , os 
import re 

__all__=["OdbLexic"   ]



class OdbLexic:
    """
    Class :
    Functions and Attributes to build a given ODB Object.
    Contains the most glossary used in the ODB/SQL software 
    (Tables , names , SQL tokens etc )
    """
    def __init__(self ):
        """
        Init class ! 
        """
        return None

    def odbSqlwords(self):
        keywords=['AND'
                  'AS',
                  'BETWEEN' ,
                  'BITMAP' ,
                  'CREATE' ,
                  'DISTINCT',
                  'INDEX',
                  'ON' ,
                  'OR' ,
                  'READONLY',
                  'SELECT' ,
                  'SORT BY',
                  'SORTBY',
                  'ORDERBY' , 
                  'ORDER BY',
                  'UNIQUE' ,
                  'UPDATABLE',
                  'UPDATED',
                  'VIEW' ,
                  'WHERE',
                   '$','#','==','<','>', '||']
        return keywords 
                
    def odbFuncs(self):
        """
        THE ODB /SQL Built-in FUNCTIONS 
        USER GUIDE :https://www.ecmwf.int/sites/default/files/elibrary/2004/76278-ifs-documentation-cy36r1-part-i-observation-processing_1.pdf
                    Page No :45

        """
        funcs=[ 
                # a
               'abs','acos','acosh','acot','acot2','acoth',
               'aday','aminute','amonth','anhour','asecond',
               'asin','asinh','atan','atan2','atanh',
               'avg','avg_distinct','ayear',
                # b
               'basetime','bcount','bcount_distinct','binhi','binlo',
                # c
               'c2f','c2k','ceil','celsius','circle','cksum','cksum32','cmp',
               'conv_llu2','corr','cos','cosh','cot','coth','count','count_distinct','covar',
                # d
               'day','dble','dd','deg2rad','degrees','density','dfloat','dint','dir',
               'dist','distance','dnint','dotp','dotp_distinct','duint',
                # e
               'exp',
                # f
               'f2c','f2k','fahrenheit','ff','float','floor','fmaxval','fminval','ftrunc',
                # h
               'hour','hours',
                # i
               'ibits','int',
                # k
               'k2c','k2f','km',
                # l
               'ldexp','lg','linregr_a','linregr_b','llu2','ln','log','log10','lon0to360',
                # m
               'max','maxloc','maxval','median','median_distinct','min',
               'minloc','minute','minutes','minval','mod','month',
                # n
               'nint','norm','norm_distinct',
                # p
               'pi','pow',
                # r
               'rad','rad2deg','rms','rms_distinct',
                # s
               'second', 'seconds', 'shared2argfunc', 'sign', 'sin', 'sinh', 
               'speed' , 'sqrt', 'stdev','stdev_distinct', 'sum', 'sum_distinct',
                # t
               'tan', 'tanh', 'touch', 'trunc', 'tstamp',
                # u
               'ucom', 'ucomp', 'uint', 'uv2dd', 'uv2ff',
                # v
               'var', 'var_distinct', 'vcom', 'vcomp', 'within', 'within360',
                # y
               'year']


        return funcs

    def odbTables (self):
        enkf =  [ "enkf_"+str(i)  for i in range( 121 ) ]
        enda =  [ "enda_"+str(i)  for i in range(101  ) ]
        surfb=  [ "surfbody_feedback_"+str(i) for i in range(101)]
        fc_diag=[ "fcdiagnostic_body_"+str(i) for i in range(21) ]
        
        tables=[    "desc"         ,"ddrs","hdr","body","index","poolmask","errstat","bufr","bufr_tables","bufr_tables_desc",
                    "aeolus_auxmet","aeolus_hdr","aeolus_l2c","resat","rtovs","rtovs_body","rtovs_mlev","rtovs_pred",
                    "rtovs_slev"   , "sat","satem", "satob", "scatt", "scatt_body", "smos", "ralt", "ssmi", "ssmi_body", "ssmi_mlev",
                    "ssmi_slev"    , "timeslot_index", "update", "limb", "resat_averaging_kernel", "radar", "radar_body", "radar_station",
                    "surfbody_feedback", "modsurf", "radiance", "allsky", "co2_sink", "cloud_sink", "collocated_imager_information",
                    "auxiliary"    ,   "auxiliary_body", "radiance_body", "allsky_body", "fcdiagnostic", "gbrad", "gbrad_body", "gnssro",
                    "gnssro_body"  , "ensemble", "conv", "conv_body", "raingg", "raingg_body", "update_1", "update_2", "update_3", 
                    "update_4"     ,"update_5", "update_6", "update_7", "update_8", "update_9", "update_10"]
        tables+= enkf 
        tables+= enda 
        tables+= surfb 
        tables+= fc_diag
        return tables 





    def odbCols (self):

        dict_cols={"desc":["expver","class","stream","type",
                           "andate","antime","inidate","initime",
                           "creadate","creatime","creaby","moddate",
                           "modtime","modby","mxup_traj","numtsl",
                           "poolmask.offset","poolmask.len",
                           "timeslot_index.offset","timeslot_index.len",
                           "fcdiagnostic.offset","fcdiagnostic.len","enda_member"] ,

                   "hdr":["seqno","subseqno","reportno", "bufrtype", "subtype", "subsetno", "groupid", "reportype",
                          "obstype", "codetype", "sensor", "retrtype", "instrument_type", "stalt", "date", "time", "distribtype",
                           "distribid", "gp_dist", "gp_number", "numlev", "numactiveb", "checksum", "sortbox", "areatype", 
                           "report_status", "report_event1","report_rdbflag", "report_blacklist", "report_event2", 
                           "thinningkey_1", "thinningkey_2", "thinningkey_3", "thinningtimekey",
                           "sitedep", "source", "lat", "lon", "trlat", "trlon", 
                           "modsurf.offset", "modsurf.len", "statid", "conv.offset", "conv.len","sat.offset", "sat.len", 
                           "body.offset", "body.len", "errstat.offset", "errstat.len", "update_1.offset", 
                           "update_1.len","update_2.offset", "update_2.len", "update_3.offset", "update_3.len", 
                           "update_4.offset", "update_4.len", "update_5.offset","update_5.len", "update_6.offset", 
                           "update_6.len", "update_7.offset", "update_7.len", "update_8.offset", "update_8.len","update_9.offset", 
                           "update_9.len", "update_10.offset", "update_10.len", "auxiliary.offset", "auxiliary.len", "ensemble.offset",
                           "ensemble.len", "gbrad.offset", "gbrad.len", "raingg.offset", "raingg.len", 
                           "surfbody_feedback.offset", "surfbody_feedback.len","window_offset", 
                           "hdr.offset", "hdr.len", "hdr_count", "aeolus_hdr.offset", "aeolus_hdr.len"]   , 


                    "conv":["linkoffset_t:conv.offset","linklen_t:conv.len",
                            "flight_phase","flight_dp_o_dt",
                            "anemoht","baroht","station_type",
                            "sonde_type","collection_identifier","country",
                            "unique_identifier","timeseries_index","conv_body.offset","ppcode",
                            "Bitfield:level","pk1int:datum_qcflag","pk1int:iternoconv_1dv"] ,
    
                    "conv_body":["conv_body.offset",
                                 "conv_body.len",
                                 "ppcode" ,
                                 "level" ,
                                 "datum_qcflag"] ,

                    "body":["body.offset", "body.len", "conv", "body", "entryno", "obsvalue", "varno", "vertco_type",
                            "vertco_reference_1", "vertco_reference_2", "datum_anflag", "datum_status", "datum_event1", 
                            "datum_rdbflag", "datum_blacklist","datum_event2", "varbc_ix", "biascorr", "biascorr_fg", 
                            "tbcorr", "bias_volatility", "wdeff_bcorr", "an_depar", "fg_depar","actual_depar", 
                            "actual_ndbiascorr", "qc_a", "qc_l", "qc_pge", "fc_sens_obs", "an_sens_obs", "jacobian_peak", 
                            "jacobian_peakl","jacobian_hpeak", "jacobian_hpeakl", "mf_vertco_type", "mf_log_p", 
                            "mf_stddev", "nlayer", 
                            "gnss_satellite_id", "azimuth","elevation", "refconst", "phi", "lon2", "lat2", 
                            "altitude2", "azimuth1", "elevation1", "refconst1", "phi1", "azimuth2","elevation2", "refconst2", 
                            "phi2", "azimuth3", "elevation3", "refconst3", "phi3", "azimuth4", "elevation4", "refconst4", "phi4"]
                         } # DICT TABLES 
        return dict_cols


    def odbPkind (self):
        list_ckind=["string",
                   "pk1int",
                   "pk9real",
                   "yyyymmdd", 
                   "hhmmss" ,
                   "Bitfield"]

        return list_ckind
   
    def odbIfsVar (self):
        ifs_vars={"$NMXUPD"          : 3  ,        # Maximum number of updates supported with this layout (min = 1 & max = 10)
                  "$NUMAUX"          : 9  ,        # No. of auxiliary obsvalue's per body; aux1 ==> aux[$NUMAUX]
                  "$NUMEV"           : 1  ,        # Maximum number of retained eigenvectors of obs. err. corr. matrix
                  "$NUMRAINTBDIAG"   : 14 ,        # No. of diagnostic output variables for cloud/rain assimilation,
                  "$MX_LIMB_TAN"     : 17 ,
                  "$MX_RADAR_NIV"    : 15 ,
                  "$MX_AK"           : 50 ,        # depends on jpmx_ak in module/parcma and bufr2odb_gch2.F90 and varindex_module.F90!!
                  "$MX_SATOB_WEIGHT" : 43 }

        return ifs_vars 



    def LinkPkind(  self):
        kind_dict={"string"  :"TXT"    ,
                   "pk1int"  :"INTEGER",
                   "pk9real" :"REAL"  ,
                   "yyyymmdd":"INTEGER",
                   "hhmmss"  :"INTEGER",
                   "Bitfield":"INTEGER"}
        return kind_dict



    def sqlPrepInsert( self,  query=None ): 
        """
        Takes an CMA sql statement and convert it to 
        an sqlite3 statement for value insertion 
        """

        
        cpkind={   "date"   :"TXT"      ,
                   " time"  :"INTEGER"  ,
                   " statid":"REAL"   ,
                   " varno" :"INTEGER" ,
                   " lat"   :"INTEGER" , 
                   " lon"   :"INTEGER"   ,
                   " obsvalue":"REAL" ,
                   " fg_depar":"REAL" }
    def sqliteEncode (self):
        """
        Empty method :  under building  ! 
        """
        return None


