#define PY_SSIZE_T_CLEAN
#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <Python.h>


static PyObject* odbTables_method(PyObject* Py_UNUSED(self), PyObject* Py_UNUSED(args)) 
{ 
char *db_tables[]={"desc","ddrs","hdr","body","index","poolmask","errstat","bufr","bufr_tables","bufr_tables_desc",
                 "aeolus_auxmet","aeolus_hdr","aeolus_l2c","resat","rtovs","rtovs_body","rtovs_mlev","rtovs_pred",
"rtovs_slev", "sat", "satem", "satob", "scatt", "scatt_body", "smos", "ralt", "ssmi", "ssmi_body", "ssmi_mlev",
"ssmi_slev", "timeslot_index", "update", "limb", "resat_averaging_kernel", "radar", "radar_body", "radar_station",
"surfbody_feedback", "modsurf", "radiance", "allsky", "co2_sink", "cloud_sink", "collocated_imager_information",
"auxiliary", "auxiliary_body", "radiance_body", "allsky_body", "fcdiagnostic", "gbrad", "gbrad_body", "gnssro",
"gnssro_body", "ensemble", "conv", "conv_body", "raingg", "raingg_body", 
"update_1", "update_2", "update_3", "update_4", "update_5", 
"update_6", "update_7", "update_8", "update_9", "update_10", 
"enkf_[1-$NUMAUX]", 
"enda_[1-$NUMAUX]",
"surfbody_feedback_[1-$NUMAUX]", 
"fcdiagnostic_body_[1-$NUMAUX]"}; 

long unsigned int i = 0  ; 
while ( i <= sizeof (db_tables)/ sizeof (double)-2   ) {
       i++  ; 
       printf ( "%s\n",  db_tables[i] ) ; 
}
return PyLong_FromLong(0) ; 
} 


static  PyObject*  odbVarno_method(PyObject* Py_UNUSED(self), PyObject* Py_UNUSED(args) )
{
char *var_name[]={"u", "v", "z", "dz", "rh", "pwc", "rh2m", "t", "td", "t2m", "td2m", "ts", "ptend", 
              "w", "ww", "vv", "ch", "cm", "cl", "nh", "nn", "hshs", "c", "ns", "sdepth", "e", "tgtg", "spsp1",
              "spsp2", "rs", "eses", "is", "trtr", "rr", "jj", "vs", "ds", "hwhw", "pwpw", "dwdw", "gclg",
              "rhlc", "rhmc", "rhhc", "n", "sfall", "ps", "dd", "ff", "rawbt", "rawra", "satcl", "scatss",
              "du", "dv", "u10m", "v10m", "rhlay", "cllqw", "scatdd", "scatff", "q", "scatwd", "scatws",
              "vsp", "vt", "o3lay", "height", "1dvar", "w2", "cpt", "tsts", "refl", "apdss", "bend_angle", "los",
              "aerod", "limb_radiance", "chem3", "chem2", "chem1", "cod", "rao", "od", "rfltnc", "nsoilm", "soilm",
              "flgt_phase", "height_assignment_method", "dopp", "ghg1", "ghg2", "ghg3", "bt_real", "bt_imaginary", "prc",
              "lnprc", "libksc", "ralt_swh", "ralt_sws", "rawbt_clear", "rawbt_cloudy", "binary_snow_cover", "salinity", 
	      "potential_temp", "humidity_mixing_ratio", "airframe_icing", "turbulence_index", "pstation", "pmsl", 
	      "pstandard", "vert_vv", "max_wind_shear1", "tot_zen_delay", "tot_zen_delay_err", "cloud_top_temp", "rawsca", 
	      "cloud_top_press", "mean_freq", "u_amb", "v_amb", "lwp", "tcwv", "cloud_frac_clear", "rawbt_hirs", "rawbt_amsu", 
	      "rawbt_hirs20", "sea_ice", "cloud_frac_covered", "level_mixing_ratio", "radial_velocity", "cloud_ice_water", 
	      "wind_gust", "mass_density","atmosphere_number", "lightning", "level_cloud", "rawbt_amsr_89ghz", "max_wind_shear2", 
	      "lower_layer_p", "upper_layer_p","cloud_cover", "depth", "ssh", "rawbt_mwts", "rawbt_mwhs", 
	      "tot_lightning_flash_dens", "cg_lightning_flash_dens","lidar_aerosol_extinction", "lidar_cloud_backscatter", 
	      "lidar_cloud_extinction", "cloud_radar_reflectivity","lidar_aerosol_attenuated_backscatter", 
	      "q2m", "chem6", "sla", "ice_freeboard", "snow_freeboard", "visible_spectral_reflectance","od10", 
	      "chem4", "chem5", "frac_snow_cover", "cloud_doppler_velocity", "lidar_rayleigh_backscatter","sigma0_sm", 
	      "t2m_min", "t2m_max", "ssrd", "chem7" };

char*code_no[]={"3","4","1","57","29","9","58","2","59","39","40","11","30","60","61","62","63","64","65","66","67","68","69","70",
                "71","72","73","74","75","76","77","78","79","80","81","82","83","84","85","86","87","88","89","90","91","92",
                "110","111","112","119","120","121","122","5","6","41","42","19","123","124","125","7","126","127","8","56",
                "206","156","215","160","130","12","192","128","162","187","174","163","183","182","181","175","176","177",
                "178","179","180","201","211","195","186","188","189","190","191","202","203","222","220","221","193","194",
                "223","224","225","226","227","228","107","108","109","218","219","229","230","231","233","235","241","242",
                "243","244","245","247","248","249","250","253","257","258","259","260","261","262","263","265","266","267",
                "268","269","270","271","272","273","274","275","196","197","236","237","238","239","280","281","284","287",
                "286","285","240","288","184","185","282","251","252","283","37","38","25","289"};

char *desc[]={
"upper air u component",
"upper air v component",
"geopotential",
"thickness",
"upper air rel. humidity",
"precipitable water content",
"2m rel. humidity",
"upper air temperature (K)",
"upper air dew point (K)",
"2m temperature (K)",
"2m dew point (K)",
"surface temperature (K)",
"pressure tendency",
"past weather (w)",
"present weather (ww)",
"visibility",
"type of high clouds (ch)",
"type of middle clouds (cm)",
"type of low clouds (cl)",
"cloud base height (nh) (meter)",
"low cloud amount (n)",
"additional cloud group height (hh)",
"additional cloud group type (c)",
"additional cloud group amount (ns)",
"snow depth",
"state of ground (e)",
"ground temperature (tgtg)",
"special phenomena (spsp)#1",
"special phenomena (spsp)#2",
"ice code type (rs)",
"ice thickness (eses)",
"ice (is)",
"original time period of rain obs. (trtr)",
"6hr rain (liquid part)",
"max. temperature (jj)",
"ship speed (vs)",
"ship direction (ds)",
"wave height",
"wave period",
"wave direction",
"general cloud group",
"rel. humidity from low clouds",
"rel. humidity from middle clouds",
"rel. humidity from high clouds",
"total amount of clouds",
"6hr snowfall (solid part of rain)",
"surface pressure",
"wind direction",
"wind force",
"brightness temperature (K)",
"raw radiance",
"cloud amount from satellite",
"sigma 0",
"wind shear (du)",
"wind shear (dv)",
"10m u component (m/s)",
"10m v component (m/s)",
"layer rel. humidity",
"cloud liquid water",
"ambiguous v component",
"ambiguous u component",
"specific humidity (q)",
"ambiguous wind direction",
"ambiguous wind speed",
"vertical speed",
"virtual temperature",
"layer ozone",
"height",
"1d-var model level (pseudo)-variable",
"past weather 2 (used in synoptic maps)",
"characteristic of pressure tendency (used in synoptic maps)",
"sea water temperature (used in synoptic maps)",
"radar reflectivity",
"atmospheric path delay in satellite signal",
"radio occultation bending angle",
"horizontal line-of-sight wind component",
"aerosol optical depth at 0.55 microns",
"Limb Radiances",
"chem3: co",
"chem2: so2",
"chem1: no2/nox",
"cloud optical depth",
"Ratio of fine mode to total aerosol optical depth at 0.55 microns",
"optical depth",
"Aerosol reflectance multi-channel",
"normalized soil moisture  (0-100%)",
"soil moisture",
"phase of aircraft flight",
"Height assignment method",
"radar doppler wind",
"ghg1: carbon dioxide",
"ghg2: methane",
"ghg3: nitrous oxide",
"brightness temperature real part",
"brightness temperature imaginary part",
"radar rain rate",
"log(radar rain rate mm/h + epsilon)",
"lidar backscattering",
"significant wave height (m)",
"surface wind speed (m/s)",
"brightness temperature for clear  (K)",
"brightness temperature for cloudy (K)",
"binary snow cover (0: no snow / 1: presence of snow)",
"ocean salinity (PSU)",
"potential temperature (Kelvin)",
"humidity mixing ratio (kg/kg)",
"airframe icing",
"turbulence index",
"Station pressure (Pa)",
"Mean sea-level pressure (Pa)",
"Standard level pressure (Pa)",
"Vertical visibility (m)",
"Wind shear above and below 1st maximum wind in sonde profile (s-1)",
"Total zenith delay (GPS)",
"Total zenith delay error (GPS)",
"Cloud top temperature (K)",
"Scaled radiance",
"Cloud top pressure (Pa)",
"GPSRO mean frequency",
"Ambiguous u-wind component (m/s)",
"Ambiguous v-wind component (m/s)",
"Liquid water path",
"Total column water vapour",
"Cloud clear fraction",
"Raw brightness temperature specific to HIRS (K)",
"Raw brightness temperature specific to AMSU (K)",
"Raw brightness temperature specific to HIRS (K)",
"Sea ice fraction",
"Cloud covered fraction",
"humidity_mixing_ratio]",
"Radial velocity from doppler radar",
"Cloud ice water",
"Maximum wind gust (m/s)",
"Mass density",
"SFERICS number of atmospheres",
"Lightning strike observation (ATDNET)",
"Cloud fraction (multi-level)",
"Raw brightness temperature specific to AMSR 89GHz channels (K)",
"Wind shear above and below 2nd maximum wind in sonde profile",
"Pressure at bottom of layer SBUV (Pa)",
"Pressure at top of later SBUV (Pa)",
"Total cloud cover",
"Depth (m)",
"Sea surface height (m)",
"Raw brightness temperature specific to MWTS (K)",
"Raw brightness temperature specific to MWHS (K)",
"total (cloud-to-ground plus intra-cloud) lightning flash density (fl/km2/day)",
"cloud-to-ground lightning flash density ( fl/km2/day)",
"lidar aerosol extinction (1/m)",
"lidar cloud backscatter",
"lidar cloud extinction",
"cloud radar reflectivity",
"lidar aerosol attenuated backscatter (1/m*sr)",
"specific humidity at 2m (kg/kg)",
"volcanic SO2",
"sea level anomaly",
"Height of sea ice above open water",
"Height of snow and sea ice above open water",
"Visible Spectral Reflectance",
"optical depth at 10 microns",
"chem4: hcho",
"chem5: go3",
"fractional snow cover",
"vertical radar doppler velocity",
"lidar Rayleigh backscatter",
"backscatter coefficient normalized at 40 degree (db)",
"minimum 2m temperature (K)",
"maximum 2m temperature (K)",
"downward surface solar radiation (J/m2)",
"chem7: h2o" };


for ( long unsigned int i=0 ; i <= (sizeof (code_no ) /sizeof ( double)) -1  ; i++){
  printf (  "%s          %s          //%s \n" ,  code_no[i], var_name[i], desc[i] ) ; 

}
return PyLong_FromLong(0) ;

}




static PyObject* odbFunctions_method(PyObject* Py_UNUSED(self))
{
char *cap_funcs[]={ 
	"Ln", "Lg" , "ftrunc", "Cot", "ACot", "ACot2", "Coth", "Asinh", "Acosh", "Atanh", "ACoth", "sign",
"touch", "binlo", "binhi", "dnint", "dint", "cmp" , "fmaxval", "fminval", "Cksum32", "duint", "dfloat", "dble",
"k2c", "c2k", "c2f", "f2c", "k2f", "f2k", "rad", "distance", "dist", "km", "pow", "circle", "lon0to360", "ibits",
"aYear", "aMonth", "aDay", "anHour", "aMinute", "aSecond"} ;


/* Wind speed (ff), direction (dd), u- and v-components */
char *ws_funcs[]={"Speed", "uv2ff", "ff"   , "Dir" , "dd"  , "Ucom", "ucomp", "Vcom", "vcomp"};


/*Aliases */
char *agg_cap_funcs[]= {"Within", "Within360", "llu2", "Min",
"Max", "Density", "Count", "Bcount", "Sum", "Avg", "Median", "Stdev", "Var", "Rms", "Dotp", "Norm", "Count_distinct",
"Bcount_distinct", "Sum_distinct", "Avg_distinct", "Median_distinct", "Stdev_distinct", "Var_distinct", "Rms_distinct",
"Dotp_distinct", "Norm_distinct", "Shared2ArgFunc", "Covar", "Corr", "Linregr_a", "Linregr_b", "Maxloc", "Minloc" };


/* Compile-time evaluable functions */
char *ct_funcs[] ={"sin"     ,  "cos"     ,  "tan"      ,  "cot"      , "asin"    ,   "acos"     ,  "atan"     , "atan2"  ,   "acot",
  "acot2"   ,  "sinh"    ,  "cosh"     ,  "tanh"     , "coth"    ,   "asinh"    ,  "acosh"    , "atanh"  ,   "acoth"   ,   
  "sqrt"    ,  "mod"     ,  "pow"      ,  "exp"      , "log"     ,   "ln"       ,  "lg"       , "log10"  ,   "floor"   ,   
  "ceil"    ,  "ldexp"   ,  "abs"      ,  "trunc"    , "touch"   ,   "sign"     ,  "binlo"    , "binhi"  ,   "int"     ,   
  "nint"    ,  "cmp"     ,  "max"      ,  "maxval"   , "min"     ,   "minval"   ,  "cksum"    , "uint"   ,   "float"   ,  
  "dble"    ,  "celsius" ,  "k2c"      ,  "c2k"      , "fahrenheit", "k2f"      ,  "f2k"      , "c2f"    ,   "f2c"      , 
  "rad"     ,  "distance",  "dist"     ,  "km"       , "lon0to360",  "ibits"    ,  "circle"   , "year"   ,   
  "month"   ,  "day"     ,  "hour"     ,  "hours"    , "minutes"  ,  "minute"   ,  "seconds"  , "second" ,    
  "pi"      ,  "speed"   ,  "uv2ff"    ,  "ff"       , "dir"      ,  "uv2dd"    ,  "dd"       , "u"      ,   "ucomp"   ,"ucom","v"        ,  "vcomp","vcom","within","within","tstamp", "basetime"};


  /* Aggregate functions (prepended with underscore when with FUNCS_C) */
char *agg_funcs[]={"density" , "count"  , "bcount" , "sum" , "avg" , "median" , "stdev", "var", "rms", "dotp",
  "norm"    , "count_distinct"    , "bcount_distinct", "sum_distinct"   , "avg_distinct", "median_distinct", 
  "stdev_distinct"      , "var_distinct"  , "rms_distinct"    , "dotp_distinct", "norm_distinct","covar",
  "corr"    , "linregr_a", "linregr_b", "minloc","maxloc","Conv_llu2" };


//cap
for ( long unsigned int i=0 ; i<=(sizeof (cap_funcs)/sizeof( double))-1;i++)     {printf(  "%s    \n" ,cap_funcs[i]);}
//ws 
for ( long unsigned int i=0 ; i<=(sizeof (ws_funcs)/sizeof( double))-1;i++)      {printf(  "%s    \n" ,ws_funcs[i]);}
//agg_cap
for ( long unsigned int i=0 ; i<=(sizeof (agg_cap_funcs)/sizeof( double))-1;i++) {printf(  "%s    \n" ,agg_cap_funcs[i]);}
//ct
for ( long unsigned int i=0 ; i<=(sizeof (ct_funcs)/sizeof( double))-1;i++)      {printf(  "%s    \n" ,ct_funcs[i]);}
//agg 
for ( long unsigned int i=0 ; i<=(sizeof (agg_funcs)/sizeof( double))-1;i++)     {printf(  "%s    \n" ,agg_funcs[i]);}




return PyLong_FromLong(0) ;
}

/*static PyObject* odbSchema_method (  ){
//      1  desc
//      2  +---> poolmask
//      3  +---> timeslot_index
//      4  |     +---> index
//      5  |     |     +---> hdr
//      6  |     |     |     +---> sat
//      7  |     |     |     |     +---> radiance
//      8  |     |     |     |     |     +---> allsky
//      9  |     |     |     |     |           +---> allsky_body
//     11  |     |     |     |     |     +---> cloud_sink
//     12  |     |     |     |     |     +---> radiance_body
//     14  |     |     |     |     |     +---> collocated_imager_information
//     15  |     |     |     |     +---> resat
//     16  |     |     |     |     |     +---> resat_averaging_kernel
//     17  |     |     |     |     +---> gnssro
//     18  |     |     |     |     |     +---> gnssro_body
//     19  v     v     v     v     +---> satob
//     20  |     |     |     |     +---> scatt
//     21  |     |     |     |     |     +---> scatt_body
//     22  |     |     |     |     +---> ssmi
//     23  |     |     |     |     |     +---> ssmi_body
//     24  |     |     |     |     +---> limb
//     25  |     |     |     |     +---> aeolus_hdr
//     26  |     |     |     |     |     +---> aeolus_auxmet
//     27  |     |     |     |     |     +---> aeolus_l2c
//     28  |     |     |     |     +---> radar_station
//     29  |     |     |     |     +---> radar            
//     30  |     |     |     |     |     +---> radar_body 
//     31  |     |     !     !     +---> smos
//     32  |     |     |     +---> body
//     33  |     |     |     +---> auxiliary
//     34  |     |     |     +---> errstat
//     35  |     |     |     +---> update_1
//     36  |     |     |     +---> update_2
return PyLong_FromLong( 0 ) ; 
}*/


