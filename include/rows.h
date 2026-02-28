#ifndef ROWS_H
#define ROWS_H
#include <stdio.h>
#include "odbdump.h"

/**
 * @param : database path  
 * @param : sql_query 
 * @param : nfunc     number of functions in the SQL statement  (ex: avg, sin, cos)
 *
 * @return : total number of rows in the ODB 
 */


static int getMaxrows(const char* database, const char* sql_query)
{
    void *h        = NULL;
    char *poolmask = NULL;
    int maxcols    = 0;
    int nrows      = 0;

    // ouverture ODB
    h = odbdump_open(database, sql_query, NULL, poolmask, NULL, &maxcols);
    if (!h || maxcols <= 0) {
        fprintf(stderr, "In function getMaxrows. Warning: cannot open database or no columns found.\n");
        return 0;
    }

    int new_dataset = 0;
    double *d = NULL;
    int nd;
    int (*nextrow)(void *, void *, int, int *) = odbdump_nextrow;
    int dlen = (unsigned int) maxcols;
    ALLOCX(d, maxcols);
    
    while ((nd = nextrow(h, d, dlen, &new_dataset)) > 0) {
        if (new_dataset) {
            // ignorer dataset switches
            new_dataset = 0;
        }
        ++nrows;
    }
    odbdump_close(h);
    FREEX(d);
    return nrows;
}

#endif   
