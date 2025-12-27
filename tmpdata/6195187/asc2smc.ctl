!Control file for ASC2SMC      ! first line
! Revision of program involving a change in the control file on this date:
   02/02/12
!Name of summary file:
 asc2smc.sum
!n2skip (-1=headers preceded by !; 0=no headers; otherwise number of headers to skip)
 1
!write headers to smc file (even if n2skip > 0)? (Y/N)
 Y
!sps (0.0 = obtain from input file)
 0
!N columns to read, column number for time and data columns 
!  (for files made using blpadflt, period is in column 1 and sd, pv, pa, rv, 
!  aa are in columns 2, 3, 4, 5, 6, respectively)
! Note: if sps .ne. 0.0, then column number for time is ignored (but a placeholder is
! still needed--e.g., 1 1 1 (read one column, which contains the data; 1 20 1 would be the same)
! But note: if the data are not in the first column, but only the data column is to be read
! (because sps will be used to establish the time values),
! then ncolumns must be the column corresponding to the data.  For example, assume that
! the data are in column 3 and that columns 1 and 2 contain time and some other variable, but
! the time column is not to be used (perhaps because accumulated error in creating the column
! leads to a slight shift in the time values).  Then the input line should be:
!  3 1 3
!
! This program assumes one data point per row; if there are more points (as, for example,
! in files with N points per line), use the program wrapped2asc).
!
 3 1 3
!Xfactr
 1.0
!Read input format (used if the format is such that the values are not separated by spaces,
!in which case a free format cannot be used for input)?
  N
!If yes, specify a format; if not, still need a placeholder
 (3e13.5)
!For output, use old (standard) smc format or new
!higher precision format.   Specify "high" for
!high precision; any other word defaults to standard
!precision (but some word is needed as a placeholder, even if
!standard precision is desired).
 high
!String to append to input file name for the output filename.
 .smc8.EW
!Input file name (time,data pairs; "stop" in any column to quit):
6195187.SLB.acc.bbp
STOP
