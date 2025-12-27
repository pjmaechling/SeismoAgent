!Control file for program SMC2FS2
! Revision of program involving a change in the control file on this date:
   03/10/10
! As many comment lines as desired, each starting with "!"
! The string "pp:" indicates a new set of processing parameters
! to be applied to the following smc files.  The parameters are given on the
! lines following "pp:", until the next "pp:" line or until "stop" is 
! encountered.
! NOTE: Use the tapers with caution, choosing them so that important signal
! is not reduced by the tapering.  This can be particularly a problem with 
! analog data from relatively small earthquakes that triggered near the 
! S-wave arrival.  
!
! -----------------------------------------------------------------------------
!
! Meaning of smoothing input parameters
!
! NO SMOOTHING
! itype = 0
! SMOOTHING OVER EQUALLY SPACED FREQUENCIES
! itype = 1: box weighting function
!   smooth_param = width of box weighting function (Hz)
! itype = 2: triangular weighting function
!   smooth_param = width of triangular weighting function (Hz)
! SMOOTHING OVER LOGARITHMICALLY SPACED FREQUENCIES
! itype = 3: box weighting function
!   smooth_param = xi, which is the fraction of a decade for the
!                  box weighting function 
! itype = 4: triangular weighting function
!   smooth_param = xi, which is the fraction of a decade for the
!                  triangular weighting function 
! itype = 5: Konno and Ohmachi weighting function (see BSSA 88, 228-241)
!   smooth_param = xi, which is the fraction of a decade for which
!                  the Konno and Ohmachi weighting function is greater
!                  than 0.043.(it is related to
!                  their smoothing parameter b by the equation
!                  b = 4.0/smooth_param, so we have this correspondence between
!                  b and smooth_param
!                         b smooth_param 
!                        10         0.40
!                        20         0.20
!                        40         0.10
!                  
!                  b = 40 seems to be commonly used, but I do not think that it
!                  gives enough smoothing; I PREFER SMOOTH_PARAM = 0.2, 
!                  corresponding to b = 20. 
!
! ipow = power of FAS to be smoothed (2 = smoothing energy spectrum)
!
! df_smooth: Note: need df_smooth for linearly-spaced smoothers, 
! and generally it should be the df from the fft.  For general x data, it is
! the spacing between x values, assumed to be constant,  The reason for
! including it as an input parameter is to "fool" the
! program to do smoothing over a specified number of points by
! setting df_smooth = 1 and smooth_param = number of points (including 
! points with zero weight at ends; e.g., smooth_param = 5 will 
! give a smoother with weights 0, 1/4, 2/4, 1/4, 0; smooth_param
! should be odd).
!
! -----------------------------------------------------------------------------
! Meaning of frequency specification parameters:
!
!SPECIFY_FREQUENCIES? (y/n):
! <enter Y or N>
!FREQUENCY SPECIFICATION: 
!  If specify_frequencies = Y, then enter the 
!    number of frequencies, freq(1), freq(2)..., freq(nfreq)
!  If specify_frequencies = N, then enter 
!    f_low, f_high, log-spaced (0=N, 1=Y), freq_param
!         if freq_param = 0.0, there is no interpolation, and the FFT frequencies 
!            are used between f_low and f_high (log-spaced is ignored).
!         if freq_param /= 0.0 and log-spaced = 0, then freq_param is the spacing of the
!            interpolated frequencies between f_low and f_high
!         if freq_param /= 0.0 and log-spaced = 1, then freq_param is the number of 
!            interpolated frequencies between f_low and f_high (NOTE: f_low must be > 0.0)
! -----------------------------------------------------------------------------
!
!Name of summary file:
 smc2fs2.sum
PP: new set of parameters
!tskip, tlength
   0.0 2000.0
!dc_remove?
  .true.        
!Length of taper at beginning and end of time series, before adding zeros
! to make the number of points in the record a power of two.
 0.0 0.0
!signnpw2(<0, backup for npw2, no zpad):
 +1.0
!smoothing: itype, ipow, df_smooth (0 = FFT df), smooth_param
! (see above for the meaning of these input parameters):
   0 1 0.0 0.20
!SPECIFY_FREQUENCIES? (y/n):
  N
!FREQUENCY SPECIFICATION
   0.01 100.0 0 0.0 
!character string to append to filename:
   .no_smooth.fs.col
!Output in smc format (Y,N)?
! ***IMPORTANT NOTE: Output cannot be in smc format if use log-spaced 
! frequencies because programs such as smc2asc have not been modified
! to deal with log-spaced frequency.
 n
!Files to process:
6195187.SLB.acc.bbp.smc8.EW
stop
