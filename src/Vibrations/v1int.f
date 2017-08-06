C FILE: V1INT.F
      SUBROUTINE V1INT(S,V,WFN,N)
C
C     CALCULATE 1-MODE INTEGRAL
C
      INTEGER N
      REAL*8 WFN(N)
      REAL*8 V(N)
      REAL*8 S
      REAL*8 TMP
      REAL*8 TMP2(N)
Cf2py intent(in) n
Cf2py intent(in) wfn
Cf2py intent(in) v
Cf2py intent(out) s
      TMP = 0.0D0

      TMP2 = V*WFN
      S = SUM(TMP2)
      END
C END FILE V1INT.F
