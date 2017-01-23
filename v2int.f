C FILE: V2INT.F
      SUBROUTINE V2INT(S,V2,WFN1,WFN2,N)
C
C     CALCULATE 2-MODE INTEGRAL
C
      INTEGER N
      REAL*8 WFN1(N)
      REAL*8 WFN2(N)
      REAL*8 V2(N,N)
      REAL*8 S
      REAL*8 TMP
Cf2py intent(in) n
Cf2py intent(in) wfn1
Cf2py intent(in) wfn2
Cf2py intent(in) v2
Cf2py intent(out) s
      TMP = 0.0D0
      DO I=1,N
        DO J=1,N
          TMP = TMP + WFN1(I) * WFN2(J) * V2(I,J)
        ENDDO
      ENDDO
      S = TMP
      END
C END FILE V2INT.F
