C FILE: V3INT.F
      SUBROUTINE V3INT(S,V3,WFN1,WFN2,WFN3,N)
C
C     CALCULATE 3-MODE INTEGRAL
C
      INTEGER N
      REAL*8 WFN1(N)
      REAL*8 WFN2(N)
      REAL*8 WFN3(N)
      REAL*8 V3(N,N,N)
      REAL*8 S
      REAL*8 TMP
Cf2py intent(in) n
Cf2py intent(in) wfn1
Cf2py intent(in) wfn2
Cf2py intent(in) wfn3
Cf2py intent(in) v3
Cf2py intent(out) s
      TMP = 0.0D0
      DO I=1,N
        DO J=1,N
          DO K=1,N
            TMP = TMP + WFN1(I) * WFN2(J) * WFN3(K) * V3(I,J,K)
          ENDDO
        ENDDO
      ENDDO
      S = TMP
      END
C END FILE V3INT.F
