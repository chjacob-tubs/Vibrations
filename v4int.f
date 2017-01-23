C FILE: V4INT.F
      SUBROUTINE V4INT(S,V4,WFN1,WFN2,WFN3,WFN4,N)
C
C     CALCULATE 4-MODE INTEGRAL
C
      INTEGER N
      REAL*8 WFN1(N)
      REAL*8 WFN2(N)
      REAL*8 WFN3(N)
      REAL*8 WFN4(N)
      REAL*8 V4(N,N,N,N)
      REAL*8 S
      REAL*8 TMP
Cf2py intent(in) n
Cf2py intent(in) wfn1
Cf2py intent(in) wfn2
Cf2py intent(in) wfn3
Cf2py intent(in) wfn4
Cf2py intent(in) v4
Cf2py intent(out) s
      TMP = 0.0D0
      PRINT *,WFN1(1)
      PRINT *,V4(1,2,3,4)
      DO I=1,N
        DO J=1,N
          DO K=1,N
            DO L=1,N
              TMP = TMP + WFN1(I) * WFN2(J) * WFN3(K) * WFN4(L)
     & * V4(I,J,K,L)

            ENDDO
          ENDDO
        ENDDO
      ENDDO
      S = TMP
C     PRINT *,TMP
      END
C END FILE V4INT.F
