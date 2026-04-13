      include  '../ynogkm.f90' 
      PROGRAM  MAIN
      USE constants	
      IMPLICIT NONE  
      Double precision  a_spin,theta_ini,r_ini,sin_ini,cos_ini,e     	
      read(unit=5,fmt=*)a_spin,theta_ini,r_ini 

      If(theta_ini.ne.90.D0)then 
          cos_ini = cos(theta_ini*dtor)
          sin_ini = sin(theta_ini*dtor)
      else
          cos_ini = zero
          sin_ini = one 
      endif
 
      e = 0.1D0   
      IF(e*e+a_spin*a_spin.ge.one)Then
      write(*,*)'e^2+a^2>=1, there can be no horizon. wrong and stoped.'
          stop
      ENDIF

      CALL pmcharges(sin_ini,cos_ini,a_spin,e,r_ini)
      END
!***************************************************************************************************** 
      SUBROUTINE pmcharges(sin_ini,cos_ini,a_spin,e,r_ini)
!*****************************************************************************************************
      use blcoordinates
      implicit none
      Double precision sin_ini,cos_ini,a_spin,e,r_ini,somega,expnu,exppsi,expmu1,expmu2,&
             vptl(1:3),velocity(1:3),velo,rhorizon,lambda,q,mve,ep,kvec(1:4),pmax,&
             ra,mua,timea,phya,sigmaa,p,theta,phy,theta_star
      integer :: m,n,i,j,k  
      logical :: cir_orbt

      cir_orbt = .false.
      theta_star = zero
      call metricg(r_ini,sin_ini,cos_ini,a_spin,e,somega,expnu,exppsi,expmu1,expmu2)
      velocity(1)=0.0D0 
      velocity(2)=0.0D0 
      velocity(3)=exppsi/expnu*(-somega)!*zero 

      velo=0.35D0 
      m=20
      n=20 
      rhorizon=one+sqrt(one-a_spin**two-e*e)   
 
      open(unit=15,file='rayx1.txt',status="replace")
      open(unit=16,file='rayy1.txt',status="replace")  	
      open(unit=17,file='rayz1.txt',status="replace")  	  			
      DO i=m/2-1,m/2-1 	 
          Do j=0,n
              vptl(1)= velo*sin(j*two*pi/n)!*sin(i*pi/(m-1))
              vptl(3)= velo*cos(j*two*pi/n)!*sin(i*pi/(m-1))
              vptl(2)= velo*cos(i*pi/(m-1))*zero 
              IF(dsqrt(vptl(1)**2+vptl(2)**2+vptl(3)**2).ge.1.D0)THEN
                  write(*,*)'supper light speed.'
                  stop
              ENDIF 
 
 
              call lambdaqm(vptl,sin_ini,cos_ini,a_spin,e,r_ini,one,velocity,lambda,q,mve,ep,kvec) 
              pmax = p_total(kvec,lambda,q,mve,ep,sin_ini,cos_ini,a_spin,e,r_ini)  
              Do k=0,800
                  p=k*pmax*0.98D0/800!0.002!deltap    
                  CALL YNOGKM(p,kvec,lambda,q,mve,ep,sin_ini,cos_ini,a_spin,e,r_ini,&
                                          ra,mua,timea,phya,sigmaa,cir_orbt,theta_star) 
              
                  If(ra .ge. rhorizon*0.99)then  
                      write(unit=15,fmt=*)sqrt(ra**two+a_spin**two)*sqrt(one-mua**two)*cos(phya)
                      write(unit=16,fmt=*)sqrt(ra**two+a_spin**two)*sqrt(one-mua**two)*sin(phya)
                      write(unit=17,fmt=*)ra*mua
                  Else
                      write(unit=15,fmt=*)sqrt(ra-100.D0) 
                      write(unit=16,fmt=*)sqrt(ra-100.D0) 
                      write(unit=17,fmt=*)sqrt(ra-100.D0)          
                  ENDIF  
              Enddo
          Enddo
      Enddo
      close(unit=15)	
      close(unit=16)
      close(unit=17) 
      return
      end subroutine pmcharges










