      include  '../ynogkm.f90' 
      PROGRAM  MAIN
      use constants
      IMPLICIT NONE  
      Double precision a_spin,beta,r_ini,sbeta,cbeta,e
	 
      read(unit=5,fmt=*)a_spin,beta,r_ini

      If(beta.ne.90.D0)then 
          cbeta=cos(beta*dtor)
          sbeta=sin(beta*dtor)
      else
          cbeta=0.D0
          sbeta=1.D0
      endif
  	
      e = 0.D0         
      IF(e*e+a_spin*a_spin.ge.one)Then
      write(*,*)'e^2+a^2>=1, there can be no horizon. wrong and stoped.'
          stop
      ENDIF
      call reashift2(sbeta,cbeta,a_spin,e,r_ini)
      END 

!*****************************************************************************************************
      subroutine reashift2(sbeta,cbeta,a_spin,e,r_ini)
!*****************************************************************************************************
      use blcoordinates
      implicit none
      Double precision sbeta,cbeta,a_spin,e,r_ini,rhorizon,theta_star,beta,&
             phy,cpsi,spsi,cphy,sphy,theta,stheta,ctheta,velocity(1:3),vptl(1:3),pmax,&
             somega,expnu,exppsi,expmu1,expmu2,lambda,q,mve,ep,kvec(1:4),ra,mua,timea,&
             phya,sigmaa,Vx,Vy,Vz,temp,p,svarphi,cvarphi 
      integer  i,j,m,n,k,t1,t2  
      logical :: cir_orbt

      cir_orbt = .false. 
      theta_star = zero 
      rhorizon=one+sqrt(one-a_spin**two-e*e) 
 
      m = 100
 
      open(unit=15,file='rayx2.txt',status="replace")
      open(unit=16,file='rayy2.txt',status="replace")  	
      open(unit=17,file='rayz2.txt',status="replace")    			
 
          Do j=0,m-1
            !Here we set gamma=0, see Fig A.1, thus phy = psi-gamma = psi.
              phy = two*pi/m*j 
              cphy = dcos(phy)
              sphy = dsin(phy)
            ! See equations (A.6)---(A.9) of Yang & Wang, 2013 A&A.
              svarphi = spsi*cbeta/dsqrt(one-sphy*sphy*sbeta*sbeta)  
              cvarphi = cpsi/dsqrt(one-sphy*sphy*sbeta*sbeta)  
              stheta = cbeta/dsqrt(one-sphy*sphy*sbeta*sbeta)  
              ctheta = -cphy*sbeta/dsqrt(one-sphy*sphy*sbeta*sbeta)   

              Vx = 0.01D0
              Vy = 0.5D0
              Vz = -0.01D0*dcos(phy)
 
              temp = sphy*sbeta
         !See equation (A.5) of Yang & Wang, 2013,A&A.
              vptl(1) = -Vy
              vptl(2) = -temp*Vx-sqrt(one-temp*temp)*Vz
              vptl(3) = sqrt(one-temp*temp)*Vx-temp*Vz 
 
              IF(dsqrt(vptl(1)**2+vptl(2)**2+vptl(3)**2).ge.1.D0)THEN
                  write(*,*)'supper light speed.'
                  stop
              ENDIF 

              velocity(1) = zero
              velocity(2) = zero 
              velocity(3) = zero
              call lambdaqm(vptl,stheta,ctheta,a_spin,e,r_ini,one,velocity,lambda,q,mve,ep,kvec) 
              t1 = 0
              t2 = 0
              pmax = r2p(kvec(1),1.5D0,lambda,q,mve,ep,a_spin,e,r_ini,t1,t2) 
              Do k = 0,200
                  p = k*pmax/200.D0 
                  CALL YNOGKM(p,kvec,lambda,q,mve,ep,stheta,ctheta,a_spin,e,r_ini,&
                                          ra,mua,timea,phya,sigmaa,cir_orbt,theta_star)  
  
                  If(ra .ge. rhorizon*1.15D0)then  
                      write(unit=15,fmt=*)sqrt(ra**two+a_spin**two)*sqrt(one-mua**two)*cos(phya+phy) 
                      write(unit=16,fmt=*)sqrt(ra**two+a_spin**two)*sqrt(one-mua**two)*sin(phya+phy)
                      write(unit=17,fmt=*)sqrt(ra**two+a_spin**two)*mua
                  Else
                      write(unit=15,fmt=*)sqrt(ra-100.D0) 
                      write(unit=16,fmt=*)sqrt(ra-100.D0) 
                      write(unit=17,fmt=*)sqrt(ra-100.D0)          
                  ENDIF  
              Enddo
          Enddo
 
      close(unit=15)	
      close(unit=16)
      close(unit=17)
 
      return
      end subroutine reashift2










