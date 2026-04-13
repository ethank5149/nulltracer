      include  '../ynogkm.f90' 
      PROGRAM  MAIN
      USE constants	
      IMPLICIT NONE  
      Double precision  a_spin,theta_ini,r_ini,sin_ini,cos_ini,e,theta_star     	
      read(unit=5,fmt=*)a_spin,theta_ini,r_ini,theta_star 

      If(theta_ini.ne.90.D0)then 
          cos_ini = cos(theta_ini*dtor)
          sin_ini = sin(theta_ini*dtor)
      else
          cos_ini = zero
          sin_ini = one 
      endif
 
      e = -0.D0   
      IF(e*e+a_spin*a_spin.ge.one)Then
      write(*,*)'e^2+a^2>=1, there can be no horizon. wrong and stoped.'
          stop
      ENDIF

      CALL pmcharges(sin_ini,cos_ini,a_spin,e,r_ini,theta_star)
      END
!***************************************************************************************************** 
      SUBROUTINE pmcharges(sin_ini,cos_ini,a_spin,e,r_ini,theta_star)
!*****************************************************************************************************
      use blcoordinates
      implicit none
      Double precision sin_ini,cos_ini,a_spin,e,r_ini,somega,expnu,exppsi,expmu1,expmu2,&
             vptl(1:3),velocity(1:3),velo,rhorizon,lambda,q,mve,ep,kvec(1:4),pmax,&
             ra,mua,timea,phya,sigmaa,p,theta,theta_star,charge_sign
      integer :: m,n,i,j,k  
      logical :: cir_orbt

      cir_orbt = .TRUE.
      charge_sign = one
 
      rhorizon=one+sqrt(one-a_spin**two-e*e)   
 
      open(unit=15,file='rayx1.txt',status="replace")
      open(unit=16,file='rayy1.txt',status="replace")  	
      open(unit=17,file='rayz1.txt',status="replace")  
  
      call spherical_motion_constants(r_ini,theta_star,a_spin,e,charge_sign,lambda,q,mve,ep)
      kvec(1) = zero
      kvec(2) = one
      kvec(3) = one
      kvec(4) = one
      !pmax = p_total(kvec,lambda,q,mve,ep,sin_ini,cos_ini,a_spin,e,r_ini)  

      Do k=0,20000
          p=k*0.006D0     
          CALL YNOGKM(p,kvec,lambda,q,mve,ep,sin_ini,cos_ini,a_spin,e,r_ini,&
                                          ra,mua,timea,phya,sigmaa,cir_orbt,theta_star) 
  
          write(unit=15,fmt=*)sqrt(ra**two+a_spin**two)*sqrt(one-mua**two)*cos(phya)
          write(unit=16,fmt=*)sqrt(ra**two+a_spin**two)*sqrt(one-mua**two)*sin(phya)
          write(unit=17,fmt=*)ra*mua 
      Enddo 

      close(unit=15)	
      close(unit=16)
      close(unit=17) 
      return
      end subroutine pmcharges










